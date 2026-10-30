package googleadk

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2aclient/agentcard"
	"github.com/a2aproject/a2a-go/v2/errordetails"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	remoteagent "google.golang.org/adk/v2/agent/remoteagent/v2"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/server/adka2a/v2"
)

// a2aCancelTimeout bounds the cancellation ADK sends for a remote task a run
// leaves unfinished, as ADK's own cleanup does.
const a2aCancelTimeout = 5 * time.Second

// NewA2AAgentCardProvider returns an AgentCardProvider for ADK's remote A2A
// agent (remoteagent.NewA2A) that fetches the agent card from source, an
// http(s) URL or a file path on the worker, in the ResolveA2AAgentCard
// Activity. Use it with NewA2AClientProvider:
//
//	remote, err := remoteagent.NewA2A(remoteagent.A2AConfig{
//		Name:              "remote",
//		AgentCardProvider: googleadk.NewA2AAgentCardProvider(cardURL, workflow.ActivityOptions{}),
//		ClientProvider: googleadk.NewA2AClientProvider(workflow.ActivityOptions{
//			StartToCloseTimeout: 10 * time.Minute,
//		}),
//	})
//
// A zero StartToCloseTimeout in ao falls back to one minute, and with no
// RetryPolicy Temporal retries a failed fetch without limit, including a card
// file missing on the worker, which another worker or a redeploy can supply.
func NewA2AAgentCardProvider(source string, ao workflow.ActivityOptions) remoteagent.AgentCardProvider {
	// A method value, not a func literal, so remoteAgentSendsFromWorkflow can
	// recognize it: a func literal gets a new code pointer when inlined.
	return a2aCardProvider{source: source, activityOptions: ao}.resolve
}

type a2aCardProvider struct {
	source          string
	activityOptions workflow.ActivityOptions
}

func (p a2aCardProvider) resolve(ctx context.Context) (*a2a.AgentCard, error) {
	wfCtx, ok := workflowContext(ctx)
	if !ok {
		return nil, errMissingContext
	}
	var card a2a.AgentCard
	if err := executeA2AActivity(wfCtx, p.activityOptions, ResolveA2AAgentCardActivityName, invocationAgentName(ctx), p.source, &card); err != nil {
		return nil, err
	}
	return &card, nil
}

// NewA2AClientProvider returns a ClientProvider for ADK's remote A2A agent
// (remoteagent.NewA2A) whose client sends each message in the SendA2AMessage
// Activity and each cancellation in CancelA2ATask.
//
// A zero StartToCloseTimeout in ao falls back to one minute, and with no
// RetryPolicy Temporal retries without limit. A send that times out is sent
// again and starts another remote task, so set a StartToCloseTimeout longer
// than the remote agent takes to answer. A cancellation gets a single
// 5-second attempt, as in ADK; the rest of ao applies to it.
//
// The client sets the message ID from the workflow's random stream when it
// sends. ADK's BeforeRequestCallbacks run earlier and see the random ID a2a-go
// gave the message, which differs on every replay.
func NewA2AClientProvider(ao workflow.ActivityOptions) remoteagent.A2AClientProvider {
	// A method value, for the same reason as in NewA2AAgentCardProvider.
	return a2aClientProvider{activityOptions: ao}.create
}

type a2aClientProvider struct {
	activityOptions workflow.ActivityOptions
}

func (p a2aClientProvider) create(ctx context.Context, card *a2a.AgentCard) (remoteagent.A2AClient, error) {
	return &a2aClient{card: card, agentName: invocationAgentName(ctx), activityOptions: p.activityOptions}, nil
}

// a2aClient is the workflow-side client NewA2AClientProvider builds. Each call
// runs an Activity; the live a2a-go client exists only worker-side.
type a2aClient struct {
	card            *a2a.AgentCard
	agentName       string
	activityOptions workflow.ActivityOptions
}

func (c *a2aClient) SendMessage(ctx context.Context, req *a2a.SendMessageRequest) (a2a.SendMessageResult, error) {
	res, err := c.send(ctx, req, false)
	if err != nil {
		return nil, err
	}
	if len(res.Events) == 1 {
		if reply, ok := res.Events[0].Event.(a2a.SendMessageResult); ok {
			return reply, nil
		}
	}
	return nil, errors.New("googleadk: SendA2AMessage returned no message or task")
}

// SendStreamingMessage sends the message in one Activity and yields its events
// once the stream ends, followed by the error that ended it, if any.
func (c *a2aClient) SendStreamingMessage(ctx context.Context, req *a2a.SendMessageRequest) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		res, err := c.send(ctx, req, true)
		if err != nil {
			yield(nil, err)
			return
		}
		for _, e := range res.Events {
			if !yield(e.Event, nil) {
				return
			}
		}
		if res.Error != "" {
			yield(nil, newApplicationError(ErrorTypeA2A, false, nil, "%s", res.Error))
		}
	}
}

func (c *a2aClient) send(ctx context.Context, req *a2a.SendMessageRequest, stream bool) (a2aSendResult, error) {
	wfCtx, ok := workflowContext(ctx)
	if !ok {
		return a2aSendResult{}, errMissingContext
	}
	// ADK stores the request in the turn's last event, so a replay must
	// rebuild the same message ID.
	if req.Message != nil {
		req.Message.ID = platform.NewUUID(ctx)
	}
	var res a2aSendResult
	in := a2aSendInput{Card: c.card, Request: req, Stream: stream}
	err := executeA2AActivity(wfCtx, c.activityOptions, SendA2AMessageActivityName, c.agentName, in, &res)
	return res, err
}

// CancelTask runs on a disconnected context, so it goes out even after the
// workflow is cancelled: ADK sends it on a context without cancellation.
func (c *a2aClient) CancelTask(ctx context.Context, req *a2a.CancelTaskRequest) (*a2a.Task, error) {
	wfCtx, ok := workflowContext(ctx)
	if !ok {
		return nil, errMissingContext
	}
	wfCtx, _ = workflow.NewDisconnectedContext(wfCtx)
	ao := c.activityOptions
	ao.StartToCloseTimeout = a2aCancelTimeout
	ao.RetryPolicy = &temporal.RetryPolicy{MaximumAttempts: 1}
	var task a2a.Task
	in := a2aCancelInput{Card: c.card, Request: req}
	if err := executeA2AActivity(wfCtx, ao, CancelA2ATaskActivityName, c.agentName, in, &task); err != nil {
		return nil, err
	}
	return &task, nil
}

// Destroy does nothing: the workflow-side client holds no connection.
func (c *a2aClient) Destroy() error { return nil }

// executeA2AActivity runs an A2A Activity, with the agent's name as its
// summary, and decodes its result into out. A zero StartToCloseTimeout in ao
// falls back to the one-minute tool default.
func executeA2AActivity(wfCtx workflow.Context, ao workflow.ActivityOptions, name, agentName string, in, out any) error {
	ao = resolveToolActivityOptions(ao, "")
	ao.Summary = agentName
	return workflow.ExecuteActivity(workflow.WithActivityOptions(wfCtx, ao), name, in).Get(wfCtx, out)
}

// invocationAgentName returns the name of the agent ctx runs, or "".
func invocationAgentName(ctx context.Context) string {
	if ic, ok := ctx.(agent.InvocationContext); ok && ic.Agent() != nil {
		return ic.Agent().Name()
	}
	return ""
}

// a2aSendInput is the serializable payload for SendA2AMessage.
type a2aSendInput struct {
	Card    *a2a.AgentCard
	Request *a2a.SendMessageRequest
	Stream  bool
}

// a2aSendResult is the serializable result of SendA2AMessage.
type a2aSendResult struct {
	// Events is the reply: one message or task, or the streamed events.
	Events []a2a.StreamResponse
	// Error is the error that ended the stream after Events. It comes back
	// with the events instead of failing the Activity because the remote agent
	// has started the task: a retry would send the message again and start a
	// second one. ADK keeps the events, turns the error into an error event and
	// cancels the task, as without Temporal.
	Error string
}

// a2aCancelInput is the serializable payload for CancelA2ATask.
type a2aCancelInput struct {
	Card    *a2a.AgentCard
	Request *a2a.CancelTaskRequest
}

// ResolveA2AAgentCard fetches the agent card from source (an http(s) URL or a
// file path) with remoteagent.NewAgentCardProvider.
func (a *Activities) ResolveA2AAgentCard(ctx context.Context, source string) (*a2a.AgentCard, error) {
	stopHeartbeats := startPeriodicHeartbeats(ctx)
	defer stopHeartbeats()
	card, err := remoteagent.NewAgentCardProvider(source)(ctx)
	if err != nil {
		return nil, a2aError("resolve A2A agent card", err)
	}
	return card, nil
}

// SendA2AMessage sends a message to a remote A2A agent and returns its reply.
// A streamed reply comes back whole, with the chunks of each artifact merged.
func (a *Activities) SendA2AMessage(ctx context.Context, in a2aSendInput) (a2aSendResult, error) {
	return withA2AClient(ctx, a, in.Card, "send A2A message", func(client *a2aclient.Client) (a2aSendResult, error) {
		if !in.Stream {
			reply, err := client.SendMessage(ctx, in.Request)
			if err != nil {
				return a2aSendResult{}, err
			}
			return a2aSendResult{Events: []a2a.StreamResponse{{Event: reply}}}, nil
		}
		var res a2aSendResult
		answered := false
		for ev, err := range client.SendStreamingMessage(ctx, in.Request) {
			// The attempt fails, and Temporal retries it, when the stream
			// fails before the remote agent answers, or when this worker ended
			// the call: it stopped, or the Activity was paused, reset or timed
			// out.
			if err != nil && (!answered || ctx.Err() != nil) {
				return a2aSendResult{}, err
			}
			if err != nil {
				res.Error = fmt.Sprintf("send A2A message: %v", err)
				return res, nil
			}
			answered = true
			res.Events = appendA2AEvent(res.Events, ev)
		}
		return res, nil
	})
}

// CancelA2ATask cancels a remote A2A task.
func (a *Activities) CancelA2ATask(ctx context.Context, in a2aCancelInput) (*a2a.Task, error) {
	return withA2AClient(ctx, a, in.Card, "cancel A2A task", func(client *a2aclient.Client) (*a2a.Task, error) {
		return client.CancelTask(ctx, in.Request)
	})
}

// withA2AClient runs call with an A2A client for card, heartbeating while it
// runs, and tags its error with op.
func withA2AClient[T any](ctx context.Context, a *Activities, card *a2a.AgentCard, op string, call func(*a2aclient.Client) (T, error)) (T, error) {
	stopHeartbeats := startPeriodicHeartbeats(ctx)
	defer stopHeartbeats()
	var zero T
	factory := a.a2aClientFactory
	if factory == nil {
		factory = defaultA2AClientFactory
	}
	client, err := factory.CreateFromCard(ctx, card)
	if err != nil {
		return zero, a2aClientError(err)
	}
	defer func() {
		if err := client.Destroy(); err != nil {
			activity.GetLogger(ctx).Warn("failed to destroy A2A client", "error", err)
		}
	}()
	res, err := call(client)
	if err != nil {
		return zero, a2aError(op, err)
	}
	return res, nil
}

// a2aClientError tags a failure to create an A2A client. A failure a2a-go
// does not wrap comes from the card or the factory, such as a card that names
// no transport the factory supports, which a retry cannot change. Otherwise
// a2a-go joins the errors of every transport it tried to open, and the call is
// retried when any of them can pass, as a2aRetryable decides, such as a
// caller's transport that could not dial or load its credentials. An interface
// URL that does not parse comes from the card and cannot pass.
func a2aClientError(err error) error {
	retryable := false
	if errors.Unwrap(err) != nil {
		failures := []error{err}
		var joined interface{ Unwrap() []error }
		if errors.As(err, &joined) {
			failures = joined.Unwrap()
		}
		for _, f := range failures {
			var urlErr *url.Error
			if !(errors.As(f, &urlErr) && urlErr.Op == "parse") && a2aRetryable(f) {
				retryable = true
			}
		}
	}
	return newApplicationError(ErrorTypeA2A, retryable, err, "create A2A client: %v", err)
}

// appendA2AEvent appends a streamed event to events. A chunk that continues
// the artifact of the event before it is merged into that event, so the result
// grows with the reply rather than with the number of chunks. ADK joins the
// chunks of an artifact into one event either way.
//
// An ADK server marks its events with ADK's partial flag, and ADK passes them
// on one by one, so they are not merged. The partial ones are dropped: ADK
// saves none of them, and the server sends their content again in a final
// event.
func appendA2AEvent(events []a2a.StreamResponse, ev a2a.Event) []a2a.StreamResponse {
	chunk, ok := ev.(*a2a.TaskArtifactUpdateEvent)
	if ok && adka2a.IsPartial(chunk.Metadata) {
		return events
	}
	if ok && chunk.Append && chunk.Artifact != nil && len(events) > 0 && !adka2a.IsPartialFlagSet(chunk.Metadata) {
		last, ok := events[len(events)-1].Event.(*a2a.TaskArtifactUpdateEvent)
		if ok && !last.LastChunk && last.Artifact != nil && last.TaskID == chunk.TaskID &&
			last.Artifact.ID == chunk.Artifact.ID &&
			reflect.DeepEqual(last.Metadata, chunk.Metadata) &&
			reflect.DeepEqual(last.Artifact.Metadata, chunk.Artifact.Metadata) {
			for _, p := range chunk.Artifact.Parts {
				last.Artifact.Parts = appendA2APart(last.Artifact.Parts, p)
			}
			last.LastChunk = chunk.LastChunk
			return events
		}
	}
	return append(events, a2a.StreamResponse{Event: ev})
}

// appendA2APart appends p to parts, joining it to a last text part of the same
// media type when neither carries metadata.
func appendA2APart(parts a2a.ContentParts, p *a2a.Part) a2a.ContentParts {
	if n := len(parts); n > 0 && p != nil && parts[n-1] != nil {
		last := parts[n-1]
		lastText, ok1 := last.Content.(a2a.Text)
		text, ok2 := p.Content.(a2a.Text)
		if ok1 && ok2 && last.Filename == p.Filename && last.MediaType == p.MediaType &&
			len(last.Metadata) == 0 && len(p.Metadata) == 0 {
			parts[n-1] = &a2a.Part{Content: lastText + text, Filename: last.Filename, MediaType: last.MediaType}
			return parts
		}
	}
	return append(parts, p)
}

// a2aError tags a failed A2A call and decides whether Temporal retries it:
//   - a call that failed with an HTTP status is retried for 408, 409, 429 and
//     5xx, as a model call is;
//   - an A2A error the remote agent answers with is not: sending again would
//     start another task, and ADK turns the failed call into an error event,
//     as without Temporal;
//   - a card source ADK cannot read, or a reply that is not valid JSON, is
//     not;
//   - anything else, such as a network error or a card file missing on this
//     worker, is.
func a2aError(op string, err error) error {
	return newApplicationError(ErrorTypeA2A, a2aRetryable(err), err, "%s: %v", op, err)
}

// a2aRetryable reports whether a2aError retries err.
func a2aRetryable(err error) bool {
	retryable := true
	var status *a2aHTTPStatusError
	var cardStatus *agentcard.ErrStatusNotOK
	var remote *a2a.Error
	var syntaxErr *json.SyntaxError
	var typeErr *json.UnmarshalTypeError
	switch {
	case errors.As(err, &status):
		retryable = retryableHTTPStatus(status.code)
	case errors.As(err, &cardStatus):
		retryable = retryableHTTPStatus(cardStatus.StatusCode)
	case errors.As(err, &remote), errors.Is(err, remoteagent.ErrUnsupportedCardSource),
		errors.As(err, &syntaxErr), errors.As(err, &typeErr):
		retryable = false
	}
	return retryable
}

// NewA2AClientFactory returns the factory the A2A Activities use by default,
// with opts added, for Config.A2AClientFactory. Use it to add options such as
// call interceptors that add credentials, while keeping the default
// transports. An option that sets a transport replaces the default one for its
// protocol.
func NewA2AClientFactory(opts ...a2aclient.FactoryOption) *a2aclient.Factory {
	return a2aclient.WithAdditionalOptions(defaultA2AClientFactory, opts...)
}

// defaultA2AClientFactory builds the A2A clients when Config.A2AClientFactory
// is nil: a2a-go's JSON-RPC and REST transports over HTTP clients with no
// timeout of their own, so the Activity's timeouts bound each call.
var defaultA2AClientFactory = a2aclient.NewFactory(
	a2aclient.WithJSONRPCTransport(&http.Client{Transport: a2aStatusTransport{}}),
	a2aclient.WithRESTTransport(&http.Client{Transport: a2aStatusTransport{keepA2AErrors: true}}),
)

// a2aStatusTransport turns an HTTP error status into an a2aHTTPStatusError, so
// a2aError can decide retries from the status. a2a-go's JSON-RPC transport
// reports the status only as text, and its REST transport turns any error body
// into an A2A error: a JSON one with no A2A reason becomes an internal error,
// which is not retried, and any other one a server error. A redirect is left
// to the http.Client, which follows it. With keepA2AErrors, an error body that
// names an A2A error, as a2a-go's REST server writes it, is left to a2a-go.
type a2aStatusTransport struct {
	keepA2AErrors bool
}

func (t a2aStatusTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := http.DefaultTransport.RoundTrip(req)
	if err != nil || resp.StatusCode == http.StatusOK || isFollowedRedirect(resp) {
		return resp, err
	}
	var body []byte
	if t.keepA2AErrors && strings.HasPrefix(resp.Header.Get("Content-Type"), "application/json") {
		// The whole body, as a2a-go would decode it: a prefix of an A2A
		// error does not parse, and would be taken for a gateway's.
		body, err = io.ReadAll(resp.Body)
	}
	_ = resp.Body.Close()
	if err == nil && isA2AErrorBody(body) {
		resp.Body = io.NopCloser(bytes.NewReader(body))
		return resp, nil
	}
	return nil, &a2aHTTPStatusError{code: resp.StatusCode, status: resp.Status}
}

// isFollowedRedirect reports whether http.Client follows resp to its Location.
func isFollowedRedirect(resp *http.Response) bool {
	switch resp.StatusCode {
	case http.StatusMovedPermanently, http.StatusFound, http.StatusSeeOther,
		http.StatusTemporaryRedirect, http.StatusPermanentRedirect:
		return resp.Header.Get("Location") != ""
	}
	return false
}

// isA2AErrorBody reports whether body is a google.rpc.Status error with an
// ErrorInfo detail in the A2A domain, which a2a-go's REST server adds to every
// error it writes. A gateway's JSON error has none. Like a2a-go, it reads a
// detail's fields only once its type says it is an ErrorInfo: an agent's own
// details can hold any JSON.
func isA2AErrorBody(body []byte) bool {
	var status struct {
		Error struct {
			Details []json.RawMessage `json:"details"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &status) != nil {
		return false
	}
	for _, raw := range status.Error.Details {
		var hint struct {
			Type string `json:"@type"`
		}
		if json.Unmarshal(raw, &hint) != nil || hint.Type != errordetails.ErrorInfoType {
			continue
		}
		var info struct {
			Domain string `json:"domain"`
		}
		if json.Unmarshal(raw, &info) == nil && info.Domain == a2a.ProtocolDomain {
			return true
		}
	}
	return false
}

// a2aHTTPStatusError is an HTTP error status from a remote A2A agent.
type a2aHTTPStatusError struct {
	code   int
	status string
}

func (e *a2aHTTPStatusError) Error() string { return "unexpected HTTP status: " + e.status }

// NewRunnerPlugin returns an ADK runner plugin that fails each run, before
// any agent runs, when root or an agent below it is a remote A2A agent that
// sends its requests from Workflow code. Add it to the runner of root:
//
//	adkPlugin, err := googleadk.NewRunnerPlugin(root)
//	// ...
//	r, err := runner.New(runner.Config{
//		Agent: root,
//		// ...
//		PluginConfig: runner.PluginConfig{Plugins: []*plugin.Plugin{adkPlugin}},
//	})
//
// Without it, only model calls check, and only the agents below the agent that
// makes the call. The runner plugin also catches a remote root agent, one
// under a sequential, loop or parallel agent, and one reached by transfer. It
// does not see a remote agent wrapped as a tool with agenttool, which is not
// part of the tree.
func NewRunnerPlugin(root agent.Agent) (*plugin.Plugin, error) {
	return plugin.New(plugin.Config{
		Name: "googleadk",
		BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
			remote := root
			if !remoteAgentSendsFromWorkflow(root) {
				remote = findRemoteAgent(root)
			}
			if remote != nil {
				return nil, remoteAgentError(remote)
			}
			return nil, nil
		},
	})
}
