package googleadk

import (
	"context"
	"errors"
	"iter"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/workflow"

	remoteagent "google.golang.org/adk/v2/agent/remoteagent/v2"
	"google.golang.org/adk/v2/platform"
)

// NewA2AAgentCardProvider returns an AgentCardProvider for ADK's remote A2A
// agent (remoteagent.NewA2A) that fetches the agent card in the
// ResolveA2AAgentCard Activity, so a replay reads the card from History
// instead of fetching it again. source is an http(s) URL or a file path on the
// worker, as for remoteagent.NewAgentCardProvider. A zero StartToCloseTimeout
// in ao falls back to the default one-minute tool timeout.
//
// Use it with NewA2AClientProvider:
//
//	remote, err := remoteagent.NewA2A(remoteagent.A2AConfig{
//	    Name:              "remote",
//	    AgentCardProvider: googleadk.NewA2AAgentCardProvider(cardURL, workflow.ActivityOptions{}),
//	    ClientProvider:    googleadk.NewA2AClientProvider(workflow.ActivityOptions{}),
//	})
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
	actx := workflow.WithActivityOptions(wfCtx, resolveToolActivityOptions(p.activityOptions, ""))
	var card a2a.AgentCard
	if err := workflow.ExecuteActivity(actx, ResolveA2AAgentCardActivityName, p.source).Get(wfCtx, &card); err != nil {
		return nil, err
	}
	return &card, nil
}

// NewA2AClientProvider returns a ClientProvider for ADK's remote A2A agent
// (remoteagent.NewA2A) whose client sends each request in an Activity:
// SendA2AMessage for messages and CancelA2ATask for cancellations. History
// records the replies, so a replay does not send the requests again. The
// Activities use a2a-go's default client. A zero StartToCloseTimeout in ao
// falls back to the default one-minute tool timeout; raise it for remote
// agents that take longer to answer.
func NewA2AClientProvider(ao workflow.ActivityOptions) remoteagent.A2AClientProvider {
	// A method value, for the same reason as in NewA2AAgentCardProvider.
	return a2aClientProvider{activityOptions: ao}.create
}

type a2aClientProvider struct {
	activityOptions workflow.ActivityOptions
}

func (p a2aClientProvider) create(_ context.Context, card *a2a.AgentCard) (remoteagent.A2AClient, error) {
	return &a2aClient{card: card, activityOptions: p.activityOptions}, nil
}

// a2aClient is the workflow-side client NewA2AClientProvider builds. Each call
// runs an Activity; the live a2a-go client exists only worker-side.
type a2aClient struct {
	card            *a2a.AgentCard
	activityOptions workflow.ActivityOptions
}

func (c *a2aClient) SendMessage(ctx context.Context, req *a2a.SendMessageRequest) (a2a.SendMessageResult, error) {
	events, err := c.send(ctx, req, false)
	if err != nil {
		return nil, err
	}
	if len(events) == 0 {
		return nil, errors.New("googleadk: SendA2AMessage returned no result")
	}
	res, ok := events[0].Event.(a2a.SendMessageResult)
	if !ok {
		return nil, errors.New("googleadk: SendA2AMessage returned an event that is not a message or task")
	}
	return res, nil
}

// SendStreamingMessage sends the message in one Activity and yields its events
// once the stream ends, so the workflow sees the same events on every replay.
func (c *a2aClient) SendStreamingMessage(ctx context.Context, req *a2a.SendMessageRequest) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		events, err := c.send(ctx, req, true)
		if err != nil {
			yield(nil, err)
			return
		}
		for _, e := range events {
			if !yield(e.Event, nil) {
				return
			}
		}
	}
}

func (c *a2aClient) send(ctx context.Context, req *a2a.SendMessageRequest, stream bool) ([]a2a.StreamResponse, error) {
	wfCtx, ok := workflowContext(ctx)
	if !ok {
		return nil, errMissingContext
	}
	// a2a-go gives the message a random ID in Workflow code, and ADK stores
	// the request in the turn's last event. Use an ID from the workflow's
	// random stream so a replay rebuilds the same events.
	if req.Message != nil {
		req.Message.ID = platform.NewUUID(ctx)
	}
	actx := workflow.WithActivityOptions(wfCtx, resolveToolActivityOptions(c.activityOptions, ""))
	var events []a2a.StreamResponse
	in := a2aSendInput{Card: c.card, Request: req, Stream: stream}
	if err := workflow.ExecuteActivity(actx, SendA2AMessageActivityName, in).Get(wfCtx, &events); err != nil {
		return nil, err
	}
	return events, nil
}

func (c *a2aClient) CancelTask(ctx context.Context, req *a2a.CancelTaskRequest) (*a2a.Task, error) {
	wfCtx, ok := workflowContext(ctx)
	if !ok {
		return nil, errMissingContext
	}
	actx := workflow.WithActivityOptions(wfCtx, resolveToolActivityOptions(c.activityOptions, ""))
	var task a2a.Task
	in := a2aCancelInput{Card: c.card, Request: req}
	if err := workflow.ExecuteActivity(actx, CancelA2ATaskActivityName, in).Get(wfCtx, &task); err != nil {
		return nil, err
	}
	return &task, nil
}

// Destroy does nothing: the workflow-side client holds no connection.
func (c *a2aClient) Destroy() error { return nil }

// a2aSendInput is the serializable payload for SendA2AMessage.
type a2aSendInput struct {
	Card    *a2a.AgentCard
	Request *a2a.SendMessageRequest
	Stream  bool
}

// a2aCancelInput is the serializable payload for CancelA2ATask.
type a2aCancelInput struct {
	Card    *a2a.AgentCard
	Request *a2a.CancelTaskRequest
}

// ResolveA2AAgentCard fetches the agent card from source (an http(s) URL or a
// file path) with remoteagent.NewAgentCardProvider.
func (a *Activities) ResolveA2AAgentCard(ctx context.Context, source string) (*a2a.AgentCard, error) {
	card, err := remoteagent.NewAgentCardProvider(source)(ctx)
	if err != nil {
		retryable := !errors.Is(err, remoteagent.ErrUnsupportedCardSource)
		return nil, newApplicationError(ErrorTypeA2A, retryable, err, "resolve A2A agent card: %v", err)
	}
	return card, nil
}

// SendA2AMessage sends a message to a remote A2A agent with a2a-go's default
// client and returns its reply: one message or task, or every streamed event
// when Stream is set.
func (a *Activities) SendA2AMessage(ctx context.Context, in a2aSendInput) ([]a2a.StreamResponse, error) {
	// No-op unless the caller set a HeartbeatTimeout; then a slow remote agent
	// must not be heartbeat-timed-out and sent the message again.
	stopHeartbeats := startPeriodicHeartbeats(ctx)
	defer stopHeartbeats()
	client, err := newA2AClient(ctx, in.Card)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err := client.Destroy(); err != nil {
			activity.GetLogger(ctx).Warn("failed to destroy A2A client", "error", err)
		}
	}()
	if !in.Stream {
		res, err := client.SendMessage(ctx, in.Request)
		if err != nil {
			return nil, a2aError("send A2A message", err)
		}
		return []a2a.StreamResponse{{Event: res}}, nil
	}
	var events []a2a.StreamResponse
	for ev, err := range client.SendStreamingMessage(ctx, in.Request) {
		if err != nil {
			return nil, a2aError("send A2A message", err)
		}
		events = append(events, a2a.StreamResponse{Event: ev})
	}
	return events, nil
}

// CancelA2ATask cancels a remote A2A task with a2a-go's default client.
func (a *Activities) CancelA2ATask(ctx context.Context, in a2aCancelInput) (*a2a.Task, error) {
	client, err := newA2AClient(ctx, in.Card)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err := client.Destroy(); err != nil {
			activity.GetLogger(ctx).Warn("failed to destroy A2A client", "error", err)
		}
	}()
	task, err := client.CancelTask(ctx, in.Request)
	if err != nil {
		return nil, a2aError("cancel A2A task", err)
	}
	return task, nil
}

// newA2AClient builds a2a-go's default client for card. A failure comes from
// the card, such as one with no supported transport, so it is non-retryable.
func newA2AClient(ctx context.Context, card *a2a.AgentCard) (*a2aclient.Client, error) {
	client, err := a2aclient.NewFromCard(ctx, card)
	if err != nil {
		return nil, newApplicationError(ErrorTypeA2A, false, err, "create A2A client: %v", err)
	}
	return client, nil
}

// a2aError tags an A2A call failure. An A2A error the remote agent answered
// with, such as a task it does not know, is non-retryable: sending again
// cannot fix it, and ADK turns the failed call into an error event, as it does
// without Temporal. Internal and server errors, and errors that never reached
// the agent, stay retryable.
func a2aError(op string, err error) error {
	var remote *a2a.Error
	retryable := !errors.As(err, &remote) || errors.Is(err, a2a.ErrInternalError) || errors.Is(err, a2a.ErrServerError)
	return newApplicationError(ErrorTypeA2A, retryable, err, "%s: %v", op, err)
}
