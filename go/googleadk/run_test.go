package googleadk_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/a2aproject/a2a-go/v2/errordetails"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/server/adka2a/v2"
	"google.golang.org/adk/v2/tool"

	"go.temporal.io/googleadk"
)

// TestMultiAgentSubAgents drives a real two-agent tree: the root model emits
// transfer_to_agent (a control tool that runs in-workflow, never as an
// Activity), and the specialist sub-agent then produces the answer — its model
// call still a durable InvokeModel Activity.
func TestMultiAgentSubAgents(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env, counter := newEnv(t, &s, googleadk.Config{
		Models: map[string]googleadk.ModelFactory{
			"root-model": scriptedModelFactory(
				googleadk.FunctionCallResponse("c1", "transfer_to_agent", map[string]any{"agent_name": "specialist"}),
				googleadk.TextResponse("(root fallback)"),
			),
			"specialist-model": scriptedModelFactory(googleadk.TextResponse("specialist answer")),
		},
	})

	env.ExecuteWorkflow(agentRunWorkflow, runInput{
		ModelName:   "root-model",
		UserMessage: "please handle this specialist task",
		SubAgents: []subAgentSpec{{
			Name:        "specialist",
			Description: "handles specialist tasks",
			ModelName:   "specialist-model",
		}},
	})

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	var res runResult
	require.NoError(t, env.GetWorkflowResult(&res))
	assert.Contains(t, res.Texts, "specialist answer")
	assert.Contains(t, res.Authors, "specialist", "the specialist sub-agent should author an event")

	// Two durable model calls: the root turn that transfers, and the specialist.
	assert.GreaterOrEqual(t, counter.get(googleadk.InvokeModelActivityName), 2)
}

// TestRemoteSubAgentRejected proves a remote A2A agent that would send from
// Workflow code fails the run with a clear error before it sends anything: on
// the model call of an agent above it, or, with the runner plugin, right
// before it runs. Every replay of such an agent would send again.
func TestRemoteSubAgentRejected(t *testing.T) {
	workflowSide := func(url string) []subAgentSpec {
		sa := remoteSubAgent(url)
		sa.WorkflowSideA2A = true
		return []subAgentSpec{sa}
	}
	for _, tc := range []struct {
		name string
		in   func(url string) runInput
		// wantSent is set for the control: the remote agent runs and sends.
		wantSent bool
	}{
		{
			name: "below the agent making a model call",
			in: func(url string) runInput {
				return runInput{ModelName: "root-model", UserMessage: "ask the remote agent", SubAgents: workflowSide(url)}
			},
		},
		{
			name: "under a sequential agent, with the runner plugin",
			in: func(url string) runInput {
				return runInput{SequentialRoot: true, RunnerPlugin: true, UserMessage: "ask the remote agent", SubAgents: workflowSide(url)}
			},
		},
		{
			name: "under a sequential agent, without the runner plugin",
			in: func(url string) runInput {
				return runInput{SequentialRoot: true, UserMessage: "ask the remote agent", SubAgents: workflowSide(url)}
			},
			wantSent: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			url, requests := startA2AServer(t, a2aServer{
				Reply: a2aReply(a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("remote answer")), nil),
			})
			var s testsuite.WorkflowTestSuite
			env, counter := newEnv(t, &s, googleadk.Config{
				Models: map[string]googleadk.ModelFactory{"root-model": transferToRemoteModel()},
			})

			env.ExecuteWorkflow(agentRunWorkflow, tc.in(url))

			require.True(t, env.IsWorkflowCompleted())
			if tc.wantSent {
				require.NoError(t, env.GetWorkflowError())
				assert.Positive(t, requests.Load(), "without the runner plugin, the remote agent sends from Workflow code")
				return
			}
			assert.Zero(t, requests.Load(), "no A2A request may leave Workflow code")
			assert.Zero(t, counter.get(googleadk.InvokeModelActivityName), "the run must fail before the first model call")
			err := env.GetWorkflowError()
			require.Error(t, err, "a tree with a remote A2A agent must fail the run")
			assert.Contains(t, err.Error(), `agent "remote" is a remote A2A agent`)
		})
	}
}

// bearerInterceptor adds "Authorization: Bearer <token>" to every A2A call.
type bearerInterceptor struct {
	a2aclient.PassthroughInterceptor
	token string
}

func (i bearerInterceptor) Before(ctx context.Context, req *a2aclient.Request) (context.Context, any, error) {
	req.ServiceParams.Append("Authorization", "Bearer "+i.token)
	return ctx, nil, nil
}

// TestRemoteSubAgentErrorNotRetried proves an A2A call the remote agent
// refuses, or a client the card cannot make, fails the SendA2AMessage Activity
// once, without retries, and the turn ends as it would without Temporal: ADK
// turns the error into an event.
// The last case is the control for the HTTP 401 ones: the caller's
// A2AClientFactory adds the right credentials and the remote agent answers.
func TestRemoteSubAgentErrorNotRetried(t *testing.T) {
	for _, tc := range []struct {
		name      string
		server    a2aServer
		factory   *a2aclient.Factory
		sse       bool
		wantError string
		wantText  string
	}{
		{name: "A2A error", server: a2aServer{Reply: a2aReply(nil, a2a.ErrInvalidParams)}, wantError: "invalid params"},
		{name: "executor error", server: a2aServer{Reply: a2aReply(nil, errors.New("boom"))}, wantError: "internal error"},
		{
			name:      "HTTP 401",
			server:    a2aServer{Reply: a2aReply(a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("remote answer")), nil), Token: "secret"},
			wantError: "401 Unauthorized",
		},
		{
			name:      "REST gateway JSON error",
			server:    a2aServer{REST: true, Front: failA2ARequests(0, http.StatusBadRequest, "application/json", gatewayErrorBody(http.StatusBadRequest))},
			wantError: "400 Bad Request",
		},
		{
			name:      "REST A2A error in an HTTP error",
			server:    a2aServer{REST: true, Front: failA2ARequests(0, http.StatusBadRequest, "application/json", restA2AErrorBody(a2a.ErrInvalidParams))},
			wantError: "invalid params",
		},
		{
			// The media type the A2A spec recommends, with a status the
			// HTTP rule would retry.
			name:      "REST A2A error as application/a2a+json in an HTTP 500",
			server:    a2aServer{REST: true, Front: failA2ARequests(0, http.StatusInternalServerError, "application/a2a+json", restA2AErrorBody(a2a.ErrInvalidParams))},
			wantError: "invalid params",
		},
		{name: "REST A2A error in the stream", server: a2aServer{REST: true, Reply: a2aReply(nil, a2a.ErrInvalidParams)}, sse: true, wantError: "invalid params"},
		{
			name:      "REST A2A error, 1 KiB of details",
			server:    a2aServer{REST: true, Reply: a2aReply(nil, diagnosticA2AError(1<<10))},
			wantError: "agent failed",
		},
		{
			// The body is longer than any limit on reading it would be.
			name:      "REST A2A error, 2 MiB of details",
			server:    a2aServer{REST: true, Reply: a2aReply(nil, diagnosticA2AError(2<<20))},
			wantError: "agent failed",
		},
		{
			// Diagnostic details whose fields are not strings, beside the
			// A2A ErrorInfo.
			name:      "REST A2A error, object domain in details",
			server:    a2aServer{REST: true, Reply: a2aReply(nil, a2a.NewError(a2a.ErrInternalError, "remote rejected").WithDetails(map[string]any{"domain": map[string]any{"service": "remote"}}))},
			wantError: "remote rejected",
		},
		{
			name:      "REST A2A error, numeric domain in details",
			server:    a2aServer{REST: true, Reply: a2aReply(nil, a2a.NewError(a2a.ErrInternalError, "remote rejected").WithDetails(map[string]any{"domain": 17}))},
			wantError: "remote rejected",
		},
		{
			name:      "HTTP 307 without Location",
			server:    a2aServer{Front: failA2ARequests(0, http.StatusTemporaryRedirect, "text/plain", "moved")},
			wantError: "307 Temporary Redirect",
		},
		{
			name:      "card with no transport the factory supports",
			server:    a2aServer{Reply: a2aReply(a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("remote answer")), nil)},
			factory:   a2aclient.NewFactory(a2aclient.WithDefaultsDisabled(), a2aclient.WithRESTTransport(nil)),
			wantError: "no compatible transports",
		},
		{
			name:      "card with an interface URL that does not parse",
			server:    a2aServer{Reply: a2aReply(a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("remote answer")), nil), REST: true, Path: "/%zz"},
			wantError: "invalid URL escape",
		},
		{
			// JSON-RPC parses the URL on each call, not when the client is made.
			name:      "card with a JSON-RPC interface URL that does not parse",
			server:    a2aServer{Reply: a2aReply(a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("remote answer")), nil), Path: "/%zz"},
			wantError: "invalid URL escape",
		},
		{
			name:      "card whose interface URLs all do not parse",
			server:    a2aServer{Reply: a2aReply(a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("remote answer")), nil), REST: true, Path: "/%zz", AlsoRESTPath: "/%zy"},
			wantError: "invalid URL escape",
		},
		{
			// a2a-go's REST transport turns a JSON error body with no A2A
			// reason into an internal error, which is not retried, though a
			// retry would get the answer.
			name:      "REST gateway JSON 503 with a2a-go's factory",
			server:    a2aServer{Reply: a2aReply(a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("remote answer")), nil), REST: true, Front: failA2ARequests(1, http.StatusServiceUnavailable, "application/json", gatewayErrorBody(http.StatusServiceUnavailable))},
			factory:   a2aclient.NewFactory(),
			wantError: "internal error",
		},
		{
			name:      "wrong credentials from the A2AClientFactory",
			server:    a2aServer{Reply: a2aReply(a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("remote answer")), nil), Token: "secret"},
			factory:   googleadk.NewA2AClientFactory(a2aclient.WithCallInterceptors(bearerInterceptor{token: "expired"})),
			wantError: "401 Unauthorized",
		},
		{
			name:     "credentials from the A2AClientFactory",
			server:   a2aServer{Reply: a2aReply(a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("remote answer")), nil), Token: "secret"},
			factory:  googleadk.NewA2AClientFactory(a2aclient.WithCallInterceptors(bearerInterceptor{token: "secret"})),
			wantText: "remote answer",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			url, _ := startA2AServer(t, tc.server)
			var s testsuite.WorkflowTestSuite
			env, counter := newEnv(t, &s, googleadk.Config{
				Models:           map[string]googleadk.ModelFactory{"root-model": transferToRemoteModel()},
				A2AClientFactory: tc.factory,
			})

			env.ExecuteWorkflow(agentRunWorkflow, runInput{
				ModelName:   "root-model",
				UserMessage: "ask the remote agent",
				SubAgents:   []subAgentSpec{remoteSubAgent(url)},
				SSE:         tc.sse,
			})

			require.True(t, env.IsWorkflowCompleted())
			require.NoError(t, env.GetWorkflowError())
			var res runResult
			require.NoError(t, env.GetWorkflowResult(&res))
			assert.Equal(t, 1, counter.get(googleadk.SendA2AMessageActivityName), "a refused A2A call must not be retried")
			if tc.wantError != "" {
				require.Len(t, res.Errors, 1, "ADK turns the failed call into an error event")
				assert.Contains(t, res.Errors[0], tc.wantError)
			} else {
				assert.Empty(t, res.Errors)
				assert.Contains(t, res.Texts, tc.wantText)
			}
		})
	}
}

// TestRemoteSubAgentHTTPDelivered proves the default A2A client follows a
// redirect, as a2a-go's own client does, and retries an HTTP status the
// retry rule retries, whatever the body: a gateway's JSON error is not an A2A
// error. The case with a2a-go's factory is the control for the redirects. A
// caller's transport that fails to open once is retried too, even when the
// card's other interface can never open; the cases where it does not fail are
// the controls.
func TestRemoteSubAgentHTTPDelivered(t *testing.T) {
	answer := a2aReply(a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("remote answer")), nil)
	for _, tc := range []struct {
		name      string
		server    a2aServer
		factory   *a2aclient.Factory
		wantSends int
	}{
		{
			name:      "JSON-RPC 307",
			server:    a2aServer{Reply: answer, Path: "/old/", Front: redirectA2ARequests(http.StatusTemporaryRedirect)},
			wantSends: 1,
		},
		{
			name:      "REST 308",
			server:    a2aServer{Reply: answer, REST: true, Path: "/old", Front: redirectA2ARequests(http.StatusPermanentRedirect)},
			wantSends: 1,
		},
		{
			name:      "JSON-RPC 307 with a2a-go's factory",
			server:    a2aServer{Reply: answer, Path: "/old/", Front: redirectA2ARequests(http.StatusTemporaryRedirect)},
			factory:   a2aclient.NewFactory(),
			wantSends: 1,
		},
		{
			name:      "REST gateway JSON 503",
			server:    a2aServer{Reply: answer, REST: true, Front: failA2ARequests(1, http.StatusServiceUnavailable, "application/json", gatewayErrorBody(http.StatusServiceUnavailable))},
			wantSends: 2,
		},
		{
			name:      "REST plaintext 503",
			server:    a2aServer{Reply: answer, REST: true, Front: failA2ARequests(1, http.StatusServiceUnavailable, "text/plain", "temporary overload")},
			wantSends: 2,
		},
		{
			name:      "REST gateway JSON 503, 2 MiB body",
			server:    a2aServer{Reply: answer, REST: true, Front: failA2ARequests(1, http.StatusServiceUnavailable, "application/json", longGatewayErrorBody(http.StatusServiceUnavailable, 2<<20))},
			wantSends: 2,
		},
		{
			name:      "JSON-RPC gateway JSON 503",
			server:    a2aServer{Reply: answer, Front: failA2ARequests(1, http.StatusServiceUnavailable, "application/json", gatewayErrorBody(http.StatusServiceUnavailable))},
			wantSends: 2,
		},
		{
			name:      "A2AClientFactory transport that fails to open once",
			server:    a2aServer{Reply: answer},
			factory:   flakyA2AFactory(1),
			wantSends: 2,
		},
		{
			name:      "A2AClientFactory transport that opens",
			server:    a2aServer{Reply: answer},
			factory:   flakyA2AFactory(),
			wantSends: 1,
		},
		{
			name:      "A2AClientFactory transport that fails to open once, other interface URL does not parse",
			server:    a2aServer{Reply: answer, AlsoRESTPath: "/%zz"},
			factory:   flakyA2AFactory(1),
			wantSends: 2,
		},
		{
			name:      "A2AClientFactory transport that opens, other interface URL does not parse",
			server:    a2aServer{Reply: answer, AlsoRESTPath: "/%zz"},
			factory:   flakyA2AFactory(),
			wantSends: 1,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			url, _ := startA2AServer(t, tc.server)
			var s testsuite.WorkflowTestSuite
			env, counter := newEnv(t, &s, googleadk.Config{
				Models:           map[string]googleadk.ModelFactory{"root-model": transferToRemoteModel()},
				A2AClientFactory: tc.factory,
			})

			env.ExecuteWorkflow(agentRunWorkflow, runInput{
				ModelName:   "root-model",
				UserMessage: "ask the remote agent",
				SubAgents:   []subAgentSpec{remoteSubAgent(url)},
			})

			require.True(t, env.IsWorkflowCompleted())
			require.NoError(t, env.GetWorkflowError())
			var res runResult
			require.NoError(t, env.GetWorkflowResult(&res))
			assert.Empty(t, res.Errors)
			assert.Contains(t, res.Texts, "remote answer")
			assert.Equal(t, tc.wantSends, counter.get(googleadk.SendA2AMessageActivityName))
		})
	}
}

// redirectA2ARequests answers the A2A requests under /old with a redirect of
// code to the same path without /old.
func redirectA2ARequests(code int) func(http.ResponseWriter, *http.Request) bool {
	return func(w http.ResponseWriter, r *http.Request) bool {
		rest, ok := strings.CutPrefix(r.URL.Path, "/old")
		if !ok {
			return false
		}
		if rest == "" {
			rest = "/"
		}
		http.Redirect(w, r, rest, code)
		return true
	}
}

// failA2ARequests answers the first n A2A requests, or all of them when n is
// 0, with status and body.
func failA2ARequests(n int32, status int, contentType, body string) func(http.ResponseWriter, *http.Request) bool {
	var seen atomic.Int32
	return func(w http.ResponseWriter, r *http.Request) bool {
		if n > 0 && seen.Add(1) > n {
			return false
		}
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
		return true
	}
}

// flakyA2AFactory returns an A2AClientFactory whose JSON-RPC transport fails
// to open on the calls in fail, counted from 1, as a caller's transport does
// when it cannot load its credentials for a moment.
func flakyA2AFactory(fail ...int32) *a2aclient.Factory {
	var calls atomic.Int32
	return a2aclient.NewFactory(a2aclient.WithTransport(a2a.TransportProtocolJSONRPC,
		a2aclient.TransportFactoryFn(func(_ context.Context, _ *a2a.AgentCard, iface *a2a.AgentInterface) (a2aclient.Transport, error) {
			if slices.Contains(fail, calls.Add(1)) {
				return nil, errors.New("load credentials: temporarily unavailable")
			}
			return a2aclient.NewJSONRPCTransport(iface.URL, nil), nil
		})))
}

// gatewayErrorBody is a google.rpc.Status error as a gateway writes it, with
// no A2A ErrorInfo.
func gatewayErrorBody(code int) string {
	return fmt.Sprintf(`{"error":{"code":%d,"status":"UNAVAILABLE","message":"temporary overload"}}`, code)
}

// longGatewayErrorBody is gatewayErrorBody with a message of n bytes.
func longGatewayErrorBody(code, n int) string {
	return fmt.Sprintf(`{"error":{"code":%d,"status":"UNAVAILABLE","message":%q}}`, code, strings.Repeat("x", n))
}

// diagnosticA2AError is an A2A internal error with n bytes of diagnostic
// details, which a2a-go's REST server writes into the error body.
func diagnosticA2AError(n int) error {
	return a2a.NewError(a2a.ErrInternalError, "agent failed").WithDetails(map[string]any{"trace": strings.Repeat("x", n)})
}

// restA2AErrorBody is err as a2a-go's REST server writes it before a stream
// starts: a google.rpc.Status error with an ErrorInfo in the A2A domain.
func restA2AErrorBody(err error) string {
	return fmt.Sprintf(`{"error":{"code":400,"status":"INVALID_ARGUMENT","message":%q,"details":[{"@type":%q,"reason":%q,"domain":%q}]}}`,
		err.Error(), errordetails.ErrorInfoType, a2a.ErrorReason(err), a2a.ProtocolDomain)
}

// TestRemoteSubAgentStreaming proves a streamed reply reaches ADK through one
// SendA2AMessage Activity, with the chunks of an artifact merged so the result
// grows with the reply rather than with the number of chunks, and an ADK
// server's final events kept apart and its partial ones dropped. It also
// proves an error after the first events comes back with those events instead
// of a retry that would send the message again: ADK keeps them, records the
// error and cancels the unfinished task, as without Temporal.
func TestRemoteSubAgentStreaming(t *testing.T) {
	for _, tc := range []struct {
		name       string
		server     a2aServer
		wantTexts  []string
		factory    *a2aclient.Factory
		wantError  string
		wantCancel int
	}{
		{
			name: "chunks merged",
			server: a2aServer{Reply: func(ec *a2asrv.ExecutorContext, yield func(a2a.Event, error) bool) {
				first := a2a.NewArtifactEvent(ec, a2a.NewTextPart("Hello"))
				last := a2a.NewArtifactUpdateEvent(ec, first.Artifact.ID, a2a.NewTextPart("!"))
				last.LastChunk = true
				for _, ev := range []a2a.Event{
					a2a.NewSubmittedTask(ec, ec.Message),
					first,
					a2a.NewArtifactUpdateEvent(ec, first.Artifact.ID, a2a.NewTextPart(", world")),
					last,
					a2a.NewStatusUpdateEvent(ec, a2a.TaskStateCompleted, nil),
				} {
					if !yield(ev, nil) {
						return
					}
				}
			}},
			// Unmerged, ADK would also yield the partial chunks "Hello",
			// ", world" and "!".
			wantTexts: []string{"Hello, world!"},
		},
		{
			name: "ADK server events",
			server: a2aServer{Reply: func(ec *a2asrv.ExecutorContext, yield func(a2a.Event, error) bool) {
				events := []a2a.Event{a2a.NewSubmittedTask(ec, ec.Message)}
				var partialID, finalID a2a.ArtifactID
				for _, step := range []string{"Step 1 done", "Step 2 done"} {
					for _, chunk := range []string{step[:4], step[4:]} {
						ev := adkArtifactEvent(ec, partialID, true, chunk)
						partialID = ev.Artifact.ID
						events = append(events, ev)
					}
					ev := adkArtifactEvent(ec, finalID, false, step)
					finalID = ev.Artifact.ID
					events = append(events, ev)
				}
				for _, ev := range append(events, a2a.NewStatusUpdateEvent(ec, a2a.TaskStateCompleted, nil)) {
					if !yield(ev, nil) {
						return
					}
				}
			}},
			// Merged, the final events would become "Step 1 doneStep 2 done";
			// kept, the partial chunks would add "Step", " 1 done", "Step" and
			// " 2 done".
			wantTexts: []string{"Step 1 done", "Step 2 done"},
		},
		{
			name: "error after events",
			server: a2aServer{
				Reply: func(ec *a2asrv.ExecutorContext, yield func(a2a.Event, error) bool) {
					_ = yield(a2a.NewSubmittedTask(ec, ec.Message), nil) &&
						yield(a2a.NewArtifactEvent(ec, a2a.NewTextPart("partial")), nil)
				},
				StreamErrorAfter: 2,
			},
			wantTexts:  []string{"partial"},
			wantError:  "send A2A message",
			wantCancel: 1,
		},
		{
			// The cancellation's transport fails to open, which the retry
			// rule retries, but cleanup still gets one attempt.
			name: "error after events, cancellation fails to open",
			server: a2aServer{
				Reply: func(ec *a2asrv.ExecutorContext, yield func(a2a.Event, error) bool) {
					_ = yield(a2a.NewSubmittedTask(ec, ec.Message), nil) &&
						yield(a2a.NewArtifactEvent(ec, a2a.NewTextPart("partial")), nil)
				},
				StreamErrorAfter: 2,
			},
			factory:    flakyA2AFactory(2, 3, 4, 5),
			wantTexts:  []string{"partial"},
			wantError:  "send A2A message",
			wantCancel: 1,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			url, _ := startA2AServer(t, tc.server)
			var s testsuite.WorkflowTestSuite
			env, counter := newEnv(t, &s, googleadk.Config{
				Models:           map[string]googleadk.ModelFactory{"root-model": transferToRemoteModel()},
				A2AClientFactory: tc.factory,
			})

			env.ExecuteWorkflow(agentRunWorkflow, runInput{
				ModelName:   "root-model",
				UserMessage: "ask the remote agent",
				SubAgents:   []subAgentSpec{remoteSubAgent(url)},
				SSE:         true,
			})

			require.True(t, env.IsWorkflowCompleted())
			require.NoError(t, env.GetWorkflowError())
			var res runResult
			require.NoError(t, env.GetWorkflowResult(&res))
			assert.Equal(t, 1, counter.get(googleadk.SendA2AMessageActivityName))
			assert.Equal(t, tc.wantTexts, res.Texts)
			if tc.wantError != "" {
				require.Len(t, res.Errors, 1)
				assert.Contains(t, res.Errors[0], tc.wantError)
			} else {
				assert.Empty(t, res.Errors)
			}
			assert.Equal(t, tc.wantCancel, counter.get(googleadk.CancelA2ATaskActivityName), "ADK cancels a task the run leaves unfinished")
		})
	}
}

// adkArtifactEvent is an artifact update as an ADK server sends it in its
// default output mode, with ADK's partial flag. The partial chunks of a run
// replace one artifact, and its final events are appended to another.
func adkArtifactEvent(ec *a2asrv.ExecutorContext, id a2a.ArtifactID, partial bool, text string) *a2a.TaskArtifactUpdateEvent {
	ev := a2a.NewArtifactEvent(ec, a2a.NewTextPart(text))
	if id != "" {
		ev = a2a.NewArtifactUpdateEvent(ec, id, a2a.NewTextPart(text))
		ev.Append = !partial
	}
	ev.Metadata = map[string]any{adka2a.ToA2AMetaKey("partial"): partial}
	return ev
}

// errorModel is a model.LLM that always fails with a genai.APIError of the given
// HTTP status, so InvokeModel classifies the failure per Temporal's retry
// contract.
type errorModel struct {
	name string
	code int
}

func (m *errorModel) Name() string { return m.name }

func (m *errorModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(nil, &genai.APIError{Code: m.code, Message: "boom"})
	}
}

// TestModelErrorClassification proves a non-retryable upstream status (HTTP 400)
// is surfaced as a non-retryable ApplicationError tagged ErrorTypeModel, so the
// run fails fast and the caller can classify it without string-matching.
func TestModelErrorClassification(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env, _ := newEnv(t, &s, googleadk.Config{
		Models: map[string]googleadk.ModelFactory{
			"broken-model": func(context.Context, string) (model.LLM, error) {
				return &errorModel{name: "broken-model", code: 400}, nil
			},
		},
	})

	env.ExecuteWorkflow(agentRunWorkflow, runInput{ModelName: "broken-model", UserMessage: "hi"})

	require.True(t, env.IsWorkflowCompleted())
	err := env.GetWorkflowError()
	require.Error(t, err, "a non-retryable model error must fail the run")
	assert.True(t, googleadk.IsNonRetryable(err), "HTTP 400 must be classified non-retryable")

	var appErr *temporal.ApplicationError
	require.True(t, errors.As(err, &appErr), "error must be a Temporal ApplicationError")
	assert.Equal(t, googleadk.ErrorTypeModel, appErr.Type())
}

// boomToolWorkflow wires an in-workflow function tool whose handler always
// returns an error, to prove ADK's tool-error contract: the error is handed back
// to the model rather than failing the workflow, and the model recovers on its
// next turn. (In-workflow tools are not dispatched through an Activity, so there
// is no Temporal retry; the error propagates straight into the agent loop.)
func boomToolWorkflow(ctx workflow.Context) (runResult, error) {
	boom, err := funcTool("boom", func(agent.Context, map[string]any) (map[string]any, error) {
		return nil, errors.New("transient boom")
	})
	if err != nil {
		return runResult{}, err
	}
	return runAgent(ctx, agentBuild{
		modelName:   "fake-model",
		userMessage: "call boom",
		tools:       []tool.Tool{boom},
	})
}

// TestToolErrorFedBackToModel proves an in-workflow tool failure is — per ADK's
// tool-error contract — handed back to the model rather than failing the
// workflow, so the model can recover on the following turn.
func TestToolErrorFedBackToModel(t *testing.T) {
	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()
	env.RegisterWorkflow(boomToolWorkflow)
	wireActivities(t, env, googleadk.Config{
		Models: map[string]googleadk.ModelFactory{
			"fake-model": scriptedModelFactory(
				googleadk.FunctionCallResponse("c1", "boom", map[string]any{}),
				googleadk.TextResponse("recovered after tool failure"),
			),
		},
	})

	env.ExecuteWorkflow(boomToolWorkflow)

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	var res runResult
	require.NoError(t, env.GetWorkflowResult(&res))
	assert.Contains(t, res.Texts, "recovered after tool failure")
	assert.Contains(t, res.FunctionCalls, "boom")
}
