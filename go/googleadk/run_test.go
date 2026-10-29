package googleadk_test

import (
	"context"
	"errors"
	"iter"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
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
// refuses fails the SendA2AMessage Activity once, without retries, and the
// turn ends as it would without Temporal: ADK turns the error into an event.
// The last case is the control for the HTTP 401 one: the caller's
// A2AClientFactory adds the credentials and the remote agent answers.
func TestRemoteSubAgentErrorNotRetried(t *testing.T) {
	for _, tc := range []struct {
		name      string
		server    a2aServer
		factory   *a2aclient.Factory
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
			name:     "credentials from the A2AClientFactory",
			server:   a2aServer{Reply: a2aReply(a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("remote answer")), nil), Token: "secret"},
			factory:  a2aclient.NewFactory(a2aclient.WithCallInterceptors(bearerInterceptor{token: "secret"})),
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

// TestRemoteSubAgentStreaming proves a streamed reply reaches ADK through one
// SendA2AMessage Activity, with the chunks of an artifact merged so the result
// grows with the reply rather than with the number of chunks, and that an
// error after the first events comes back with those events instead of a
// retry that would send the message again: ADK keeps them, records the error
// and cancels the unfinished task, as without Temporal.
func TestRemoteSubAgentStreaming(t *testing.T) {
	for _, tc := range []struct {
		name       string
		server     a2aServer
		wantTexts  []string
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
	} {
		t.Run(tc.name, func(t *testing.T) {
			url, _ := startA2AServer(t, tc.server)
			var s testsuite.WorkflowTestSuite
			env, counter := newEnv(t, &s, googleadk.Config{
				Models: map[string]googleadk.ModelFactory{"root-model": transferToRemoteModel()},
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
