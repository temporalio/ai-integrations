package googleadk_test

// End-to-end integration tests that need a real Temporal server: the streaming
// side channel (workflowstreams publishes chunks via a signal to the parent
// workflow, which a unit-test mock client cannot service) and history replay.
// They share a local dev server, downloading the CLI on first use, and FAIL
// when it cannot be started (no binary, no network), so a run that never
// exercised them cannot pass. The default unit suite already
// covers the durable agent loop, aggregation, determinism providers, and failure
// classification without a server.

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	enumspb "go.temporal.io/api/enums/v1"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/contrib/workflowstreams"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/worker"
	sdkworkflow "go.temporal.io/sdk/workflow"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"

	"go.temporal.io/googleadk"
)

// startWorker boots a real worker for the test workflows with the plugin
// Activities wired from cfg, plus the guarded-trio activities the
// multi-decision HITL workflow dispatches.
func startWorker(t *testing.T, c client.Client, taskQueue string, cfg googleadk.Config) worker.Worker {
	t.Helper()
	w := worker.New(c, taskQueue, worker.Options{})
	w.RegisterWorkflow(agentRunWorkflow)
	w.RegisterWorkflow(multiConfirmHitlWorkflow)
	registerGuardedTrio(w)
	acts, err := googleadk.NewActivities(cfg)
	require.NoError(t, err)
	acts.Register(w)
	require.NoError(t, w.Start())
	t.Cleanup(w.Stop)
	return w
}

// TestStreamingIntegration is the end-to-end streaming proof: with StreamingTopic
// set, the InvokeModel Activity drives the model in streaming mode and PUBLISHES
// each chunk into the workflow's stream (asserted via the workflowstreams offset
// query, which works even after completion), while the aggregated final response
// is returned into the workflow so the agent loop stays deterministic.
func TestStreamingIntegration(t *testing.T) {
	c := testClient(t)
	taskQueue := testTaskQueue(t)

	cm := &chunkedModel{name: "stream-model", chunks: []string{"Hello", ", ", "world"}}
	// Three 600,000-character chunks fit the Activity result, but their publish
	// batch is over the server's 2 MiB payload limit. With the 1,800,000-character
	// aggregate after them, the result still fits but the batch (about 4.8 MB)
	// is over the server's 4 MiB gRPC message limit.
	big := strings.Repeat("x", 600_000)
	oversized := &chunkedModel{name: "oversized-stream-model", chunks: []string{big, big, big}}
	tooLarge := &chunkedModel{name: "too-large-stream-model", chunks: []string{big, big, big}, final: big + big + big}
	// The same refused batch, but the model then fails: its error wins.
	failing := &chunkedModel{name: "failing-stream-model", chunks: []string{big, big, big},
		fail: &genai.APIError{Code: 400, Message: "bad request"}}
	startWorker(t, c, taskQueue, googleadk.Config{
		Models: map[string]googleadk.ModelFactory{
			"stream-model":           func(context.Context, string) (model.LLM, error) { return cm, nil },
			"oversized-stream-model": func(context.Context, string) (model.LLM, error) { return oversized, nil },
			"too-large-stream-model": func(context.Context, string) (model.LLM, error) { return tooLarge, nil },
			"failing-stream-model":   func(context.Context, string) (model.LLM, error) { return failing, nil },
		},
	})
	ctx := context.Background()
	run := startTestWorkflow(t, c, ctx, client.StartWorkflowOptions{
		ID:        taskQueue + "-streaming",
		TaskQueue: taskQueue,
	}, agentRunWorkflow, runInput{
		ModelName:      "stream-model",
		UserMessage:    "greet",
		StreamingTopic: "run-stream",
	})

	var res runResult
	require.NoError(t, run.Get(ctx, &res))
	assert.Contains(t, res.Texts, "Hello, world", "streamed chunks aggregate into one final response")
	assert.True(t, cm.streamed(), "StreamingTopic must drive the model in streaming mode")

	// The chunks were published into the workflow stream: the offset query returns
	// the number of appended items (>= the number of streamed chunks).
	val, err := c.QueryWorkflow(ctx, run.GetID(), run.GetRunID(), workflowstreams.OffsetQueryName)
	require.NoError(t, err)
	var offset int64
	require.NoError(t, val.Get(&offset))
	assert.GreaterOrEqual(t, offset, int64(len(cm.chunks)), "every streamed chunk must be published to the topic")

	// A batch the server refuses fails the model call instead of returning the
	// reply as if subscribers had every chunk. A retry would call the model again
	// and send the same batch, so the failure is non-retryable. A model error
	// stays the call's error.
	for _, tc := range []struct{ name, errType string }{
		{"oversized-stream-model", googleadk.ErrorTypeStream},
		{"too-large-stream-model", googleadk.ErrorTypeStream},
		{"failing-stream-model", googleadk.ErrorTypeModel},
	} {
		run = startTestWorkflow(t, c, ctx, client.StartWorkflowOptions{
			ID:        taskQueue + "-refused-" + tc.name,
			TaskQueue: taskQueue,
		}, agentRunWorkflow, runInput{
			ModelName:      tc.name,
			UserMessage:    "greet",
			StreamingTopic: "run-stream",
		})
		getCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		err = run.Get(getCtx, nil)
		cancel()
		require.Error(t, err, "%s: a refused publish batch must fail the model call", tc.name)
		var appErr *temporal.ApplicationError
		require.ErrorAs(t, err, &appErr, tc.name)
		assert.Equal(t, tc.errType, appErr.Type(), tc.name)
		assert.True(t, appErr.NonRetryable(), "%s: must not be retried: %v", tc.name, err)
	}
}

// TestReplaySingleAndMultiAgent runs real single-agent, multi-agent, MCP,
// remote A2A sub-agent (plain and streamed), and multi-decision HITL workflows
// against the dev server, then replays each recorded history with
// worker.WorkflowReplayer — the canonical determinism guarantee. A replay
// failure here is exactly the non-determinism the plugin's NewContext
// (time/uuid/task providers) exists to prevent. The replay must also send no
// A2A request: the remote agent's card and messages come from History.
func TestReplaySingleAndMultiAgent(t *testing.T) {
	c := testClient(t)
	taskQueue := testTaskQueue(t)

	a2aURL, a2aRequests := startA2AServer(t, a2aServer{
		Reply: a2aReply(a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("remote answer")), nil),
	})

	models := map[string]googleadk.ModelFactory{
		"root-model": scriptedModelFactory(
			googleadk.FunctionCallResponse("c1", "transfer_to_agent", map[string]any{"agent_name": "specialist"}),
			googleadk.TextResponse("(root fallback)"),
		),
		"specialist-model":         scriptedModelFactory(googleadk.TextResponse("specialist answer")),
		"solo-model":               scriptedModelFactory(googleadk.TextResponse("hello from solo")),
		"remote-root-model":        transferToRemoteModel(),
		"remote-stream-root-model": transferToRemoteModel(),
		"mcp-model": scriptedModelFactory(
			googleadk.FunctionCallResponse("c1", "search", map[string]any{"query": "temporal"}),
			googleadk.TextResponse("found it"),
		),
	}
	for name, factory := range multiConfirmModels() {
		models[name] = factory
	}
	startWorker(t, c, taskQueue, googleadk.Config{
		Models:      models,
		MCPToolsets: map[string]googleadk.MCPFactory{"search-server": searchServer().Factory()},
	})

	ctx := context.Background()
	inputs := map[string]runInput{
		taskQueue + "-solo": {
			ModelName:   "solo-model",
			UserMessage: "hi",
		},
		taskQueue + "-mcp": {
			ModelName:   "mcp-model",
			UserMessage: "search for temporal",
			MCPToolset:  "search-server",
		},
		taskQueue + "-multi": {
			ModelName:   "root-model",
			UserMessage: "handle this",
			SubAgents: []subAgentSpec{{
				Name:        "specialist",
				Description: "handles specialist tasks",
				ModelName:   "specialist-model",
			}},
		},
		taskQueue + "-remote": {
			ModelName:   "remote-root-model",
			UserMessage: "ask the remote agent",
			SubAgents:   []subAgentSpec{remoteSubAgent(a2aURL)},
		},
		taskQueue + "-remote-stream": {
			ModelName:   "remote-stream-root-model",
			UserMessage: "ask the remote agent",
			SubAgents:   []subAgentSpec{remoteSubAgent(a2aURL)},
			SSE:         true,
		},
	}

	type execution struct{ id, runID string }
	var executions []execution
	runs := map[string]client.WorkflowRun{}
	for id, in := range inputs {
		runs[id] = startTestWorkflow(t, c, ctx, client.StartWorkflowOptions{ID: id, TaskQueue: taskQueue}, agentRunWorkflow, in)
	}
	for id, run := range runs {
		var res runResult
		require.NoError(t, run.Get(ctx, &res))
		if len(inputs[id].SubAgents) == 1 && inputs[id].SubAgents[0].RemoteCardURL != "" {
			require.Contains(t, res.Texts, "remote answer")
		}
		executions = append(executions, execution{run.GetID(), run.GetRunID()})
	}
	liveA2ARequests := a2aRequests.Load()
	require.Positive(t, liveA2ARequests, "the live run must reach the A2A server")

	// The multi-decision confirmation resume: one turn pauses on three guarded
	// ActivityAsTool tools and a single batched ConfirmationResponse — with the
	// decisions deliberately rotated to (gamma, alpha, beta) — approves all of
	// them: the shape whose re-queue order was Go-map-random (and therefore not
	// replay-stable) before the adk/v2 minimum required in go.mod. The strict
	// order assertion pins that the resumed responses follow the confirmations'
	// request order, not the decisions' position, so a resume that silently did
	// nothing (or re-dispatched in decision order) fails here rather than
	// replaying cleanly below. It drives its own two-pass workflow rather than
	// agentRunWorkflow, so it is started directly and appended to the same
	// replay set.
	mcRun := startTestWorkflow(t, c, ctx, client.StartWorkflowOptions{
		ID:        taskQueue + "-multiconfirm",
		TaskQueue: taskQueue,
	}, multiConfirmHitlWorkflow)
	var mcRes multiConfirmResult
	require.NoError(t, mcRun.Get(ctx, &mcRes))
	require.Equal(t, 3, mcRes.PendingCount, "all three guarded tools must pause before the batched resume")
	require.Equal(t, []string{"guarded_alpha", "guarded_beta", "guarded_gamma"}, mcRes.ResumedToolResponses,
		"resumed responses must follow the confirmations' request order, not the rotated decision order")
	executions = append(executions, execution{mcRun.GetID(), mcRun.GetRunID()})

	// The plugin's Activities get a 10-second heartbeat timeout by default. The
	// guarded tools are the user's own Activities (ActivityAsTool) and get none.
	heartbeatTimeouts := map[string]bool{}
	for _, e := range executions {
		events := c.GetWorkflowHistory(ctx, e.id, e.runID, false, enumspb.HISTORY_EVENT_FILTER_TYPE_ALL_EVENT)
		for events.HasNext() {
			event, err := events.Next()
			require.NoError(t, err)
			if a := event.GetActivityTaskScheduledEventAttributes(); a != nil {
				heartbeatTimeouts[a.GetActivityType().GetName()+" "+a.GetHeartbeatTimeout().AsDuration().String()] = true
			}
		}
	}
	assert.Equal(t, map[string]bool{
		"googleadk.InvokeModel 10s":         true,
		"googleadk.ListMcpTools 10s":        true,
		"googleadk.CallMcpTool 10s":         true,
		"googleadk.ResolveA2AAgentCard 10s": true,
		"googleadk.SendA2AMessage 10s":      true,
		"guarded_alpha 0s":                  true,
		"guarded_beta 0s":                   true,
		"guarded_gamma 0s":                  true,
	}, heartbeatTimeouts)

	replayer := worker.NewWorkflowReplayer()
	replayer.RegisterWorkflow(agentRunWorkflow)
	replayer.RegisterWorkflow(multiConfirmHitlWorkflow)
	for _, e := range executions {
		err := replayer.ReplayWorkflowExecution(ctx, c.WorkflowService(), nil, "default", sdkworkflow.Execution{
			ID:    e.id,
			RunID: e.runID,
		})
		require.NoError(t, err, "recorded history for %s must replay deterministically", e.id)
	}
	assert.Equal(t, liveA2ARequests, a2aRequests.Load(), "replay must not send A2A requests again")
}
