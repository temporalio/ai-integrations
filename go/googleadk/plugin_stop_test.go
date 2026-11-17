package googleadk_test

// Tests that stop a real Worker while MCP runs are open. They need the dev
// server: the workflow test environment has no Worker stop and no second
// Worker.

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	enumspb "go.temporal.io/api/enums/v1"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/worker"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/mcptoolset"

	"go.temporal.io/googleadk"
)

type noteArgs struct {
	Note string `json:"note"`
}

// noteWriter returns a factory for ADK's mcptoolset over an in-memory go-sdk
// MCP server with one "write" tool. A call with the note "held" reports on
// started and returns once release is closed. Any other call returns at once,
// so a test that holds no call passes nil channels.
func noteWriter(started chan<- struct{}, release <-chan struct{}) googleadk.MCPFactory {
	return func(context.Context) (tool.Toolset, error) {
		srv := mcp.NewServer(&mcp.Implementation{Name: "writer", Version: "1"}, nil)
		mcp.AddTool(srv, &mcp.Tool{Name: "write", Description: "Write a note."},
			func(_ context.Context, _ *mcp.CallToolRequest, in noteArgs) (*mcp.CallToolResult, any, error) {
				if in.Note == "held" {
					select {
					case started <- struct{}{}:
					default:
					}
					<-release
				}
				return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "wrote " + in.Note}}}, nil, nil
			})
		clientTransport, serverTransport := mcp.NewInMemoryTransports()
		if _, err := srv.Connect(context.Background(), serverTransport, nil); err != nil {
			return nil, err
		}
		return mcptoolset.New(mcptoolset.Config{Transport: clientTransport})
	}
}

// noteWriterConfig is the Config of a Worker that serves noteRun: the "writer"
// toolset and a model that calls "write" with the run's user message as the
// note, then answers. The model keeps no state, so any Worker can serve any
// step of a run.
func noteWriterConfig(started chan<- struct{}, release <-chan struct{}) googleadk.Config {
	m := &ruleModel{name: "fake-model", rule: func(req *model.LLMRequest) *model.LLMResponse {
		if lastIsFunctionResponse(req) {
			return googleadk.TextResponse("done")
		}
		return googleadk.FunctionCallResponse("c1", "write", map[string]any{"note": lastUserText(req)})
	}}
	return googleadk.Config{
		Models:      map[string]googleadk.ModelFactory{"fake-model": m.factory()},
		MCPToolsets: map[string]googleadk.MCPFactory{"writer": noteWriter(started, release)},
	}
}

// noteRun is the input of an agentRunWorkflow run that writes note.
func noteRun(note string) runInput {
	return runInput{ModelName: "fake-model", UserMessage: note, MCPToolset: "writer"}
}

// startNoteWorker starts a real Worker for agentRunWorkflow on taskQueue,
// wired with its own plugin, as NewPlugin asks.
func startNoteWorker(t *testing.T, c client.Client, taskQueue string, cfg googleadk.Config, stopTimeout time.Duration) worker.Worker {
	t.Helper()
	plugin, err := googleadk.NewPlugin(cfg)
	require.NoError(t, err)
	w := worker.New(c, taskQueue, worker.Options{Plugins: []worker.Plugin{plugin}, WorkerStopTimeout: stopTimeout})
	w.RegisterWorkflow(agentRunWorkflow)
	require.NoError(t, w.Start())
	return w
}

// retriedAfterClose reports whether an Activity attempt of run found its
// Activities value closed and was retried.
func retriedAfterClose(t *testing.T, c client.Client, ctx context.Context, run client.WorkflowRun) bool {
	t.Helper()
	events := c.GetWorkflowHistory(ctx, run.GetID(), run.GetRunID(), false, enumspb.HISTORY_EVENT_FILTER_TYPE_ALL_EVENT)
	for events.HasNext() {
		event, err := events.Next()
		require.NoError(t, err)
		if strings.Contains(event.GetActivityTaskStartedEventAttributes().GetLastFailure().GetMessage(), "MCP toolsets are closed") {
			return true
		}
	}
	return false
}

// TestWorkerStopFailsNoMcpRun proves a graceful Worker stop fails no MCP run.
// A Worker stops while one of its MCP calls is in flight, a second Worker
// serves the same Task Queue, and new MCP runs start right after the stop
// begins. If the plugin closed its toolsets before its Worker had stopped, the
// stopping Worker would still start a new run's ListMcpTools, which would wait
// for the close and then find the toolsets closed.
func TestWorkerStopFailsNoMcpRun(t *testing.T) {
	for _, tc := range []struct {
		name        string
		stopTimeout time.Duration
	}{
		{"default WorkerStopTimeout", 0},
		{"WorkerStopTimeout longer than the call", 30 * time.Second},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := testClient(t)
			taskQueue := testTaskQueue(t)
			ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
			defer cancel()
			started := make(chan struct{}, 1)
			release := make(chan struct{})
			var releaseOnce, stopOnce sync.Once
			releaseHeld := func() { releaseOnce.Do(func() { close(release) }) }

			stopping := startNoteWorker(t, c, taskQueue, noteWriterConfig(started, release), tc.stopTimeout)
			stopped := make(chan struct{})
			stop := func() {
				stopOnce.Do(func() {
					stopping.Stop()
					close(stopped)
				})
			}
			t.Cleanup(func() {
				releaseHeld()
				stop()
			})

			held := startTestWorkflow(t, c, ctx, client.StartWorkflowOptions{ID: taskQueue + "-held", TaskQueue: taskQueue},
				agentRunWorkflow, noteRun("held"))
			select {
			case <-started:
			case <-ctx.Done():
				t.Fatal("the held MCP call never started")
			}
			serving := startNoteWorker(t, c, taskQueue, noteWriterConfig(started, release), 0)
			t.Cleanup(serving.Stop)

			go stop()
			// Let the stop reach the Worker's pollers before the new runs
			// start. A Workflow Task that a Worker takes while its pollers shut
			// down is not run, and the server hands it out again only when it
			// times out, after 10 seconds.
			time.Sleep(200 * time.Millisecond)
			var newRuns []client.WorkflowRun
			for i := range 8 {
				newRuns = append(newRuns, startTestWorkflow(t, c, ctx,
					client.StartWorkflowOptions{ID: fmt.Sprintf("%s-new-%d", taskQueue, i), TaskQueue: taskQueue},
					agentRunWorkflow, noteRun("new")))
			}
			// The second Worker can finish every new run while the call is
			// held. A run that waits for a closing toolset cannot, so the
			// wait is bounded.
			holdCtx, endHold := context.WithTimeout(ctx, 5*time.Second)
			for _, run := range newRuns {
				_ = run.Get(holdCtx, nil)
			}
			endHold()
			releaseHeld()
			<-stopped

			wrote := func(run client.WorkflowRun, note string) {
				t.Helper()
				var res runResult
				require.NoError(t, run.Get(ctx, &res), run.GetID())
				assert.Equal(t, map[string]any{"output": "wrote " + note}, res.ToolResponsePayloads["write"], run.GetID())
				if tc.stopTimeout > 0 {
					// The stop waited for the held call, so the toolsets
					// closed after the last Activity of the stopping Worker.
					assert.False(t, retriedAfterClose(t, c, ctx, run), "%s: an Activity found the toolsets closed", run.GetID())
				}
			}
			for _, run := range newRuns {
				wrote(run, "new")
			}
			wrote(held, "held")
		})
	}
}

// TestMcpActivityAfterCloseIsRetried proves an MCP Activity that reaches an
// Activities value after Close does not fail the run: its attempt fails with a
// retryable error, and a Worker whose toolsets are open runs the next one.
// Here the Activities are wired by hand and closed while their Worker still
// polls.
func TestMcpActivityAfterCloseIsRetried(t *testing.T) {
	c := testClient(t)
	taskQueue := testTaskQueue(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	acts, err := googleadk.NewActivities(noteWriterConfig(nil, nil))
	require.NoError(t, err)
	closedWorker := worker.New(c, taskQueue, worker.Options{})
	closedWorker.RegisterWorkflow(agentRunWorkflow)
	acts.Register(closedWorker)
	require.NoError(t, closedWorker.Start())
	t.Cleanup(closedWorker.Stop)
	require.NoError(t, acts.Close())

	run := startTestWorkflow(t, c, ctx, client.StartWorkflowOptions{ID: taskQueue, TaskQueue: taskQueue},
		agentRunWorkflow, noteRun("new"))
	// Wait until the first attempt has failed on the closed Activities, or
	// until the run has ended because that failure was final.
	require.Eventually(t, func() bool {
		desc, err := c.DescribeWorkflowExecution(ctx, run.GetID(), run.GetRunID())
		if err != nil {
			return false
		}
		if desc.GetWorkflowExecutionInfo().GetStatus() != enumspb.WORKFLOW_EXECUTION_STATUS_RUNNING {
			return true
		}
		for _, a := range desc.GetPendingActivities() {
			if a.GetLastFailure() != nil {
				return true
			}
		}
		return false
	}, 30*time.Second, 20*time.Millisecond)

	closedWorker.Stop()
	openWorker := startNoteWorker(t, c, taskQueue, noteWriterConfig(nil, nil), 0)
	t.Cleanup(openWorker.Stop)

	var res runResult
	require.NoError(t, run.Get(ctx, &res))
	assert.Equal(t, map[string]any{"output": "wrote new"}, res.ToolResponsePayloads["write"])
	assert.True(t, retriedAfterClose(t, c, ctx, run), "the first attempt must have found the Activities closed")
}
