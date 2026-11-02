package googleadk_test

import (
	"testing"

	"github.com/stretchr/testify/require"
	enumspb "go.temporal.io/api/enums/v1"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"
)

func sharedServerWorkflow(ctx workflow.Context) error {
	return workflow.Await(ctx, func() bool { return false })
}

// The first subtest leaves a running workflow for cleanup. The next subtest
// must reuse the same client and server after the worker stops, find the
// original history, and observe that the workflow was terminated.
func TestClientSharedAcrossTests(t *testing.T) {
	var firstClient client.Client
	var run client.WorkflowRun
	if !t.Run("StartWorkflow", func(t *testing.T) {
		c := testClient(t)
		firstClient = c
		taskQueue := testTaskQueue(t)
		w := worker.New(c, taskQueue, worker.Options{})
		w.RegisterWorkflow(sharedServerWorkflow)
		require.NoError(t, w.Start())
		t.Cleanup(w.Stop)

		run = startTestWorkflow(t, c, t.Context(), client.StartWorkflowOptions{
			ID:        taskQueue,
			TaskQueue: taskQueue,
		}, sharedServerWorkflow)
	}) {
		return
	}

	t.Run("ReuseClient", func(t *testing.T) {
		c := testClient(t)
		require.Same(t, firstClient, c)
		description, err := c.DescribeWorkflowExecution(t.Context(), run.GetID(), run.GetRunID())
		require.NoError(t, err)
		require.Equal(t, enumspb.WORKFLOW_EXECUTION_STATUS_TERMINATED, description.WorkflowExecutionInfo.Status)
	})
}
