package googleadk_test

import (
	"context"
	"errors"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	"go.temporal.io/api/serviceerror"
	"go.temporal.io/sdk/client"
)

// Connect when the first integration test needs the launcher-owned server.
var testClientState struct {
	once   sync.Once
	client client.Client
	err    error
}

func TestMain(m *testing.M) {
	code := m.Run()
	if c := testClientState.client; c != nil {
		c.Close()
	}
	os.Exit(code)
}

// testClient shares one client for the package.
func testClient(t *testing.T) client.Client {
	t.Helper()
	address := os.Getenv("TEMPORAL_TEST_ADDRESS")
	if address == "" {
		t.Fatal("Run live Temporal tests with `make test` or set TEMPORAL_TEST_ADDRESS to a running server.")
	}
	testClientState.once.Do(func() {
		testClientState.client, testClientState.err = client.Dial(client.Options{HostPort: address})
	})
	if testClientState.err != nil {
		t.Fatalf("Temporal test server unavailable: %v", testClientState.err)
	}
	return testClientState.client
}

func testTaskQueue(t *testing.T) string {
	return t.Name() + "-" + uuid.NewString()
}

// startTestWorkflow starts a workflow and registers termination at test cleanup,
// so an early assertion failure cannot leave it running on the shared server.
func startTestWorkflow(t *testing.T, c client.Client, ctx context.Context, options client.StartWorkflowOptions, wf any, args ...any) client.WorkflowRun {
	t.Helper()
	run, err := c.ExecuteWorkflow(ctx, options, wf, args...)
	require.NoError(t, err)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		err := c.TerminateWorkflow(ctx, run.GetID(), run.GetRunID(), "test cleanup")
		var notFound *serviceerror.NotFound
		if err != nil && !errors.As(err, &notFound) {
			t.Errorf("failed terminating test workflow %s: %v", run.GetID(), err)
		}
	})
	return run
}
