package googleadk_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	"go.temporal.io/api/serviceerror"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/testsuite"
)

// Start the server only when an integration test needs it. Remember failures
// too, so later tests fail fast instead of repeating a failed CLI download.
var devServerState struct {
	once   sync.Once
	server *testsuite.DevServer
	err    error
}

func TestMain(m *testing.M) {
	code := m.Run()
	if srv := devServerState.server; srv != nil {
		srv.Client().Close()
		if err := srv.Stop(); err != nil {
			// The SDK's console interrupt can fail on Windows CI. Keep shutdown
			// best effort, as it was when every test owned a separate server.
			fmt.Fprintf(os.Stderr, "failed stopping Temporal dev server: %v\n", err)
		}
	}
	os.Exit(code)
}

// Only initial provisioning uses retries. The SDK loses the process handle
// after a readiness failure, so retrying that path would leak another server.
func startDevServerWithRetry(start func() (*testsuite.DevServer, error)) (*testsuite.DevServer, error) {
	var startupErr error
	for attempt := 1; ; attempt++ {
		srv, err := start()
		if err == nil {
			return srv, nil
		}
		startupErr = errors.Join(startupErr, fmt.Errorf("attempt %d: %w", attempt, err))
		if attempt >= 2 || errors.Is(err, context.Canceled) || strings.HasPrefix(err.Error(), "failed connecting after timeout,") {
			return nil, startupErr
		}
	}
}

// devServer returns a test-owned client connected to the package's shared
// server. Workers and workflows must finish before this test's cleanup closes
// the client; TestMain owns the server and its original startup client.
func devServer(t *testing.T) client.Client {
	t.Helper()
	devServerState.once.Do(func() {
		// On Windows the SDK may fail to stop a server without an attached
		// console. Detached stdio keeps that process from holding go test's
		// output pipes open; the CI runner reaps it when the job exits.
		opts := testsuite.DevServerOptions{Stdout: io.Discard, Stderr: io.Discard}
		devServerState.server, devServerState.err = startDevServerWithRetry(func() (*testsuite.DevServer, error) {
			// The first use may need to download the CLI on a slow runner.
			ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
			defer cancel()
			return testsuite.StartDevServer(ctx, opts)
		})
	})
	if devServerState.err != nil {
		t.Fatalf("dev server unavailable (first use downloads the Temporal CLI, which needs network): %v", devServerState.err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	c, err := client.DialContext(ctx, client.Options{HostPort: devServerState.server.FrontendHostPort()})
	require.NoError(t, err)
	t.Cleanup(c.Close)
	return c
}

func testTaskQueue(t *testing.T) string {
	return t.Name() + "-" + uuid.NewString()
}

// A failed test must not leave work running against the shared server. This
// cleanup runs before the worker and client cleanups registered at setup.
func cleanupWorkflow(t *testing.T, c client.Client, run client.WorkflowRun) {
	t.Helper()
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		err := c.TerminateWorkflow(ctx, run.GetID(), run.GetRunID(), "test cleanup")
		var notFound *serviceerror.NotFound
		if err != nil && !errors.As(err, &notFound) {
			t.Errorf("failed terminating test workflow %s: %v", run.GetID(), err)
		}
	})
}
