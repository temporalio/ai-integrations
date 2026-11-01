package googleadk

import (
	"context"
	"iter"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/worker"
)

// TestSendA2AMessageFailsWhenWorkerEndsStream proves a stream the worker ends
// after its first events, as a stop, pause or reset does by cancelling the
// Activity's context, fails the attempt so Temporal retries it, instead of
// completing with the cut stream. The workflow test environment cannot show
// it: it reports a timeout whatever the Activity returns.
func TestSendA2AMessageFailsWhenWorkerEndsStream(t *testing.T) {
	sent := make(chan struct{})
	release := make(chan struct{})
	executor := a2asrv.AgentExecutorFunc(func(_ context.Context, ec *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
		return func(yield func(a2a.Event, error) bool) {
			if yield(a2a.NewSubmittedTask(ec, ec.Message), nil) && yield(a2a.NewArtifactEvent(ec, a2a.NewTextPart("partial")), nil) {
				close(sent)
				<-release
			}
		}
	})
	srv := httptest.NewServer(a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(executor)))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) })
	card := &a2a.AgentCard{
		Name: "remote", Capabilities: a2a.AgentCapabilities{Streaming: true},
		SupportedInterfaces: []*a2a.AgentInterface{{
			URL: srv.URL, ProtocolBinding: a2a.TransportProtocolJSONRPC, ProtocolVersion: a2a.Version,
		}},
	}

	a, err := NewActivities(Config{})
	require.NoError(t, err)
	bg, stop := context.WithCancel(context.Background())
	defer stop()
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()
	env.SetWorkerOptions(worker.Options{BackgroundActivityContext: bg})
	env.RegisterActivityWithOptions(a.SendA2AMessage, activity.RegisterOptions{Name: SendA2AMessageActivityName})
	go func() {
		<-sent
		time.Sleep(300 * time.Millisecond) // the client reads both events
		stop()
	}()

	_, err = env.ExecuteActivity(SendA2AMessageActivityName, a2aSendInput{
		Card:    card,
		Request: &a2a.SendMessageRequest{Message: a2a.NewMessage(a2a.MessageRoleUser, a2a.NewTextPart("hi"))},
		Stream:  true,
	})
	require.Error(t, err, "the attempt must fail, not complete with the cut stream")
	assert.Contains(t, err.Error(), "context canceled")
}
