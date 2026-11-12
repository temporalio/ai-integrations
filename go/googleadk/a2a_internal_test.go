package googleadk

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/worker"
	"go.temporal.io/sdk/workflow"

	remoteagent "google.golang.org/adk/v2/agent/remoteagent/v2"
)

func TestA2AUntrustedCardErrorNotRetryable(t *testing.T) {
	untrusted := remoteagent.ErrUntrustedCardInterface
	for name, err := range map[string]error{
		"sentinel": untrusted,
		"wrapped":  fmt.Errorf("invalid interface: %w", untrusted),
		"joined":   errors.Join(assert.AnError, untrusted),
		"nested":   fmt.Errorf("resolve: %w", errors.Join(assert.AnError, fmt.Errorf("invalid interface: %w", untrusted))),
	} {
		t.Run(name, func(t *testing.T) {
			assert.True(t, IsNonRetryable(a2aError("resolve A2A agent card", err)))
		})
	}
	assert.False(t, IsNonRetryable(a2aError("resolve A2A agent card", assert.AnError)))
	assert.False(t, IsNonRetryable(a2aError("resolve A2A agent card", errors.New(untrusted.Error()))), "an unrelated error with the same message is not the sentinel")
}

// Card resolution succeeds for same-origin interfaces and rejects cross-origin
// interfaces without retrying the Activity.
func TestResolveA2AAgentCardUntrustedInterfaceNotRetried(t *testing.T) {
	for _, crossOrigin := range []bool{false, true} {
		t.Run(fmt.Sprintf("cross-origin=%t", crossOrigin), func(t *testing.T) {
			var requests atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				interfaceURL := "http://" + r.Host
				if crossOrigin {
					interfaceURL = "https://other.example"
				}
				require.NoError(t, json.NewEncoder(w).Encode(&a2a.AgentCard{
					Name: "remote",
					SupportedInterfaces: []*a2a.AgentInterface{{
						URL: interfaceURL, ProtocolBinding: a2a.TransportProtocolJSONRPC, ProtocolVersion: a2a.Version,
					}},
				}))
			}))
			t.Cleanup(srv.Close)

			acts, err := NewActivities(Config{})
			require.NoError(t, err)
			t.Cleanup(func() { require.NoError(t, acts.Close()) })
			var s testsuite.WorkflowTestSuite
			env := s.NewTestWorkflowEnvironment()
			env.RegisterActivityWithOptions(acts.ResolveA2AAgentCard, activity.RegisterOptions{Name: ResolveA2AAgentCardActivityName})
			env.ExecuteWorkflow(func(ctx workflow.Context) error {
				ctx = workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
					StartToCloseTimeout: time.Second,
					RetryPolicy:         &temporal.RetryPolicy{MaximumAttempts: 3},
				})
				return workflow.ExecuteActivity(ctx, ResolveA2AAgentCardActivityName, srv.URL).Get(ctx, nil)
			})
			if crossOrigin {
				require.Error(t, env.GetWorkflowError())
				assert.ErrorContains(t, env.GetWorkflowError(), remoteagent.ErrUntrustedCardInterface.Error())
				assert.True(t, IsNonRetryable(env.GetWorkflowError()))
			} else {
				require.NoError(t, env.GetWorkflowError())
			}
			assert.EqualValues(t, 1, requests.Load(), "card resolution must not be retried")
		})
	}
}

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
