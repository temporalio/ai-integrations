package googleadk_test

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.temporal.io/sdk/testsuite"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/openaimodel"

	"go.temporal.io/googleadk"
)

// errStubOpenAIAPI returns an OpenAI error response with the given status.
type errStubOpenAIAPI struct{ status int }

func (e errStubOpenAIAPI) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Body != nil {
		_, _ = io.Copy(io.Discard, req.Body)
		_ = req.Body.Close()
	}
	return &http.Response{
		StatusCode: e.status,
		Status:     http.StatusText(e.status),
		Proto:      "HTTP/1.1",
		ProtoMajor: 1,
		ProtoMinor: 1,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(`{"error":{"message":"The model does not exist","type":"invalid_request_error","param":null,"code":"model_not_found"}}`)),
		Request:    req,
	}, nil
}

// TestOpenAIModelErrorIsNonRetryable drives ADK's real OpenAI model whose
// backend returns HTTP 404, without and with streaming. ADK wraps openai-go's
// error when not streaming and returns it unwrapped when streaming; either way
// the plugin reads its status, and the model Activity is attempted once.
func TestOpenAIModelErrorIsNonRetryable(t *testing.T) {
	openaiFactory := func(ctx context.Context, name string) (model.LLM, error) {
		return openaimodel.NewModel(ctx, name, &openaimodel.ClientConfig{
			APIKey:     "stub-key",
			HTTPClient: &http.Client{Transport: errStubOpenAIAPI{status: http.StatusNotFound}},
		})
	}

	for name, streamingTopic := range map[string]string{"not streamed": "", "streamed": "openai-stream"} {
		t.Run(name, func(t *testing.T) {
			var s testsuite.WorkflowTestSuite
			env, counter := newEnv(t, &s, googleadk.Config{
				Models: map[string]googleadk.ModelFactory{"gpt-4o-mini": openaiFactory},
			})

			env.ExecuteWorkflow(agentRunWorkflow, runInput{
				ModelName:      "gpt-4o-mini",
				UserMessage:    "hi",
				StreamingTopic: streamingTopic,
			})

			require.True(t, env.IsWorkflowCompleted())
			err := env.GetWorkflowError()
			require.Error(t, err)
			assert.True(t, googleadk.IsNonRetryable(err), "an OpenAI 404 must be non-retryable; got %v", err)
			assert.Equal(t, 1, counter.get(googleadk.InvokeModelActivityName))
		})
	}
}
