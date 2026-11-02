package googleadk

// Internal (white-box) tests for the streaming aggregation logic. This is the
// load-bearing correctness of the streaming feature: regardless of how many
// chunks the model streams to external consumers, the InvokeModel Activity must
// fold them into ONE coherent response handed back into the workflow so replay
// stays deterministic. These run by default with no Temporal server — the real
// publish side channel is exercised by the dev-server-gated integration test.

import (
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	enumspb "go.temporal.io/api/enums/v1"
	"go.temporal.io/api/serviceerror"
	"go.temporal.io/sdk/contrib/workflowstreams"
	"go.temporal.io/sdk/temporal"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

func partial(text string) *model.LLMResponse {
	return &model.LLMResponse{
		Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: text}}},
		Partial: true,
	}
}

// TestAggregateResponsesConcatenatesText proves streamed text chunks fold into a
// single message in order.
func TestAggregateResponsesConcatenatesText(t *testing.T) {
	var agg *model.LLMResponse
	for _, c := range []string{"Hello", ", ", "world"} {
		agg = aggregateResponses(agg, partial(c))
	}
	require.NotNil(t, agg)
	require.NotNil(t, agg.Content)
	assert.Equal(t, "Hello, world", concatText(agg.Content))
}

// TestAggregateResponsesCopiesFirst proves the first fold copies the response so
// the model's own buffer is never mutated across iterations.
func TestAggregateResponsesCopiesFirst(t *testing.T) {
	first := partial("a")
	agg := aggregateResponses(nil, first)
	_ = aggregateResponses(agg, partial("b"))
	// The original first response must be untouched (still just "a").
	assert.Equal(t, "a", concatText(first.Content))
	assert.Equal(t, "ab", concatText(agg.Content))
}

// TestAggregateResponsesFoldsMetadata proves later-chunk metadata wins: finish
// reason, usage, model version are carried onto the aggregate.
func TestAggregateResponsesFoldsMetadata(t *testing.T) {
	agg := aggregateResponses(nil, partial("hi"))
	next := partial(" there")
	next.FinishReason = genai.FinishReasonStop
	next.ModelVersion = "gemini-test"
	next.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{TotalTokenCount: 7}

	agg = aggregateResponses(agg, next)
	assert.Equal(t, "hi there", concatText(agg.Content))
	assert.Equal(t, genai.FinishReasonStop, agg.FinishReason)
	assert.Equal(t, "gemini-test", agg.ModelVersion)
	require.NotNil(t, agg.UsageMetadata)
	assert.EqualValues(t, 7, agg.UsageMetadata.TotalTokenCount)
}

// TestAggregateResponsesNilNext proves a nil chunk is a no-op.
func TestAggregateResponsesNilNext(t *testing.T) {
	agg := aggregateResponses(nil, partial("x"))
	assert.Same(t, agg, aggregateResponses(agg, nil))
}

// TestClassifyModelErrorRetryability proves the HTTP-status → retry-contract
// mapping used by InvokeModel: 429/5xx retryable, other 4xx non-retryable,
// status-less errors retryable.
func TestClassifyModelErrorRetryability(t *testing.T) {
	cases := []struct {
		code      int
		retryable bool
	}{
		{400, false},
		{404, false},
		{408, true},
		{429, true},
		{500, true},
		{503, true},
		{0, true}, // no status → transient
	}
	for _, tc := range cases {
		var err error = &genai.APIError{Code: tc.code, Message: "boom"}
		if tc.code == 0 {
			err = assert.AnError
		}
		classified := classifyModelError(err)
		assert.Equal(t, !tc.retryable, IsNonRetryable(classified), "code %d", tc.code)
	}
}

// TestStreamPublishErrorRetryability proves the retry contract for a stream
// that could not be published: a batch the server refused (invalid, or over
// the gRPC message limit) or one the stream client gave up on is final; rate
// limiting and an outage stay retryable.
func TestStreamPublishErrorRetryability(t *testing.T) {
	in := invokeModelInput{Request: &model.LLMRequest{Model: "m"}, StreamingTopic: "t"}
	cases := []struct {
		name      string
		err       error
		retryable bool
	}{
		{"over the payload limit", serviceerror.NewInvalidArgument("Blob data size exceeds limit."), false},
		{"over the gRPC message limit", status.Error(codes.ResourceExhausted, "grpc: received message after decompression larger than max 4194304"), false},
		{"rate limited", serviceerror.NewResourceExhausted(enumspb.RESOURCE_EXHAUSTED_CAUSE_RPS_LIMIT, "namespace rate limit exceeded"), true},
		{"rate limited, gRPC status", status.Error(codes.ResourceExhausted, "namespace rate limit exceeded"), true},
		{"client gave up", &workflowstreams.FlushTimeoutError{}, false},
		{"server unavailable", serviceerror.NewUnavailable("connection refused"), true},
	}
	for _, tc := range cases {
		err := streamPublishError(in, tc.err)
		var appErr *temporal.ApplicationError
		require.ErrorAs(t, err, &appErr, tc.name)
		assert.Equal(t, ErrorTypeStream, appErr.Type(), tc.name)
		assert.Equal(t, !tc.retryable, appErr.NonRetryable(), tc.name)
	}
}

// TestAggregateResponsesFoldsAllLateFields proves metadata a model emits only on
// a later streaming chunk — citations, logprobs, transcriptions, custom metadata,
// error fields — is carried onto the aggregate rather than dropped.
func TestAggregateResponsesFoldsAllLateFields(t *testing.T) {
	agg := aggregateResponses(nil, partial("hi"))

	last := partial(" there")
	last.CitationMetadata = &genai.CitationMetadata{}
	last.LogprobsResult = &genai.LogprobsResult{}
	last.CustomMetadata = map[string]any{"trace": "xyz"}
	last.OutputTranscription = &genai.Transcription{}
	last.AvgLogprobs = -0.25
	last.ErrorCode = "SAFETY"
	last.ErrorMessage = "blocked by safety"

	agg = aggregateResponses(agg, last)

	assert.Equal(t, "hi there", concatText(agg.Content))
	assert.NotNil(t, agg.CitationMetadata, "citation metadata from a later chunk must survive")
	assert.NotNil(t, agg.LogprobsResult)
	assert.NotNil(t, agg.OutputTranscription)
	require.NotNil(t, agg.CustomMetadata)
	assert.Equal(t, "xyz", agg.CustomMetadata["trace"])
	assert.Equal(t, -0.25, agg.AvgLogprobs)
	assert.Equal(t, "SAFETY", agg.ErrorCode)
	assert.Equal(t, "blocked by safety", agg.ErrorMessage)
}

// customStatusError stands for a provider SDK's error with an int StatusCode
// field, as openai-go's *openai.Error has. TestOpenAIModelErrorIsNonRetryable
// covers the real error that ADK's OpenAI model returns.
type customStatusError struct{ StatusCode int }

func (e customStatusError) Error() string { return fmt.Sprintf("status %d", e.StatusCode) }

// TestClassifyModelErrorWrappedStatusCode proves the status is found when the
// error is wrapped, also by a model supplied through Config.Models that joins
// it with other errors, through either Unwrap form, and that the status rule
// still applies.
func TestClassifyModelErrorWrappedStatusCode(t *testing.T) {
	ctxErr := errors.New("model context")
	for code, retryable := range map[int]bool{400: false, 401: false, 404: false, 408: true, 409: true, 429: true, 500: true, 503: true} {
		statusErr := &customStatusError{code}
		for name, err := range map[string]error{
			"value":       customStatusError{code},
			"pointer":     statusErr,
			"wrapped":     fmt.Errorf("openai: call failed: %w", statusErr),
			"joined":      errors.Join(statusErr, nil),
			"joined last": errors.Join(ctxErr, statusErr),
			"after zero":  errors.Join(customStatusError{0}, statusErr),
			"multiple %w": fmt.Errorf("%w: %w", ctxErr, statusErr),
			"nested":      fmt.Errorf("outer: %w", errors.Join(ctxErr, fmt.Errorf("openai: %w", statusErr))),
		} {
			assert.Equal(t, code, apiErrorCode(err), "%s %d", name, code)
			assert.Equal(t, !retryable, IsNonRetryable(classifyModelError(err)), "%s %d", name, code)
		}
	}
	// Joined status-less errors stay retryable, and a genai.APIError anywhere
	// in the tree still decides first.
	assert.False(t, IsNonRetryable(classifyModelError(errors.Join(ctxErr, assert.AnError))), "joined transient")
	assert.False(t, IsNonRetryable(classifyModelError(errors.Join(&customStatusError{404}, &genai.APIError{Code: 500}))), "joined genai 500")
}
