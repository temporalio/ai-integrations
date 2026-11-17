package googleadk_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/testsuite"
	"go.temporal.io/sdk/workflow"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"

	"go.temporal.io/googleadk"
)

type SchemaIdentity struct {
	ID string `json:"id"`
}

type jsonActivityArgs struct {
	SchemaIdentity
	Labels   map[string]string `json:"labels"`
	At       time.Time         `json:"at"`
	Optional string            `json:"optional,omitempty"`
}

func jsonSchemaActivity(_ context.Context, in jsonActivityArgs) (map[string]any, error) {
	return map[string]any{"id": in.ID, "label": in.Labels["env"], "at": in.At.Format(time.RFC3339)}, nil
}

func TestActivityToolSchemaMatchesNativeADK(t *testing.T) {
	wrapped, err := googleadk.ActivityAsTool(jsonSchemaActivity, googleadk.ActivityToolOptions{Name: "json_schema"})
	require.NoError(t, err)
	native, err := functiontool.New(functiontool.Config{Name: "json_schema"}, func(agent.Context, jsonActivityArgs) (map[string]any, error) { return nil, nil })
	require.NoError(t, err)
	got := toolInputSchema(t, wrapped)
	assert.Equal(t, native.(declarer).Declaration().ParametersJsonSchema, got)
	assert.Contains(t, got.Properties, "id")
	assert.NotContains(t, got.Properties, "SchemaIdentity")
	assert.Equal(t, "object", got.Properties["labels"].Type)
	assert.Equal(t, "string", got.Properties["at"].Type)
	assert.ElementsMatch(t, []string{"id", "labels", "at"}, got.Required)
}

func jsonSchemaWorkflow(ctx workflow.Context) (runResult, error) {
	wrapped, err := googleadk.ActivityAsTool(jsonSchemaActivity, googleadk.ActivityToolOptions{Name: "json_schema"})
	if err != nil {
		return runResult{}, err
	}
	return runAgent(ctx, agentBuild{modelName: "fake-model", userMessage: "look up record", tools: []tool.Tool{wrapped}})
}

func TestActivityToolJSONArgumentsReachActivity(t *testing.T) {
	args := map[string]any{"id": "record-7", "labels": map[string]any{"env": "production"}, "at": "2026-11-16T12:00:00Z"}
	wrapped, err := googleadk.ActivityAsTool(jsonSchemaActivity, googleadk.ActivityToolOptions{})
	require.NoError(t, err)
	resolved, err := toolInputSchema(t, wrapped).Resolve(nil)
	require.NoError(t, err)
	require.NoError(t, resolved.Validate(args), "the model's arguments must match the declaration")
	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()
	env.RegisterWorkflow(jsonSchemaWorkflow)
	env.RegisterActivityWithOptions(jsonSchemaActivity, activity.RegisterOptions{Name: "json_schema"})
	wireActivities(t, env, googleadk.Config{Models: map[string]googleadk.ModelFactory{"fake-model": scriptedModelFactory(googleadk.FunctionCallResponse("c1", "json_schema", args), googleadk.TextResponse("done"))}})
	env.ExecuteWorkflow(jsonSchemaWorkflow)
	require.NoError(t, env.GetWorkflowError())
	var res runResult
	require.NoError(t, env.GetWorkflowResult(&res))
	assert.Equal(t, map[string]any{"id": "record-7", "label": "production", "at": "2026-11-16T12:00:00Z"}, res.ToolResponsePayloads["json_schema"])
}

// encodedLabel has a JSON representation that cannot be inferred from its fields.
type encodedLabel struct{ Value string }

func (v encodedLabel) MarshalJSON() ([]byte, error)  { return json.Marshal(v.Value) }
func (v *encodedLabel) UnmarshalJSON(b []byte) error { return json.Unmarshal(b, &v.Value) }

func TestActivityToolCustomInputSchema(t *testing.T) {
	type customArgs struct {
		Label encodedLabel `json:"label"`
	}
	fn := func(_ context.Context, in customArgs) (map[string]any, error) {
		return map[string]any{"label": in.Label.Value}, nil
	}
	override := &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{"label": {Type: "string"}}, Required: []string{"label"}}
	wrapped, err := googleadk.ActivityAsTool(fn, googleadk.ActivityToolOptions{InputSchema: override})
	require.NoError(t, err)
	resolved, err := toolInputSchema(t, wrapped).Resolve(nil)
	require.NoError(t, err)
	args := map[string]any{"label": "custom"}
	require.NoError(t, resolved.Validate(args))
	var s testsuite.WorkflowTestSuite
	env := s.NewTestActivityEnvironment()
	env.RegisterActivityWithOptions(fn, activity.RegisterOptions{Name: "custom_schema"})
	result, err := env.ExecuteActivity("custom_schema", args)
	require.NoError(t, err)
	var got map[string]any
	require.NoError(t, result.Get(&got))
	assert.Equal(t, map[string]any{"label": "custom"}, got)
}

func TestActivityToolRejectsUnsupportedArgumentSchema(t *testing.T) {
	fn := func(context.Context, struct{ Callback func() }) (map[string]any, error) { return nil, nil }
	_, err := googleadk.ActivityAsTool(fn, googleadk.ActivityToolOptions{})
	require.ErrorContains(t, err, "infer ActivityAsTool argument schema")
	_, err = googleadk.ActivityAsTool(jsonSchemaActivity, googleadk.ActivityToolOptions{InputSchema: &jsonschema.Schema{Ref: "#/$defs/missing"}})
	require.ErrorContains(t, err, "resolve ActivityAsTool argument schema")
}
