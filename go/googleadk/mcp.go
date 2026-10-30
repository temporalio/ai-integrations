package googleadk

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"

	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/workflow"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
	"google.golang.org/adk/v2/tool/toolutils"
)

// MCPToolsetOptions configures NewMCPToolset.
type MCPToolsetOptions struct {
	// Name is the logical toolset name. It must match a key in
	// Config.MCPToolsets so the worker-side factory can be found.
	Name string
	// ToolFilter optionally restricts which advertised MCP tools are exposed to
	// the model. It runs on the workflow side over declaration-only proxy tools.
	ToolFilter tool.Predicate
	// ActivityOptions overrides the per-call Activity options for the ListMcpTools
	// and CallMcpTool Activities. A zero StartToCloseTimeout falls back to the
	// default one-minute tool timeout.
	ActivityOptions workflow.ActivityOptions
}

// NewMCPToolset returns a workflow-side, stateless proxy for an MCP toolset. It
// advertises the remote tools (full declarations, including parameters) via the
// ListMcpTools Activity and executes each call via the CallMcpTool Activity. The
// real, stateful mcptoolset.New(...) — which holds a live network session — runs
// worker-side behind those Activities, never in the workflow.
//
// Add the returned Toolset to your agent like any other:
//
//	a, _ := llmagent.New(llmagent.Config{
//	    Model:    myModel,
//	    Toolsets: []tool.Toolset{googleadk.NewMCPToolset(googleadk.MCPToolsetOptions{Name: "filesystem"})},
//	})
func NewMCPToolset(opts MCPToolsetOptions) tool.Toolset {
	return &mcpToolset{name: opts.Name, filter: opts.ToolFilter, activityOptions: opts.ActivityOptions}
}

// mcpToolset is the workflow-side proxy. It implements tool.Toolset and ADK's
// internal RequestProcessor (so its proxy tools are packed into the model
// request).
type mcpToolset struct {
	name            string
	filter          tool.Predicate
	activityOptions workflow.ActivityOptions
}

func (m *mcpToolset) Name() string { return m.name }

// Tools returns the toolset's declaration-only proxy tools. Until the
// toolset's ProcessRequest has run in the agent run, it lists them via the
// ListMcpTools Activity, applies ToolFilter, and keeps the list for the
// ProcessRequest call that follows in the same model step. After that,
// ProcessRequest lists, and Tools returns the list ToolFilter last allowed
// there, without an Activity and without running the filter. See
// ProcessRequest.
//
// So the first Tools call of each agent run lists and filters. ADK resumes a
// confirmed tool call from this list before any ProcessRequest of the run, so
// a tool the filter no longer allows does not run. The later calls, which ADK
// v2.6.0 and later make before every model step, issue no Workflow commands,
// so the toolset and its filter issue them at the same points on every ADK
// release.
//
// A toolset wrapped in a toolset without ProcessRequest, such as
// tool.FilterToolset, lists and filters on every call, because nothing else
// refreshes its list.
func (m *mcpToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	l := listingsFrom(ctx)
	run := l.startRun(m, ctx)
	if tools, ok := l.processedList(run); ok {
		return tools, nil
	}
	tools, err := m.list(ctx)
	if err != nil {
		return nil, err
	}
	l.keepPending(run, tools)
	return m.allowed(ctx, tools), nil
}

// allowed returns the tools the toolset's ToolFilter allows for ctx.
func (m *mcpToolset) allowed(ctx agent.ReadonlyContext, tools []tool.Tool) []tool.Tool {
	if m.filter == nil {
		return tools
	}
	out := make([]tool.Tool, 0, len(tools))
	for _, t := range tools {
		if m.filter(ctx, t) {
			out = append(out, t)
		}
	}
	return out
}

// list runs the ListMcpTools Activity and wraps each declaration as a proxy
// tool. It does not apply ToolFilter; see allowed.
func (m *mcpToolset) list(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	wfCtx, ok := workflowContext(ctx)
	if !ok {
		return nil, errMissingContext
	}
	actx := workflow.WithActivityOptions(wfCtx, resolveToolActivityOptions(m.activityOptions, ""))
	var decls []*genai.FunctionDeclaration
	if err := workflow.ExecuteActivity(actx, ListMcpToolsActivityName, mcpListInput{Toolset: m.name}).Get(wfCtx, &decls); err != nil {
		return nil, err
	}
	tools := make([]tool.Tool, 0, len(decls))
	for _, d := range decls {
		tools = append(tools, &mcpProxyTool{owner: m, toolset: m.name, decl: d, activityOptions: m.activityOptions})
	}
	return tools, nil
}

// ProcessRequest packs the toolset's current proxy tools into the model
// request. ADK calls it before each model call, after the request processors
// (instructions and the others) have run.
//
// ADK v2.5.0 and earlier call Tools only in the first model step of an agent
// run, before the request processors. v2.6.0 and later call Tools before the
// request processors in every step. To schedule the same commands in the same
// order on every ADK release, the toolset lists in Tools only until its first
// ProcessRequest of the agent run, and in ProcessRequest from then on. Each
// model step lists once: the first through Tools, with ProcessRequest reusing
// that list, and every later step here. The request carries this list alone,
// so a tool the MCP server no longer advertises is neither declared nor
// dispatched, whatever list ADK kept from an earlier Tools call.
func (m *mcpToolset) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	l := listingsFrom(ctx)
	run := l.current(m, ctx)
	tools, ok := l.takePending(run)
	if !ok {
		var err error
		if tools, err = m.list(ctx); err != nil {
			return err
		}
	}
	tools = m.allowed(ctx, tools)
	l.processed(run, tools)
	for _, t := range tools {
		if err := packTool(req, t.(*mcpProxyTool)); err != nil {
			return err
		}
	}
	return nil
}

// mcpListingsKey is the private context key for the mcpListings NewContext
// installs.
type mcpListingsKey struct{}

// mcpListings records, for each MCP toolset and agent run, the lists the
// toolset took, so that Tools and ProcessRequest list once per model step at
// the same point on every ADK release.
type mcpListings struct {
	mu   sync.Mutex
	runs map[mcpAgentKey]*mcpRunState
}

// mcpAgentKey identifies a toolset of one agent in one invocation. Agent names
// are unique within an agent tree, and each runner.Run call has its own
// invocation ID, except one that resumes or retries a paused run. Concurrent
// agent-tool calls each make their own runner.Run call.
type mcpAgentKey struct {
	toolset    *mcpToolset
	invocation string
	agent      string
}

// mcpRunState is the state of a toolset in one agent run: one agent in one
// execution of a runner.Run iterator.
type mcpRunState struct {
	// run is the InvocationContext ADK creates for the agent in each execution
	// of a runner.Run iterator, a resume, a retry or another pass over a kept
	// iterator included, and passes to every Tools call of that agent run. It
	// tells apart the runs of one agent in one invocation.
	run agent.InvocationContext
	// pending is the list Tools took for the ProcessRequest call that follows
	// in the same model step, when hasPending is set.
	pending    []tool.Tool
	hasPending bool
	// processed reports that the toolset's ProcessRequest has run in the agent
	// run. From then on ProcessRequest lists, and Tools does not.
	processed bool
	// allowed is the list ToolFilter allowed in the last ProcessRequest, which
	// Tools returns once processed.
	allowed []tool.Tool
}

// listingsFrom returns the mcpListings NewContext installed, or nil.
func listingsFrom(ctx context.Context) *mcpListings {
	l, _ := ctx.Value(mcpListingsKey{}).(*mcpListings)
	return l
}

// agentRunOf returns the InvocationContext of the agent run ctx belongs to, or
// nil. ADK passes Tools a ReadonlyContext that holds it in the exported
// InvocationContext field. It is not the RunConfig: ADK reuses one RunConfig
// when a runner.Run iterator without an LLM agent at its root runs again.
func agentRunOf(ctx agent.ReadonlyContext) agent.InvocationContext {
	v := reflect.ValueOf(ctx)
	if v.Kind() != reflect.Pointer || v.IsNil() || v.Elem().Kind() != reflect.Struct {
		return nil
	}
	f := v.Elem().FieldByName("InvocationContext")
	if !f.IsValid() || !f.CanInterface() {
		return nil
	}
	ic, ok := f.Interface().(agent.InvocationContext)
	if !ok || reflect.ValueOf(ic).Kind() != reflect.Pointer {
		return nil
	}
	return ic
}

// startRun returns the state of the toolset in ctx's agent run, for Tools. It
// starts a new state when ctx belongs to another agent run than the state
// kept for the agent and invocation. When ctx does not tell its agent run, it
// drops the kept state and returns nil.
//
// Without a state, the toolset lists and filters on every Tools and
// ProcessRequest call, and its proxy tools pack themselves.
func (l *mcpListings) startRun(m *mcpToolset, ctx agent.ReadonlyContext) *mcpRunState {
	if l == nil {
		return nil
	}
	run := agentRunOf(ctx)
	k := mcpAgentKey{m, ctx.InvocationID(), ctx.AgentName()}
	l.mu.Lock()
	defer l.mu.Unlock()
	if run == nil {
		delete(l.runs, k)
		return nil
	}
	if l.runs == nil {
		l.runs = map[mcpAgentKey]*mcpRunState{}
	}
	s, ok := l.runs[k]
	if !ok || s.run != run {
		s = &mcpRunState{run: run}
		l.runs[k] = s
	}
	return s
}

// current returns the state the toolset's last Tools call for ctx's agent and
// invocation kept, or nil. ADK calls Tools before ProcessRequest in the first
// model step of every agent run.
func (l *mcpListings) current(m *mcpToolset, ctx agent.ReadonlyContext) *mcpRunState {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.runs[mcpAgentKey{m, ctx.InvocationID(), ctx.AgentName()}]
}

// processedList returns the list ToolFilter last allowed in the run, once the
// toolset's ProcessRequest has run there.
func (l *mcpListings) processedList(s *mcpRunState) ([]tool.Tool, bool) {
	if s == nil {
		return nil, false
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return s.allowed, s.processed
}

// keepPending keeps a list Tools took for the ProcessRequest call that follows.
func (l *mcpListings) keepPending(s *mcpRunState, tools []tool.Tool) {
	if s == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	s.pending, s.hasPending = tools, true
}

// takePending returns and drops the list Tools kept, if any.
func (l *mcpListings) takePending(s *mcpRunState) ([]tool.Tool, bool) {
	if s == nil {
		return nil, false
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	tools, ok := s.pending, s.hasPending
	s.pending, s.hasPending = nil, false
	return tools, ok
}

// processed marks the toolset's ProcessRequest as run in the agent run, and
// records the list ToolFilter allowed there.
func (l *mcpListings) processed(s *mcpRunState, allowed []tool.Tool) {
	if s == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	s.processed, s.allowed = true, allowed
}

// packedByToolset reports whether the toolset's ProcessRequest has run in the
// agent run. Its proxy tools then leave packing to it.
func (l *mcpListings) packedByToolset(m *mcpToolset, ctx agent.ReadonlyContext) bool {
	_, ok := l.processedList(l.current(m, ctx))
	return ok
}

// mcpProxyTool is a declaration-only stand-in for a remote MCP tool. The model
// sees its declaration; when the model calls it, Run (invoked in-workflow by
// ADK) dispatches the CallMcpTool Activity so the live, stateful MCP session
// runs worker-side, never in the workflow.
type mcpProxyTool struct {
	owner           *mcpToolset
	toolset         string
	decl            *genai.FunctionDeclaration
	activityOptions workflow.ActivityOptions
}

func (t *mcpProxyTool) Name() string                            { return t.decl.Name }
func (t *mcpProxyTool) Description() string                     { return t.decl.Description }
func (t *mcpProxyTool) IsLongRunning() bool                     { return false }
func (t *mcpProxyTool) Declaration() *genai.FunctionDeclaration { return t.decl }

// ProcessRequest packs the proxy tool. ADK calls it for each tool it kept from
// a Tools call. Once the owning toolset's ProcessRequest packs the current list
// in the agent run, it does nothing: the tool may come from an older list, which
// ADK v2.5.0 and earlier keep for the whole agent run.
func (t *mcpProxyTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	if listingsFrom(ctx).packedByToolset(t.owner, ctx) {
		return nil
	}
	return packTool(req, t)
}

// Run executes in-workflow and dispatches the CallMcpTool Activity, which builds
// the live MCP toolset worker-side and runs the named tool over a reconstructed,
// read-only context. MCP calls are inherently I/O, so they always run as an
// Activity. ADK's standard tool-confirmation protocol tunnels through: the
// snapshot carries any human decision out to the worker-side tool, and a
// confirmation the tool requested there is replayed into the workflow-side
// EventActions so the runner pauses exactly like an in-workflow tool.
func (t *mcpProxyTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	wfCtx, ok := workflowContext(ctx)
	if !ok {
		return nil, errMissingContext
	}
	ao := resolveToolActivityOptions(t.activityOptions, "")
	ao.Summary = toolSummary(ctx, t.Name())
	actx := workflow.WithActivityOptions(wfCtx, ao)
	argsMap, _ := args.(map[string]any)
	in := mcpCallInput{Toolset: t.toolset, Tool: t.Name(), Args: argsMap, Ctx: snapshotContext(ctx)}
	var out mcpCallOutput
	if err := workflow.ExecuteActivity(actx, CallMcpToolActivityName, in).Get(wfCtx, &out); err != nil {
		return nil, err
	}
	if out.Confirmation != nil {
		// Re-record the worker-side confirmation request into the REAL
		// workflow-side EventActions (keyed by this call's FunctionCallID,
		// identical on both sides — RequestConfirmation also sets
		// SkipSummarization), then return ADK's sentinel so the flow emits the
		// adk_request_confirmation event and the runner pauses.
		if rerr := ctx.RequestConfirmation(out.Confirmation.Hint, out.Confirmation.Payload); rerr != nil {
			return nil, rerr
		}
		return nil, fmt.Errorf("error tool %q %w", t.Name(), tool.ErrConfirmationRequired)
	}
	if out.Result == nil {
		return map[string]any{}, nil
	}
	return out.Result, nil
}

// mcpListInput is the serializable payload for ListMcpTools.
type mcpListInput struct {
	Toolset string
}

// mcpCallInput is the serializable payload for CallMcpTool.
type mcpCallInput struct {
	Toolset string
	Tool    string
	Args    map[string]any
	Ctx     contextSnapshot
}

// mcpCallOutput is the serializable result of CallMcpTool. Confirmation is
// non-nil when the tool paused on ADK's tool.ErrConfirmationRequired protocol
// instead of running; the workflow-side proxy re-records it so the runner
// surfaces the pause exactly like an in-workflow tool.
type mcpCallOutput struct {
	Result       map[string]any
	Confirmation *toolconfirmation.ToolConfirmation
}

// ListMcpTools returns the full declarations (name + description + parameters)
// of the tools in the named MCP toolset, by constructing the live toolset
// worker-side and listing it. Returning full declarations — not just
// {name, description} — lets the model produce well-formed arguments.
func (a *Activities) ListMcpTools(ctx context.Context, in mcpListInput) ([]*genai.FunctionDeclaration, error) {
	log := activity.GetLogger(ctx)
	// No-op unless the user set a HeartbeatTimeout on the toolset's Activity
	// options; then a slow MCP server (or first-connect) must not be falsely
	// heartbeat-timed-out and retried. Same rationale as InvokeModel.
	stopHeartbeats := startPeriodicHeartbeats(ctx)
	defer stopHeartbeats()
	ts, err := a.mcpToolset(ctx, in.Toolset)
	if err != nil {
		return nil, err
	}
	tools, err := ts.Tools(newActivityToolContext(ctx, contextSnapshot{}))
	if err != nil {
		return nil, newApplicationError(ErrorTypeMCP, true, err,
			"list tools for MCP toolset %q: %v", in.Toolset, err)
	}
	decls := make([]*genai.FunctionDeclaration, 0, len(tools))
	for _, t := range tools {
		if d, ok := t.(interface {
			Declaration() *genai.FunctionDeclaration
		}); ok {
			decls = append(decls, d.Declaration())
		}
	}
	log.Debug("listed MCP tools", "toolset", in.Toolset, "count", len(decls))
	return decls, nil
}

// CallMcpTool executes a single tool of the named MCP toolset worker-side over a
// reconstructed, read-only ToolContext. When the tool pauses on ADK's standard
// confirmation protocol (e.g. mcptoolset's RequireConfirmation option) the
// Activity succeeds with the pending confirmation in its output rather than
// failing, so the workflow can pause the agent for the human.
func (a *Activities) CallMcpTool(ctx context.Context, in mcpCallInput) (*mcpCallOutput, error) {
	log := activity.GetLogger(ctx)
	// No-op unless the user set a HeartbeatTimeout on the toolset's Activity
	// options; then a slow MCP tool call must not be falsely heartbeat-timed-out
	// and retried. Same rationale as InvokeModel.
	stopHeartbeats := startPeriodicHeartbeats(ctx)
	defer stopHeartbeats()
	ts, err := a.mcpToolset(ctx, in.Toolset)
	if err != nil {
		return nil, err
	}
	tctx := newActivityToolContext(ctx, in.Ctx)
	tools, err := ts.Tools(tctx)
	if err != nil {
		return nil, newApplicationError(ErrorTypeMCP, true, err,
			"list tools for MCP toolset %q: %v", in.Toolset, err)
	}
	for _, t := range tools {
		if t.Name() != in.Tool {
			continue
		}
		rt, ok := t.(runnable)
		if !ok {
			return nil, newApplicationError(ErrorTypeMCP, false, nil,
				"MCP tool %q is not runnable", in.Tool)
		}
		log.Debug("running MCP tool", "toolset", in.Toolset, "tool", in.Tool)
		res, runErr := rt.Run(tctx, in.Args)
		if runErr != nil {
			if errors.Is(runErr, tool.ErrConfirmationRequired) {
				// The tool paused on ADK's confirmation protocol rather than
				// failing. Hand back the request it recorded via
				// RequestConfirmation so the workflow-side proxy can replay it
				// into the real EventActions.
				tc, ok := tctx.acts.RequestedToolConfirmations[in.Ctx.FunctionCallID]
				if !ok {
					tc = toolconfirmation.ToolConfirmation{}
				}
				return &mcpCallOutput{Confirmation: &tc}, nil
			}
			if errors.Is(runErr, tool.ErrConfirmationRejected) {
				// Deterministic human denial: retrying can never succeed.
				return nil, newApplicationError(ErrorTypeMCP, false, runErr,
					"tool %q failed: %v", in.Tool, runErr)
			}
			return nil, classifyToolError(ErrorTypeMCP, in.Tool, runErr)
		}
		return &mcpCallOutput{Result: res}, nil
	}
	return nil, newApplicationError(ErrorTypeMCP, false, nil,
		"tool %q not found in MCP toolset %q", in.Tool, in.Toolset)
}

// mcpToolset resolves the named toolset, running its registered factory at most
// once and caching the result for the life of the Activities value. The cache
// is what bounds the resource cost: a live MCP toolset lazily opens and retains
// a client session (for CommandTransport, a server subprocess) that upstream
// exposes no way to close per call, so constructing per Activity execution
// would leak one session per call. Construction errors are NOT cached — the
// next Temporal retry re-runs the factory. mcpMu is held across the factory
// call deliberately: it serializes concurrent first use so two sessions are
// never constructed for one name.
func (a *Activities) mcpToolset(ctx context.Context, name string) (tool.Toolset, error) {
	f, ok := a.mcp[name]
	if !ok {
		return nil, newApplicationError(ErrorTypeMCP, false, nil,
			"no MCP toolset %q registered; add it to Config.MCPToolsets", name)
	}
	a.mcpMu.Lock()
	defer a.mcpMu.Unlock()
	if a.mcpClosed {
		// Close is terminal: rebuilding here would cache a toolset nothing will
		// ever close. Non-retryable — the worker is shutting down.
		return nil, newApplicationError(ErrorTypeMCP, false, nil,
			"MCP toolsets are closed (Activities.Close was called)")
	}
	if ts, ok := a.mcpCache[name]; ok {
		return ts, nil
	}
	ts, err := f(ctx)
	if err != nil {
		return nil, newApplicationError(ErrorTypeMCP, true, err,
			"construct MCP toolset %q: %v", name, err)
	}
	a.mcpCache[name] = ts
	return ts, nil
}

// packTool advertises a proxy tool's declaration to the model via ADK's public
// tool-declaration packer (tool/toolutils), recording the live tool under
// req.Tools (json:"-", so it never crosses the wire) so ADK can later resolve it
// to dispatch. It is idempotent: ADK runs both toolPreprocess (each expanded
// toolset tool's ProcessRequest) and toolsetPreprocess (the toolset's own
// ProcessRequest), so a toolset tool is packed twice with identical
// declarations. A name already present is skipped, rather than erroring as
// toolutils.PackTool does on a duplicate.
func packTool(req *model.LLMRequest, t toolutils.Tool) error {
	if req.Tools != nil {
		if _, ok := req.Tools[t.Name()]; ok {
			return nil
		}
	}
	return toolutils.PackTool(req, t)
}
