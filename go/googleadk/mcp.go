package googleadk

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"weak"

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
	// the model. It runs on the workflow side over declaration-only proxy tools,
	// each time the toolset lists: in every Tools call before the toolset's
	// first ProcessRequest of an agent run, and in every ProcessRequest. With
	// only ADK's own calls, that is twice in the first model step and once in
	// each later one. ADK resumes confirmed tool calls from the first result and
	// draws its generated IDs for it.
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
// ProcessRequest lists, and Tools returns the list the first Tools call of the
// agent run returned, without an Activity and without running the filter. See
// ProcessRequest.
//
// So the first Tools call of each agent run lists and filters. ADK resumes a
// confirmed tool call from this list before any ProcessRequest of the run, so
// a tool the filter no longer allows does not run. The later calls, which ADK
// v2.6.0 and later make before every model step, issue no Workflow commands,
// so the toolset and its filter issue them at the same points on every ADK
// release. They return the list ADK v2.5.0 and earlier keep for the whole
// agent run, even when the MCP server, the filter or another Tools call, such
// as one from an InstructionProvider, has changed it since: ADK draws a UUID
// from the workflow's random stream for each tool it keeps before every model
// step, so the IDs it generates, such as tool confirmation IDs, stay the same
// on every release. From the second model step, the proxy tools from that
// list no longer pack themselves, so ProcessRequest alone decides what the
// model request declares and dispatches. In the first step ADK packs them
// before ProcessRequest runs, so that request also declares the tools of the
// first list.
//
// One case still keeps different lists on different releases: an agent that
// reaches the same toolset more than once, for example directly and through
// tool.FilterToolset, when the MCP server changes its tools between those
// listings in the first step. ADK v2.5.0 and earlier keep each list; with
// v2.6.0 the later Tools calls return the first list for all of them.
//
// Agent and model callbacks get contexts that wrap the agent run's context,
// not its invocation, so their Tools calls are not part of the run: each one
// lists and filters, on every release.
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
	allowed := m.allowed(ctx, tools)
	l.keepPending(run, tools, allowed)
	return allowed, nil
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
// ADK v2.5.0 and earlier call Tools in the first model step of an agent run,
// before the request processors, and in later steps only while the agent has
// no tools at all. v2.6.0 and later call Tools before the request processors
// in every step. To schedule the same commands in the same order on every ADK
// release, the toolset lists in Tools only until its first ProcessRequest of
// the agent run, and in ProcessRequest from then on. With only ADK's own
// calls, each model step lists once: the first through Tools, with
// ProcessRequest reusing that list, and every later step here. Other Tools
// calls before the first ProcessRequest, such as an InstructionProvider's,
// list too. From the second step, the request carries this list
// alone, so a tool the MCP server no longer advertises is neither declared nor
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
	l.processed(run)
	for _, t := range m.allowed(ctx, tools) {
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
	runs map[mcpRunKey]*mcpRunState
	// warned reports that fallBack has logged.
	warned bool
}

// mcpRunKey identifies a toolset in one agent run: one agent in one execution
// of a runner.Run iterator.
type mcpRunKey struct {
	toolset *mcpToolset
	// run points weakly to the invocation ADK creates each time an agent runs:
	// in each execution of a runner.Run iterator, a resume, a retry or another
	// pass over a kept iterator included, and in each of the concurrent runs
	// of one agent that share an invocation ID, such as a custom agent running
	// it on several branches. See agentRunOf. The pointer is weak so that the
	// store does not keep a finished run's invocation, and the session events
	// it holds, alive; once the garbage collector reclaims it, startRun drops
	// the state.
	run weak.Pointer[byte]
}

// mcpRunState is the state of a toolset in one agent run.
type mcpRunState struct {
	// pending is the list Tools took for the ProcessRequest call that follows
	// in the same model step, when hasPending is set.
	pending    []tool.Tool
	hasPending bool
	// processed reports that the toolset's ProcessRequest has run in the agent
	// run. From then on ProcessRequest lists, and Tools does not.
	processed bool
	// returned is the list the first Tools call of the agent run returned,
	// when hasReturned is set. Tools returns it again once processed.
	returned    []tool.Tool
	hasReturned bool
}

// listingsFrom returns the mcpListings NewContext installed, or nil.
func listingsFrom(ctx context.Context) *mcpListings {
	l, _ := ctx.Value(mcpListingsKey{}).(*mcpListings)
	return l
}

// agentRunOf returns a weak pointer to the invocation of the agent run ctx
// belongs to, and whether ctx tells it. ADK runs each agent on a new
// invocation, which it passes, wrapped, both to the toolset's Tools and to the
// ProcessRequest calls of the toolset and its tools:
//
//   - Tools gets a ReadonlyContext whose exported InvocationContext field holds
//     the agent run's context.
//   - ProcessRequest gets a tool context, whose unexported context field holds
//     a copy of that context.
//
// The agent run's context, and its copy, hold the invocation in an unexported
// invocationContext field, or are the invocation themselves. It is not the
// invocation ID, agent name or RunConfig: concurrent runs of one agent can
// share all three.
//
// It reads only ADK's own context types, and only their direct fields. Any
// other context, such as one a caller wraps around ADK's, does not tell its
// agent run.
//
// Weak pointers made from the same invocation are equal, and stay distinct
// from those made from any other, even after it is reclaimed.
func agentRunOf(ctx context.Context) (weak.Pointer[byte], bool) {
	v := reflect.ValueOf(ctx)
	for depth := range 4 {
		if v.Kind() == reflect.Interface {
			v = v.Elem()
		}
		if !isADKContext(v) {
			return weak.Pointer[byte]{}, false
		}
		if ic, ok := interfaceField(v, "invocationContext"); ok {
			return weakTo(ic.Elem())
		}
		next, ok := interfaceField(v, "InvocationContext")
		if !ok {
			next, ok = interfaceField(v, "context")
		}
		if !ok {
			if depth == 0 {
				return weak.Pointer[byte]{}, false
			}
			return weakTo(v)
		}
		v = next
	}
	return weak.Pointer[byte]{}, false
}

// isADKContext reports whether v points to a struct of one of the ADK
// packages that define its contexts.
func isADKContext(v reflect.Value) bool {
	if v.Kind() != reflect.Pointer || v.IsNil() || v.Elem().Kind() != reflect.Struct {
		return false
	}
	switch v.Elem().Type().PkgPath() {
	case "google.golang.org/adk/v2/agent", "google.golang.org/adk/v2/internal/context":
		return true
	}
	return false
}

// weakTo returns a weak pointer to the ADK context v points to.
func weakTo(v reflect.Value) (weak.Pointer[byte], bool) {
	if !isADKContext(v) {
		return weak.Pointer[byte]{}, false
	}
	return weak.Make((*byte)(v.UnsafePointer())), true
}

// interfaceField returns the non-nil interface field name of the struct v
// points to, if it is a direct field, not one promoted from an embedded
// struct.
func interfaceField(v reflect.Value, name string) (reflect.Value, bool) {
	sf, ok := v.Elem().Type().FieldByName(name)
	if !ok || len(sf.Index) != 1 {
		return reflect.Value{}, false
	}
	f := v.Elem().FieldByIndex(sf.Index)
	if f.Kind() != reflect.Interface || f.IsNil() {
		return reflect.Value{}, false
	}
	return f, true
}

// startRun returns the state of the toolset in ctx's agent run, for Tools,
// and starts it if needed. When ctx does not tell its agent run, it returns
// nil.
//
// Without a state, the toolset lists and filters on every Tools and
// ProcessRequest call, and its proxy tools pack themselves.
//
// When it starts a state, it also drops the states of finished agent runs,
// whose invocation the garbage collector has reclaimed. No later call can
// belong to such a run, so a long conversation on one NewContext keeps only
// the states of runs that may still call the toolset.
func (l *mcpListings) startRun(m *mcpToolset, ctx agent.ReadonlyContext) *mcpRunState {
	if l == nil {
		return nil
	}
	run, ok := agentRunOf(ctx)
	if !ok {
		l.fallBack(m, ctx)
		return nil
	}
	k := mcpRunKey{m, run}
	l.mu.Lock()
	defer l.mu.Unlock()
	if s, ok := l.runs[k]; ok {
		return s
	}
	for key := range l.runs {
		if key.run.Value() == nil {
			delete(l.runs, key)
		}
	}
	if l.runs == nil {
		l.runs = map[mcpRunKey]*mcpRunState{}
	}
	s := &mcpRunState{}
	l.runs[k] = s
	return s
}

// current returns the state the toolset's Tools call kept for ctx's agent run,
// or nil. ADK calls Tools before ProcessRequest in the first model step of
// every agent run.
func (l *mcpListings) current(m *mcpToolset, ctx context.Context) *mcpRunState {
	if l == nil {
		return nil
	}
	run, ok := agentRunOf(ctx)
	if !ok {
		l.fallBack(m, ctx)
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.runs[mcpRunKey{m, run}]
}

// fallBack logs a warning, once per NewContext and not during replay, that
// the toolset cannot tell ctx's agent run. This happens when ADK changes the
// contexts agentRunOf reads, or when the caller wraps them.
func (l *mcpListings) fallBack(m *mcpToolset, ctx context.Context) {
	wfCtx, ok := workflowContext(ctx)
	if !ok || workflow.IsReplaying(wfCtx) {
		return
	}
	l.mu.Lock()
	warned := l.warned
	l.warned = true
	l.mu.Unlock()
	if !warned {
		workflow.GetLogger(wfCtx).Warn("MCP toolset cannot tell which agent run called it, so it lists on every call; its Histories may not replay with another ADK release",
			"toolset", m.name)
	}
}

// processedList returns the list the first Tools call of the run returned,
// once the toolset's ProcessRequest has run there.
func (l *mcpListings) processedList(s *mcpRunState) ([]tool.Tool, bool) {
	if s == nil {
		return nil, false
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return s.returned, s.processed
}

// keepPending keeps a list Tools took for the ProcessRequest call that
// follows, and, in the first Tools call of the agent run, the list Tools
// returns. Other Tools calls can come before that ProcessRequest, such as one
// from an InstructionProvider; they replace the pending list but not the
// returned one, which ADK v2.5.0 and earlier keep.
func (l *mcpListings) keepPending(s *mcpRunState, tools, returned []tool.Tool) {
	if s == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	s.pending, s.hasPending = tools, true
	if !s.hasReturned {
		s.returned, s.hasReturned = returned, true
	}
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

// processed marks the toolset's ProcessRequest as run in the agent run.
func (l *mcpListings) processed(s *mcpRunState) {
	if s == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	s.processed = true
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
	if out.Error != nil {
		// The tool's own error, such as an MCP isError result. Give ADK an
		// error with the tool's message, so callbacks and the model see it as
		// they do without Temporal. Only the message crosses the Activity
		// boundary, not the error's Go type.
		return nil, errors.New(*out.Error)
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
// surfaces the pause exactly like an in-workflow tool. Error is the message of
// the tool's own error, which the proxy hands to ADK; see CallMcpTool. Result
// then holds {"error": message}, the response ADK builds from a tool error, so
// a workflow worker without the Error field gives the model the same response.
type mcpCallOutput struct {
	Result       map[string]any
	Confirmation *toolconfirmation.ToolConfirmation
	Error        *string
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
//
// A tool's own error, one that wraps no other error and is not an
// ApplicationError, also goes in the output, so the model sees it once, as it
// does without Temporal. ADK's MCP tools return an isError result (and, before
// ADK v2.4.0, a result with no text) that way. They wrap the error of a failed
// MCP request, which still fails the Activity with a retryable McpError, as
// does an error returned after the Activity's context is done, such as when
// the Worker stops. An ApplicationError returned as is fails the Activity with
// its own type and retryability; wrapped in another error, it fails as a
// retryable wrapError.
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
			// An error after ctx is done, such as ctx.Err() when the Worker
			// stops, is not the tool's answer, whatever it wraps.
			if ctx.Err() == nil && isToolAnswer(runErr) {
				msg := runErr.Error()
				return &mcpCallOutput{Result: map[string]any{"error": msg}, Error: &msg}, nil
			}
			return nil, classifyToolError(ErrorTypeMCP, in.Tool, runErr)
		}
		return &mcpCallOutput{Result: res}, nil
	}
	return nil, newApplicationError(ErrorTypeMCP, false, nil,
		"tool %q not found in MCP toolset %q", in.Tool, in.Toolset)
}

// isToolAnswer reports whether err, returned by an MCP tool's Run, is the
// tool's own answer: an error that wraps no other error and is not an
// ApplicationError.
func isToolAnswer(err error) bool {
	var appErr interface{ NonRetryable() bool }
	if errors.As(err, &appErr) {
		return false
	}
	switch e := err.(type) {
	case interface{ Unwrap() error }:
		return e.Unwrap() == nil
	case interface{ Unwrap() []error }:
		return len(e.Unwrap()) == 0
	}
	return true
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
// ProcessRequest), so in the first model step of an agent run a toolset tool
// is packed twice with identical declarations. A name already present is
// skipped, rather than erroring as toolutils.PackTool does on a duplicate.
func packTool(req *model.LLMRequest, t toolutils.Tool) error {
	if req.Tools != nil {
		if _, ok := req.Tools[t.Name()]; ok {
			return nil
		}
	}
	return toolutils.PackTool(req, t)
}
