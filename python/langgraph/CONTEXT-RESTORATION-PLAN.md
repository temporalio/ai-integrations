# Typed runtime context

## Decisions

- Restore declared dataclass and supported Pydantic v2 contexts in Graph API
  Activity nodes and Functional API Activity tasks.
- Make restoration the default. This is a breaking change for code that reads
  class-based contexts as dictionaries. Document migration to attribute access.
- Keep TypedDict and plain dictionary contexts as dictionaries.
- Keep context separate from task arguments, task results, and graph state.
- Leave backwards compatibility to the separate bonus project below. Do not add
  a compatibility flag or legacy adapter to the main implementation.

These decisions extend [PR #126](https://github.com/temporalio/ai-integrations/pull/126)
and its dataclass context fix. The starting revision is
`c3a4ef02e94c6ecc4cb8fad3f43190cef987f307`.

## Main implementation

1. Collect supported `context_schema` classes from registered graphs and
   entrypoints. Give each class an ID based on its module and qualified name.
   Reject conflicting registrations. Both the Workflow and Activity sides use
   the same registry.
2. Before scheduling an Activity, identify the active context's registered type.
   Add its string ID to `langgraph_config` beside the context data. Choose the ID
   for each call: a task can be shared by entrypoints with different contexts.
   Reject unregistered typed contexts rather than guessing a schema.
3. On the Activity worker, resolve the ID through the registry. Do not load a
   class from a payload-supplied import path. Graph nodes can also use their
   already captured schema.
4. Restore context before installing LangGraph Runtime. Keep the current
   dataclass constructor handling, including saved `init=False` fields. For
   supported Pydantic v2 models, call `model_validate`. Preserve already restored
   instances and absent context. Report unknown IDs and invalid models clearly.
5. Document the new default and the required Pydantic payload converter. The
   plugin does not replace the application's converter.

Relevant files:

- `_plugin.py`: schema registration and wrapper setup.
- `_activity.py`: attach and resolve schema IDs at the Activity boundary.
- `_langgraph_config.py`: shared type checks and reconstruction.
- Existing graph, Functional API, validation, and replay tests.
- README and this plan. Release notes come from commit messages; no changelog.

Example fields inside `langgraph_config`:

```json
{
  "context_schema_id": "app.UserContext",
  "context": {"user_id": "alice"}
}
```

The ID is data. The class itself stays in the worker registry. This extends the
serialized Activity input; it does not preserve the previous payload contract.

## Scope

Support standard object-shaped Pydantic v2 BaseModel contexts, including nested
BaseModels. Test aliases against the configured converter's actual output.
Validation runs again after decoding, so validators must be safe to repeat.

Recursive restoration of arbitrary nested dataclasses, custom serializers that
change the context shape, RootModel payloads, Pydantic v1, LangGraph Store
transport, and the Functional API's Python 3.10 limitation are outside this work.
Restoration creates a worker-side object; it does not share object identity or
automatically return context mutations to the Workflow.

## Validation

Extend existing tests first:

- A real Worker round trip for dataclass and Pydantic instances, plus dictionary
  input with a declared class schema.
- Both explicit Runtime parameters and `get_runtime()` where supported.
- One Functional API task shared by entrypoints using different schemas,
  including concurrent runs.
- Pydantic nested models, aliases, and clear invalid-input failures.
- None, TypedDict, and saved dataclass `init=False` fields as controls.
- Schema registration conflicts, missing registrations, and unknown IDs.
- Replay of histories produced by the new implementation.

Run the plugin's `make lint`, `make test`, and `make build` targets from this
worktree. They resolve their project from the current plugin directory. The
installed local uv is 0.11.21, while the repository requires >=0.12.5,<0.13.
Record any resulting local blocker and use the PR's locked CI matrix for
supported-package checks. Do not install tools without permission.

Initial implementation checks:

- A source-loaded dispatch probe passed concurrent dataclass, Pydantic,
  dictionary, and absent contexts, including aliases and a nested model.
- Using the installed Temporal CLI, source-loaded runs of the changed test
  functions passed all 14 graph context cases, three graph replay cases,
  concurrent Functional API contexts and their replay, and schema/payload error
  cases. These checks used cached SDK 1.33.0 and LangGraph 1.2.9 dependencies;
  they are not the supported-package test suite.
- Formatting passed through the make target with an already cached Ruff 0.15.22
  executable. No tools or dependencies were installed.
- The standard local lint, test, and build targets stop at the uv version
  requirement. The locked PR CI matrix must validate the supported SDK and
  package installation before delivery.

## Bonus project: preserve compatibility

This project is separate and does not gate the main implementation.

1. Preserve the old caller-visible dictionary type through an explicit legacy
   registration, versioned adapter, or old worker deployment. Do not change the
   new default. A schema alone cannot identify which behavior an old caller
   expects.
2. Define how to read old payloads without schema IDs and retain aliases or
   versioned schemas for renamed classes. Do not infer classes from field names.
3. Plan routing for existing Workflow runs and pending or retried Activities.
   Consider worker deployment versioning, separate task queues, or draining old
   work before cutover. Apply Workflow patching only where commands change.
4. Save old payload and history fixtures and verify each supported upgrade path,
   including old application code, new code, retries, and worker routing.

Deliver an upgrade guide and tested support matrix. Decoding old bytes,
preserving application behavior, and continuing old Workflow runs are separate
compatibility promises.
