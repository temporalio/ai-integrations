/**
 * The graphs, entrypoints and tasks of the tests. Both the test Workflows (in
 * the sandbox) and the tests' Workers (for the Activities) import them.
 */

import { AIMessage, BaseMessage } from '@langchain/core/messages';
import { RunnableLambda } from '@langchain/core/runnables';
import {
  Annotation,
  Command,
  END,
  START,
  Send,
  StateGraph,
  entrypoint,
  getConfig,
  getWriter,
  interrupt,
  task,
} from '@langchain/langgraph';
import type { LangGraphRunnableConfig } from '@langchain/langgraph';
import { z } from 'zod';
import { defineLangGraph } from '../workflow';
import type { ExecuteIn, LangGraphActivityOptions, LangGraphDefinitions, NodeOptions, SummaryFn } from '../workflow';

 
type AnyDefinitions = LangGraphDefinitions<any, any>;

const DEFAULTS: LangGraphActivityOptions = { startToCloseTimeout: '10s' };
const NO_RETRY: LangGraphActivityOptions = { startToCloseTimeout: '10s', retry: { maximumAttempts: 1 } };
const EXECUTE_IN: readonly ExecuteIn[] = ['activity', 'workflow'];

/** How many times each Activity node or task ran on the Worker, by key. */
export const executionCounts: Record<string, number> = {};

function count(key: string): void {
  executionCounts[key] = (executionCounts[key] ?? 0) + 1;
}

export function resetExecutionCounts(): void {
  for (const key of Object.keys(executionCounts)) delete executionCounts[key];
}

/** Rejects when `signal` aborts: the Activity was cancelled, e.g. by its timeout. */
function untilAborted(signal: AbortSignal | undefined): Promise<never> {
  return new Promise((_, reject) => {
    if (signal === undefined) throw new Error('No abort signal');
    signal.addEventListener('abort', () => reject(signal.reason), { once: true });
  });
}

const ValueState = Annotation.Root({ value: Annotation<string> });
type Value = typeof ValueState.State;

const ResultsState = Annotation.Root({
  mode: Annotation<string>,
  items: Annotation<string[]>,
  results: Annotation<string[]>({ reducer: (left, right) => left.concat(right), default: () => [] }),
});
type Results = typeof ResultsState.State;

const ItemState = Annotation.Root({ item: Annotation<string> });

function userOf(config: LangGraphRunnableConfig | undefined): string {
  const context = config?.context as { userId?: string; tenant?: { name: string } } | undefined;
  if (context?.userId === undefined) return '';
  return context.tenant ? `${context.userId}:${context.tenant.name}` : context.userId;
}

function typeName(value: unknown): string {
  if (value === null) return 'null';
  if (BaseMessage.isInstance(value)) return AIMessage.isInstance(value) ? 'AIMessage' : 'BaseMessage';
  if (value instanceof Uint8Array) return 'Uint8Array';
  if (value instanceof Date) return 'Date';
  if (value instanceof Command) return 'Command';
  return typeof value;
}

// ---------------------------------------------------------------------------
// Two nodes: context and value types across the Activity boundary.

const TwoNodesState = Annotation.Root({
  value: Annotation<string>,
  message: Annotation<AIMessage>,
  when: Annotation<Date>,
  data: Annotation<Uint8Array>,
  typesB: Annotation<Record<string, string>>,
});

function twoNodesGraph(first: ExecuteIn) {
  return new StateGraph(TwoNodesState)
    .addNode(
      'node_a',
      (state, config) => ({
        value: state.value + 'a' + userOf(config),
        message: new AIMessage({ content: '', tool_calls: [{ name: 'search', args: {}, id: 'call-1' }] }),
        when: new Date('2024-01-02T03:04:05.000Z'),
        data: new Uint8Array([0, 255]),
      }),
      { metadata: { executeIn: first } }
    )
    .addNode(
      'node_b',
      (state, config) => ({
        value: state.value + 'b' + userOf(config),
        typesB: {
          message: `${typeName(state.message)}:${state.message.tool_calls?.map((call: { name: string }) => call.name).join(',')}`,
          when: `${typeName(state.when)}:${state.when instanceof Date ? state.when.toISOString() : ''}`,
          data: `${typeName(state.data)}:${state.data instanceof Uint8Array ? [...state.data].join(',') : ''}`,
        },
      }),
      { metadata: { executeIn: 'activity' } }
    )
    .addEdge(START, 'node_a')
    .addEdge('node_a', 'node_b');
}

export const twoNodes = Object.fromEntries(
  EXECUTE_IN.map((first) => [
    first,
    defineLangGraph({ graphs: { 'my-graph': twoNodesGraph(first) }, defaultActivityOptions: DEFAULTS }),
  ])
) as Record<ExecuteIn, AnyDefinitions>;

// A zod state schema, LangGraph.js's counterpart of a Pydantic state model.
const ZodState = z.object({
  value: z.string(),
  messages: z.array(z.custom<BaseMessage>((value) => BaseMessage.isInstance(value))),
  answer: z.object({ value: z.number() }).optional(),
  answer_type_b: z.string().optional(),
  message_b: z.string().optional(),
});
type ZodValue = z.infer<typeof ZodState>;

function zodStateGraph(first: ExecuteIn) {
  return new StateGraph(ZodState)
    .addNode(
      'node_a',
      () => ({
        answer: { value: 1 },
        messages: [new AIMessage({ content: '', tool_calls: [{ name: 'search', args: {}, id: 'call-1' }] })],
      }),
      { metadata: { executeIn: first } }
    )
    .addNode(
      'node_b',
      (state: ZodValue) => {
        const message = state.messages.at(-1)!;
        const names = (message as AIMessage).tool_calls?.map((call) => call.name) ?? [];
        return {
          answer_type_b: typeName(state.answer),
          message_b: `${typeName(message)}:${names.join(',')}`,
        };
      },
      { metadata: { executeIn: 'activity' } }
    )
    .addEdge(START, 'node_a')
    .addEdge('node_a', 'node_b');
}

export const zodState = Object.fromEntries(
  EXECUTE_IN.map((first) => [
    first,
    defineLangGraph({ graphs: { 'my-graph': zodStateGraph(first) }, defaultActivityOptions: DEFAULTS }),
  ])
) as Record<ExecuteIn, AnyDefinitions>;

// ---------------------------------------------------------------------------
// Command and Send.

const commandGraph = new StateGraph(ValueState)
  .addNode('node_a', (state: Value) => new Command({ update: { value: state.value + 'a' }, goto: 'node_b' }), {
    ends: ['node_b'],
    metadata: { executeIn: 'activity' },
  })
  .addNode('node_b', (state: Value) => new Command({ update: { value: state.value + 'b' }, goto: END }), {
    ends: [END],
    metadata: { executeIn: 'activity' },
  })
  .addEdge(START, 'node_a');

export const command = defineLangGraph({ graphs: { 'my-graph': commandGraph }, defaultActivityOptions: DEFAULTS });

function dispatch(state: Results): Command | Command[] {
  switch (state.mode) {
    case 'single':
      return new Command({ goto: new Send('worker', { item: 'a' }) });
    case 'list':
      return new Command({ goto: [new Send('worker', { item: 'a' }), new Send('worker', { item: 'b' })] });
    case 'mixed':
      return new Command({ goto: ['tail', new Send('worker', { item: 'a' })] });
    case 'update':
      return new Command({ update: { results: ['dispatched'] }, goto: new Send('worker', { item: 'a' }) });
    case 'timeout':
      return new Command({ goto: new Send('worker', { item: 'a' }, { timeout: { runTimeout: 30_000 } }) });
    case 'command_list':
      return [new Command({ update: { results: ['listed'] }, goto: new Send('worker', { item: 'a' }) })];
    default:
      throw new Error(state.mode);
  }
}

const commandSendGraph = new StateGraph(ResultsState)
  .addNode('dispatch', dispatch, { ends: ['worker', 'tail'], metadata: { executeIn: 'activity' } })
  .addNode('worker', (state: typeof ItemState.State) => ({ results: [state.item.toUpperCase()] }), {
    input: ItemState,
    metadata: { executeIn: 'activity' },
  })
  .addNode('tail', (state: Results) => ({ results: [state.mode] }), { metadata: { executeIn: 'activity' } })
  .addEdge(START, 'dispatch')
  .addEdge('worker', END)
  .addEdge('tail', END);

export const commandSend = defineLangGraph({
  graphs: { 'send-graph': commandSendGraph },
  defaultActivityOptions: NO_RETRY,
});

const sendGraph = new StateGraph(ResultsState)
  .addNode('worker', (state: typeof ItemState.State) => ({ results: [state.item.toUpperCase()] }), {
    input: ItemState,
    metadata: { executeIn: 'activity' },
  })
  .addConditionalEdges(START, (state: Results) => state.items.map((item) => new Send('worker', { item })), ['worker'])
  .addEdge('worker', END);

export const send = defineLangGraph({ graphs: { 'my-graph': sendGraph }, defaultActivityOptions: DEFAULTS });

// ---------------------------------------------------------------------------
// Timeouts, sync nodes, Workflow nodes, metadata.

const timeoutGraph = new StateGraph(ValueState)
  .addNode('node', async (_state: Value, config: LangGraphRunnableConfig) => {
    await untilAborted(config.signal);
    return { value: 'done' };
  }, { metadata: { executeIn: 'activity' } })
  .addEdge(START, 'node');

export const timeout = defineLangGraph({
  graphs: { 'my-graph': timeoutGraph },
  defaultActivityOptions: { startToCloseTimeout: '100ms', retry: { maximumAttempts: 1 } },
});

export const syncNode = defineLangGraph({
  graphs: {
    'my-graph': new StateGraph(ValueState)
      .addNode('sync_node', (state: Value) => ({ value: state.value + '!' }), { metadata: { executeIn: 'activity' } })
      .addEdge(START, 'sync_node'),
  },
  defaultActivityOptions: DEFAULTS,
});

export const executeInWorkflow = defineLangGraph({
  graphs: {
    'my-graph': new StateGraph(ValueState)
      .addNode('node', async () => ({ value: 'done' }), { metadata: { executeIn: 'workflow' } })
      .addEdge(START, 'node'),
  },
});

export const nodeMetadata = defineLangGraph({
  graphs: {
    'my-graph': new StateGraph(ValueState)
      .addNode(
        'node',
        async (state: Value, config: LangGraphRunnableConfig) => ({
          value: state.value + String(config.metadata?.my_key ?? 'NOT_FOUND') + `:${'startToCloseTimeout' in (config.metadata ?? {})}`,
        }),
        { metadata: { executeIn: 'activity', startToCloseTimeout: '10s', my_key: 'my_value' } }
      )
      .addEdge(START, 'node'),
  },
});

// ---------------------------------------------------------------------------
// Interrupts.

export const interruptGraph = defineLangGraph({
  graphs: {
    'my-graph': new StateGraph(ValueState)
      .addNode('node', async () => ({ value: interrupt('Continue?') as string }), { metadata: { executeIn: 'activity' } })
      .addEdge(START, 'node'),
  },
  defaultActivityOptions: DEFAULTS,
});

// ---------------------------------------------------------------------------
// Streaming.

function tokenNode(state: Value) {
  const writer = getWriter();
  for (const token of ['a', 'b', 'c']) writer?.({ token });
  writer?.({ done: true });
  return { value: state.value + 'abc' };
}

async function asyncTokenNode(state: Value) {
  return tokenNode(state);
}

export const streaming = Object.fromEntries(
  EXECUTE_IN.flatMap((executeIn) =>
    (['sync', 'async'] as const).map((kind) => [
      `${executeIn}-${kind}`,
      defineLangGraph({
        graphs: {
          'streaming-ws': new StateGraph(ValueState)
            .addNode('token_node', kind === 'sync' ? tokenNode : asyncTokenNode, { metadata: { executeIn } })
            .addEdge(START, 'token_node'),
        },
        defaultActivityOptions: DEFAULTS,
        streamingTopic: 'tokens',
      }),
    ])
  )
) as Record<string, AnyDefinitions>;

export const astreamPublish = defineLangGraph({
  graphs: {
    'astream-publish': new StateGraph(ValueState)
      .addNode('node_a', (state: Value) => ({ value: state.value + 'a' }), { metadata: { executeIn: 'activity' } })
      .addNode('node_b', (state: Value) => ({ value: state.value + 'b' }), { metadata: { executeIn: 'activity' } })
      .addEdge(START, 'node_a')
      .addEdge('node_a', 'node_b'),
  },
  defaultActivityOptions: DEFAULTS,
});

// ---------------------------------------------------------------------------
// Subgraphs.

async function childNode() {
  return { value: 'child' };
}

export const activitySubgraph = defineLangGraph({
  graphs: {
    parent: new StateGraph(ValueState)
      .addNode(
        'parent_node',
        async (state: Value) =>
          new StateGraph(ValueState).addNode('child_node', childNode).addEdge(START, 'child_node').compile().invoke(state),
        { metadata: { executeIn: 'activity' } }
      )
      .addEdge(START, 'parent_node'),
  },
  defaultActivityOptions: DEFAULTS,
});

export const workflowSubgraph: AnyDefinitions = defineLangGraph({
  graphs: {
    child: new StateGraph(ValueState)
      .addNode('child_node', childNode, { metadata: { executeIn: 'activity', startToCloseTimeout: '10s' } })
      .addEdge(START, 'child_node'),
    parent: new StateGraph(ValueState)
      .addNode(
        'parent_node',
        async (state: Value): Promise<Value> => (workflowSubgraph.graph('child') as any).compile().invoke(state),
        { metadata: { executeIn: 'workflow' } }
      )
      .addEdge(START, 'parent_node'),
  },
});

// ---------------------------------------------------------------------------
// Continue-as-new.

export const continueAsNew = defineLangGraph({
  graphs: {
    'my-graph': new StateGraph(ValueState)
      .addNode('node', async (state: Value) => ({ value: state.value + 'a' }), { metadata: { executeIn: 'activity' } })
      .addEdge(START, 'node'),
  },
  defaultActivityOptions: DEFAULTS,
});

const NumberState = Annotation.Root({ value: Annotation<number> });
type NumberValue = typeof NumberState.State;
const CACHED: NodeOptions = { executeIn: 'activity', startToCloseTimeout: '10s' };
const CACHED_NO_RETRY: NodeOptions = { ...CACHED, retry: { maximumAttempts: 1 } };

export const cachedGraph = defineLangGraph({
  graphs: {
    'cached-graph': new StateGraph(NumberState)
      .addNode('multiply_by_3', async (state: NumberValue) => (count('multiply'), { value: state.value * 3 }), {
        metadata: CACHED,
      })
      .addNode('add_100', async (state: NumberValue) => (count('add'), { value: state.value + 100 }), {
        metadata: CACHED,
      })
      .addNode('double', async (state: NumberValue) => (count('double'), { value: state.value * 2 }), {
        metadata: CACHED,
      })
      .addEdge(START, 'multiply_by_3')
      .addEdge('multiply_by_3', 'add_100')
      .addEdge('add_100', 'double'),
  },
});

export const cachedFanOut = defineLangGraph({
  graphs: {
    'cached-fan-out': new StateGraph(ResultsState)
      .addNode(
        'dispatch',
        async (state: Results) => {
          count('dispatch');
          return new Command({ goto: state.items.map((item) => new Send('upper', { item })) });
        },
        { ends: ['upper'], metadata: CACHED_NO_RETRY }
      )
      .addNode(
        'upper',
        async (state: typeof ItemState.State) => {
          count(`upper:${state.item}`);
          return { results: [state.item.toUpperCase()] };
        },
        { input: ItemState, metadata: CACHED_NO_RETRY }
      )
      .addEdge(START, 'dispatch')
      .addEdge('upper', END),
  },
});

export const cachedCommand = defineLangGraph({
  graphs: {
    'cached-command': new StateGraph(NumberState)
      .addNode(
        'goto_a',
        (state: NumberValue) => {
          count('goto_a');
          return new Command({ update: { value: state.value * 10 }, goto: 'goto_b' });
        },
        { ends: ['goto_b'], metadata: CACHED_NO_RETRY }
      )
      .addNode(
        'goto_b',
        (state: NumberValue) => {
          count('goto_b');
          return new Command({ update: { value: state.value + 1 }, goto: END });
        },
        { ends: [END], metadata: CACHED_NO_RETRY }
      )
      .addEdge(START, 'goto_a'),
  },
});

/** A state key that, with a Command-shaped value, looks like a cached Command. */
export const MARKER = '__temporal_langgraph_command__';
const MarkerState = Annotation.Root({ [MARKER]: Annotation<unknown> });

export const cachedOrdinary = defineLangGraph({
  graphs: {
    'cached-ordinary': new StateGraph(MarkerState)
      .addNode(
        'ordinary_result',
        (state: typeof MarkerState.State) => {
          count('ordinary');
          return { [MARKER]: state[MARKER] };
        },
        { metadata: CACHED_NO_RETRY }
      )
      .addEdge(START, 'ordinary_result'),
  },
});

// ---------------------------------------------------------------------------
// Summaries.

const passthrough = async (state: Value) => ({ value: state.value });

function summaryGraph(options: Partial<NodeOptions> = {}) {
  return new StateGraph(ValueState)
    .addNode('node', passthrough, { metadata: { executeIn: 'activity', ...options } })
    .addEdge(START, 'node');
}

const summarize: SummaryFn = (args) => `value=${(args[0] as Value).value}`;

function summaryDefinitions(options: Partial<NodeOptions>, defaults: LangGraphActivityOptions = {}) {
  return defineLangGraph({
    graphs: { 'summary-graph': summaryGraph(options) },
    defaultActivityOptions: { startToCloseTimeout: '10s', ...defaults },
  });
}

export const summaries = {
  summaryFn: summaryDefinitions({ summaryFn: summarize }),
  summaryFnNull: summaryDefinitions({ summaryFn: () => null }),
  summaryFnEmpty: summaryDefinitions({ summaryFn: () => '' }),
  staticSummary: summaryDefinitions({ summary: 'static' }),
  nodeStaticOverDefaultFn: summaryDefinitions({ summary: 'node-static' }, { summaryFn: () => 'global' }),
  nodeFnOverDefaultStatic: summaryDefinitions({ summaryFn: () => 'node-fn' }, { summary: 'global-static' }),
  defaultFn: summaryDefinitions({}, { summaryFn: () => 'global' }),
  hidesSummaryFn: defineLangGraph({
    graphs: {
      'summary-graph': new StateGraph(ValueState)
        .addNode(
          'node',
          async (state: Value, config: LangGraphRunnableConfig) => ({
            value: `${state.value}-has_fn=${'summaryFn' in (config.metadata ?? {})}`,
          }),
          { metadata: { executeIn: 'activity', summaryFn: () => 'dynamic', my_key: 'my_value' } }
        )
        .addEdge(START, 'node'),
    },
    defaultActivityOptions: DEFAULTS,
  }),
  workflowNode: defineLangGraph({
    graphs: {
      'summary-graph': new StateGraph(ValueState)
        .addNode('node', passthrough, {
          metadata: { executeIn: 'workflow', summaryFn: (args: unknown[]) => `wf:${(args[0] as Value).value}` },
        })
        .addEdge(START, 'node'),
    },
  }),
  workflowNodeClears: defineLangGraph({
    graphs: {
      'summary-graph': new StateGraph(ValueState)
        .addNode('a', passthrough, { metadata: { executeIn: 'workflow', summaryFn: () => 'first' } })
        .addNode('b', passthrough, { metadata: { executeIn: 'workflow', summaryFn: () => null } })
        .addEdge(START, 'a')
        .addEdge('a', 'b'),
    },
  }),
};

// ---------------------------------------------------------------------------
// Functional API.

const doubleValue = task('double_value', (x: number) => x * 2);
const addTen = task('add_ten', (x: number) => x + 10);
const tripleValue = task('triple_value', (x: number) => x * 3);
const addFive = task('add_five', (x: number) => x + 5);

function inActivity(...names: string[]): Record<string, NodeOptions> {
  return Object.fromEntries(names.map((name) => [name, { executeIn: 'activity' as const }]));
}

const FUNCTIONAL_DEFAULTS: LangGraphActivityOptions = { startToCloseTimeout: '30s' };

export const simpleFunctional = defineLangGraph({
  entrypoints: {
    e2e_simple_functional: entrypoint('e2e_simple_functional', async (value: number) => ({
      result: await addTen(await doubleValue(value)),
    })),
  },
  tasks: [doubleValue, addTen],
  activityOptions: inActivity('double_value', 'add_ten'),
  defaultActivityOptions: FUNCTIONAL_DEFAULTS,
});

export const simpleFunctionalMixed = defineLangGraph({
  entrypoints: {
    e2e_simple_functional: entrypoint('e2e_simple_functional', async (value: number) => ({
      result: await addFive(await tripleValue(value)),
    })),
  },
  tasks: [tripleValue, addFive],
  activityOptions: { triple_value: { executeIn: 'activity' }, add_five: { executeIn: 'workflow' } },
  defaultActivityOptions: FUNCTIONAL_DEFAULTS,
});

interface Values {
  message: AIMessage;
  when: Date;
  data: Uint8Array;
  nested: { messages: AIMessage[] };
}

function valueTypes(values: Values): Record<string, string> {
  return {
    message: typeName(values.message),
    when: typeName(values.when),
    data: typeName(values.data),
    nested: typeName(values.nested.messages[0]),
  };
}

const makeValues = task(
  'make_values',
  (value: number): Values => ({
    message: new AIMessage({ content: `hi ${value}`, id: 'm1' }),
    when: new Date(0),
    data: new Uint8Array([0, 255]),
    nested: { messages: [new AIMessage('nested')] },
  })
);
const describeValues = task('describe_values', (values: Values) => valueTypes(values));

export const typedValuesFunctional = defineLangGraph({
  entrypoints: {
    e2e_simple_functional: entrypoint('e2e_simple_functional', async (value: number) => {
      const values = await makeValues(value);
      return { result: [valueTypes(values), await describeValues(values)] };
    }),
  },
  tasks: [makeValues, describeValues],
  activityOptions: inActivity('make_values', 'describe_values'),
  defaultActivityOptions: FUNCTIONAL_DEFAULTS,
});

const readContext = task('read_context', (value: number) => {
  const context = getConfig()?.context as { userId?: string; tenant?: { name: string } } | undefined;
  return {
    value,
    user_id: context?.userId ? (context.tenant ? `${context.userId}:${context.tenant.name}` : context.userId) : '',
    context_type: context === undefined || context === null ? 'none' : typeof context,
  };
});

export const contextFunctional = defineLangGraph({
  entrypoints: {
    context: entrypoint('context', async (value: number) => readContext(value)),
  },
  tasks: [readContext],
  activityOptions: inActivity('read_context'),
  defaultActivityOptions: FUNCTIONAL_DEFAULTS,
});

const expensiveTaskA = task('expensive_task_a', (x: number) => (count('task_a'), x * 3));
const expensiveTaskB = task('expensive_task_b', (x: number) => (count('task_b'), new AIMessage(String(x + 100))));
const expensiveTaskC = task('expensive_task_c', (message: AIMessage) => (count('task_c'), Number(message.content) * 2));

export const continueAsNewFunctional = defineLangGraph({
  entrypoints: {
    e2e_continue_as_new_functional: entrypoint('e2e_continue_as_new_functional', async (value: number) => {
      const a = await expensiveTaskA(value);
      const b = await expensiveTaskB(a);
      return { result: await expensiveTaskC(b), task_b_type: typeName(b) };
    }),
  },
  tasks: [expensiveTaskA, expensiveTaskB, expensiveTaskC],
  activityOptions: inActivity('expensive_task_a', 'expensive_task_b', 'expensive_task_c'),
  defaultActivityOptions: FUNCTIONAL_DEFAULTS,
});

const steps = [
  task('step_1', (x: number) => (count('step_1'), x * 2)),
  task('step_2', (x: number) => (count('step_2'), x + 5)),
  task('step_3', (x: number) => (count('step_3'), x * 3)),
  task('step_4', (x: number) => (count('step_4'), x - 10)),
  task('step_5', (x: number) => (count('step_5'), x + 100)),
];

export const partialExecution = defineLangGraph({
  entrypoints: {
    e2e_partial_execution: entrypoint(
      'e2e_partial_execution',
      async ({ value, stopAfter = 5 }: { value: number; stopAfter?: number }) => {
        let result = value;
        for (let i = 0; i < stopAfter; i++) result = await steps[i]!(result);
        return { result, completed_tasks: stopAfter };
      }
    ),
  },
  tasks: steps,
  activityOptions: inActivity('step_1', 'step_2', 'step_3', 'step_4', 'step_5'),
  defaultActivityOptions: FUNCTIONAL_DEFAULTS,
});

const askHuman = task('ask_human', (question: string) => interrupt(question) as string);

export const interruptFunctional = defineLangGraph({
  entrypoints: {
    v2_interrupt: entrypoint('v2_interrupt', async (value: string) => ({
      input: value,
      answer: await askHuman('Do you approve?'),
    })),
  },
  tasks: [askHuman],
  activityOptions: inActivity('ask_human'),
  defaultActivityOptions: FUNCTIONAL_DEFAULTS,
});

const waitingTask = task('waiting_task', async (x: number) => {
  await untilAborted(getConfig()?.signal);
  return x;
});

export const slowFunctional = defineLangGraph({
  entrypoints: {
    e2e_slow_functional: entrypoint('e2e_slow_functional', async (value: number) => ({
      result: await waitingTask(value),
    })),
  },
  tasks: [waitingTask],
  defaultActivityOptions: FUNCTIONAL_DEFAULTS,
  activityOptions: {
    waiting_task: { executeIn: 'activity', startToCloseTimeout: '100ms', retry: { maximumAttempts: 1 } },
  },
});

const syncTask = task('sync_task', (x: number) => x + 1);

export const syncTaskFunctional = defineLangGraph({
  entrypoints: {
    'sync-task': entrypoint('sync-task', async (value: number) => ({ result: await syncTask(value) })),
  },
  tasks: [syncTask],
  activityOptions: inActivity('sync_task'),
  defaultActivityOptions: DEFAULTS,
});

/** A node whose runnable is not a function: the plugin can't run it as an Activity. */
export const lambdaNodeGraph = new StateGraph(ValueState)
  .addNode('node', RunnableLambda.from((state: Value) => ({ value: state.value })))
  .addEdge(START, 'node');

export { ValueState };

/** Every definitions object, by the key tests and Workflows use. */
export const registry: Record<string, AnyDefinitions> = {
  'twoNodes.activity': twoNodes.activity,
  'twoNodes.workflow': twoNodes.workflow,
  'zodState.activity': zodState.activity,
  'zodState.workflow': zodState.workflow,
  command,
  commandSend,
  send,
  timeout,
  syncNode,
  executeInWorkflow,
  nodeMetadata,
  interruptGraph,
  ...Object.fromEntries(Object.entries(streaming).map(([key, value]) => [`streaming.${key}`, value])),
  astreamPublish,
  activitySubgraph,
  workflowSubgraph,
  continueAsNew,
  cachedGraph,
  cachedFanOut,
  cachedCommand,
  cachedOrdinary,
  ...Object.fromEntries(Object.entries(summaries).map(([key, value]) => [`summaries.${key}`, value])),
  simpleFunctional,
  simpleFunctionalMixed,
  typedValuesFunctional,
  contextFunctional,
  continueAsNewFunctional,
  partialExecution,
  interruptFunctional,
  slowFunctional,
  syncTaskFunctional,
};
