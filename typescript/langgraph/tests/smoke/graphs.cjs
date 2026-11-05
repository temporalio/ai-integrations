const { Annotation, START, StateGraph, entrypoint, task } = require('@langchain/langgraph');
const { defineLangGraph } = require('@temporalio/langgraph/workflow');

const State = Annotation.Root({ value: Annotation() });

const graph = new StateGraph(State)
  .addNode('in_activity', (state) => ({ value: `${state.value} activity` }), { metadata: { executeIn: 'activity' } })
  .addNode('in_workflow', (state) => ({ value: `${state.value} workflow` }), { metadata: { executeIn: 'workflow' } })
  .addEdge(START, 'in_activity')
  .addEdge('in_activity', 'in_workflow');

const shout = task('shout', (text) => text.toUpperCase());

exports.langGraph = defineLangGraph({
  graphs: { smoke: graph },
  entrypoints: { smoke_entrypoint: entrypoint('smoke_entrypoint', async (text) => shout(text)) },
  tasks: [shout],
  activityOptions: { shout: { executeIn: 'activity' } },
  defaultActivityOptions: { startToCloseTimeout: '30 seconds' },
});
