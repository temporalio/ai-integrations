const { langGraph } = require('./graphs.cjs');

exports.graphWorkflow = async function graphWorkflow(input) {
  const result = await langGraph.graph('smoke').compile().invoke({ value: input });
  return result.value;
};

exports.entrypointWorkflow = async function entrypointWorkflow(input) {
  return langGraph.entrypoint('smoke_entrypoint').invoke(input);
};
