const { TemporalAgent } = require('@temporalio/strands-agents');

exports.agentWorkflow = async function agentWorkflow(prompt) {
  const agent = new TemporalAgent({
    printer: false,
    activityOptions: { startToCloseTimeout: '60 seconds' },
  });
  const result = await agent.invoke(prompt);
  return result.toString();
};
