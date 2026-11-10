/**
 * The Workflow-safe API of the LangGraph plugin: define graphs, entrypoints and
 * tasks, and run them from Workflows.
 *
 * ```ts
 * import { defineLangGraph } from '@temporalio/langgraph/workflow';
 *
 * export const langGraph = defineLangGraph({ graphs: { agent } });
 *
 * export async function agentWorkflow(input: string) {
 *   return langGraph.graph('agent').compile().invoke({ value: input });
 * }
 * ```
 *
 * @module
 * @experimental The LangGraph plugin is an experimental feature; APIs may change without notice.
 */

// Installs the Workflow-sandbox polyfills (gated on `inWorkflowContext()`)
// before any LangChain module is evaluated in the bundle.
// eslint-disable-next-line import/no-unassigned-import
import './load-polyfills';

export { cache, defineLangGraph, LangGraphDefinitions } from './definitions';
export type { LangGraphDefinitionOptions, LangGraphTask, RunOptions } from './definitions';
export type { ExecuteIn, LangGraphActivityOptions, NodeOptions, SummaryFn } from './options';
export type { TaskCache } from './task-cache';
export { UNKNOWN_VALUE_TYPE_FAILURE_TYPE } from './error-types';
