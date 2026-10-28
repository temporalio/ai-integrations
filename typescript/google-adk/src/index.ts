/**
 * Worker-side API for the Google ADK Temporal plugin.
 *
 * @experimental The Google ADK plugin is an experimental feature; APIs may change without notice.
 *
 * @module
 */

// Side-effect import, first: answers ADK's undeclared `uuid` import on the
// Worker before `./plugin` reaches `@google/adk`.
// eslint-disable-next-line import/no-unassigned-import
import './adk-uuid-resolution';

export { GoogleAdkPlugin } from './plugin';
export type { GoogleAdkPluginOptions } from './plugin';
export type { MCPToolsetFactory } from './mcp';
