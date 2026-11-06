/**
 * The `ApplicationFailure.type` values this plugin raises.
 *
 * @module
 */

/**
 * Error type when a value crossing the Activity boundary carries a type ID this
 * Worker can't rebuild, because Workers run incompatible versions of the
 * plugin. Non-retryable. Raised inside the Activity when it decodes its input,
 * so it arrives wrapped in an `ActivityFailure` (match it through the `.cause`
 * chain), or in the Workflow when it decodes an Activity result or a cached
 * value, where it fails the Workflow.
 */
export const UNKNOWN_VALUE_TYPE_FAILURE_TYPE = 'LangGraphUnknownValueType';
