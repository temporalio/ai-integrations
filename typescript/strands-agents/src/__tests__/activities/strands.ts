// Module-level sink: records the args getWeather was invoked with so tests can
// assert the activity actually ran, rather than relying on the stub model's
// scripted reply (which would pass even if the activity were never dispatched).
export const getWeatherCalls: Array<{ location: string }> = [];

export async function getWeather(input: { location: string }): Promise<{ city: string; conditions: string }> {
  getWeatherCalls.push(input);
  return {
    city: input.location,
    conditions: 'Sunny.',
  };
}

// Module-level sink: written by the audit activity, read in assertions.
// Activity bodies run in worker context, not the workflow sandbox, so a
// plain array is fine.
export const auditLog: string[] = [];

export async function auditTool(toolName: string): Promise<void> {
  auditLog.push(toolName);
}

// Counts attempts so the activity raises on the first invocation and
// succeeds on the second — modeling a real "approval flipped an external
// flag" check.
export const deleteState = { calls: 0 };

/** The interrupt-shaped error the plugin rewrites: an `Error` with an `interrupts` array. */
export function interruptError(name: string): Error {
  return Object.assign(new Error('interrupt'), {
    interrupts: [
      {
        toJSON: () => ({ id: `delete:${name}`, name: 'approval', reason: `delete ${name}?`, source: 'tool' }),
      },
    ],
  });
}

// Counts calls to confirmThings, which always asks for two decisions.
export const confirmState = { calls: 0 };

export async function confirmThings(input: { name: string }): Promise<string> {
  confirmState.calls += 1;
  throw Object.assign(new Error('interrupt'), {
    interrupts: ['approval', 'confirmation'].map((name) => ({
      toJSON: () => ({ id: `${name}:${input.name}`, name, reason: `${name} for ${input.name}?`, source: 'tool' }),
    })),
  });
}

export async function deleteThing(input: { name: string }): Promise<string> {
  deleteState.calls += 1;
  if (deleteState.calls === 1) {
    // Throw an interrupt-shaped error: the strands plugin's activity interceptor
    // detects `interrupts: InterruptLike[]` structurally and re-raises an
    // ApplicationFailure of type STRANDS_INTERRUPT_TYPE that the workflow-side
    // TemporalActivityTool routes through `toolContext.interrupt()`.
    throw interruptError(input.name);
  }
  return `deleted ${input.name}`;
}
