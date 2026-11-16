import { hostname } from 'node:os';
import type {
  SandboxClient,
  SandboxClientOptions,
  SandboxSession,
  SandboxSessionState,
} from '@openai/agents-core/sandbox';
import { ApplicationFailure } from '@temporalio/common';
import {
  SANDBOX_CLIENT_CAN_REUSE_PRESERVED_OWNED_SESSION_SUFFIX,
  SANDBOX_CLIENT_DELETE_SUFFIX,
  SANDBOX_CLIENT_RESUME_SUFFIX,
  SANDBOX_SESSION_DELETE_SUFFIX,
  type SandboxCanReusePreservedOwnedSessionInput,
  type SandboxLifecycleInput,
  type SandboxResumeSessionInput,
  type SandboxSessionResult,
  type SandboxSessionStateInput,
  type SerializedSandboxSessionState,
} from '../common/sandbox-activity-types';
import { isNonRetryable, SandboxProviderBase, type ActivityFunction } from './sandbox-provider';

/** The `ApplicationFailure` type of an operation on a sandbox whose Worker process is gone. */
export const SANDBOX_LOST_FAILURE_TYPE = 'SandboxLost';

export interface OnHostSandboxClientProviderOptions {
  /**
   * Names this Worker process. Its sandboxes' operations go to the Task Queue `<name>@<workerName>`, so
   * only one running Worker may use a name: give each Worker on a machine its own, and let the old
   * process exit before a restart starts the new one.
   *
   * @default the machine's hostname
   */
  workerName?: string;
}

/**
 * A named sandbox client provider for backends that run the sandbox on the Worker's own machine, such
 * as Docker and Unix-local.
 *
 * Creating a sandbox runs on the run's usual Task Queue, so any Worker can create one. The Worker process
 * that creates it keeps the live session and records its own Task Queue, `<name>@<workerName>`, in the
 * session state, and every later operation on that sandbox goes to that queue, which an extra Worker in
 * the same process polls. The plugin starts that Worker with the main one and stops it with the main one.
 *
 * The process never rebuilds a session from saved state. An operation on a session it doesn't hold, such
 * as after a restart, fails non-retryably with type `SandboxLost`, except delete, which does nothing.
 *
 * ```ts
 * const plugin = new OpenAIAgentsPlugin({
 *   modelProvider,
 *   sandboxClientProviders: [
 *     new OnHostSandboxClientProvider('docker', new DockerSandboxClient(), { workerName: 'build-box-1' }),
 *   ],
 * });
 * ```
 *
 * Reference it from Workflow code like any provider: `temporalSandboxClient('docker')`.
 *
 * @experimental Sandbox support is experimental and may change without notice.
 */
export class OnHostSandboxClientProvider extends SandboxProviderBase {
  /** The Task Queue this process's sandboxes are served on: `<name>@<workerName>`. */
  readonly taskQueue: string;

  constructor(
    name: string,
    client: SandboxClient<SandboxClientOptions, SandboxSessionState>,
    options: OnHostSandboxClientProviderOptions = {}
  ) {
    super(name, client);
    this.taskQueue = `${name}@${options.workerName ?? hostname()}`;
  }

  protected override taskQueueOfSessions(): string {
    return this.taskQueue;
  }

  protected override async session(payload: SerializedSandboxSessionState): Promise<SandboxSession> {
    const session = this._sessions.get(payload.sessionId);
    if (session) return session;
    throw ApplicationFailure.create({
      message:
        `Sandbox session '${payload.sessionId}' is lost: the Worker process that created it on Task Queue ` +
        `'${payload.taskQueue ?? this.taskQueue}' has stopped.`,
      type: SANDBOX_LOST_FAILURE_TYPE,
      // A retry would land on this same process.
      nonRetryable: true,
    });
  }

  protected override ownActivities(): Record<string, ActivityFunction> {
    const n = this.name;
    return {
      // The session is the one this process created, so there is nothing to resume.
      [`${n}${SANDBOX_CLIENT_RESUME_SUFFIX}`]: async (
        input: SandboxResumeSessionInput
      ): Promise<SandboxSessionResult> => {
        return this.sessionResult(input.state.sessionId, await this.session(input.state));
      },

      [`${n}${SANDBOX_CLIENT_DELETE_SUFFIX}`]: async (input: SandboxSessionStateInput): Promise<void> => {
        const session = this._sessions.get(input.state.sessionId);
        if (session) await this.deleteHeld(input.state.sessionId, () => this.client.delete?.(session.state));
      },

      [`${n}${SANDBOX_CLIENT_CAN_REUSE_PRESERVED_OWNED_SESSION_SUFFIX}`]: async (
        input: SandboxCanReusePreservedOwnedSessionInput
      ): Promise<boolean> => {
        return this.backendCanReuse((await this.session(input.state)).state, input);
      },

      [`${n}${SANDBOX_SESSION_DELETE_SUFFIX}`]: async (input: SandboxLifecycleInput): Promise<void> => {
        const session = this._sessions.get(input.state.sessionId);
        if (session) await this.deleteHeld(input.state.sessionId, () => this.deleteLiveSession(session, input));
      },
    };
  }

  /**
   * Runs `remove` on a held session and stops holding it, unless it fails with an error Temporal
   * retries: the retry needs the session to still remove the sandbox, since this process never rebuilds one.
   */
  private async deleteHeld(sessionId: string, remove: () => Promise<unknown> | undefined): Promise<void> {
    try {
      await remove();
    } catch (err) {
      if (isNonRetryable(err)) this._sessions.delete(sessionId);
      throw err;
    }
    this._sessions.delete(sessionId);
  }
}
