import { hostname } from 'node:os';
import type {
  SandboxClient,
  SandboxClientOptions,
  SandboxSession,
  SandboxSessionState,
} from '@openai/agents-core/sandbox';
import { Context } from '@temporalio/activity';
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
   * Names this Worker process. Its sandboxes' operations go to the Task Queue
   * `<Task Queue>/<name>@<workerName>`, where `<Task Queue>` is the Worker's own. For each Task Queue,
   * only one running process may use a name: give each process on a machine that serves the queue its
   * own, and let the old process exit before a restart starts the new one. Workers in one process can
   * share a name.
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
 * that creates it keeps the live session and records a Task Queue of its own,
 * `<Task Queue>/<name>@<workerName>`, in the session state, where `<Task Queue>` is the queue the create
 * ran on. Every later operation on that sandbox goes to that queue, which an extra Worker beside the main
 * one polls. The plugin starts that Worker with the main one and stops it with the main one.
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
  private readonly workerName: string;

  constructor(
    name: string,
    client: SandboxClient<SandboxClientOptions, SandboxSessionState>,
    options: OnHostSandboxClientProviderOptions = {}
  ) {
    super(name, client);
    this.workerName = options.workerName ?? hostname();
  }

  /**
   * The Task Queue of the extra Worker beside a main Worker on `mainTaskQueue`.
   *
   * @internal
   */
  _taskQueueFor(mainTaskQueue: string): string {
    return `${mainTaskQueue}/${this.name}@${this.workerName}`;
  }

  // The create Activity runs on the main Worker, so its own queue names the extra Worker beside it.
  protected override createdTaskQueue(): string {
    return this._taskQueueFor(Context.current().info.taskQueue);
  }

  protected override async session(payload: SerializedSandboxSessionState): Promise<SandboxSession> {
    const session = this._sessions.get(payload.sessionId);
    if (session) return session;
    const where = payload.taskQueue === undefined ? '' : ` on Task Queue '${payload.taskQueue}'`;
    throw ApplicationFailure.create({
      message: `Sandbox session '${payload.sessionId}' is lost: the Worker process that created it${where} has stopped.`,
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
        return this.sessionResult(input.state.sessionId, await this.session(input.state), input.state.taskQueue);
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
