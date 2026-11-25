import type {
  SandboxClient,
  SandboxClientCreateArgs,
  SandboxClientOptions,
  SandboxClientResumeOptions,
  SandboxPreservedSessionReuseOptions,
  SandboxRunConfig,
  SandboxSessionLike,
  SandboxSessionSerializationOptions,
} from '@openai/agents-core/sandbox';
import { Manifest, cloneManifest } from '@openai/agents-core/sandbox';
import { ApplicationFailure } from '@temporalio/common';
import { scheduleActivity, type ActivityOptions } from '@temporalio/workflow';
import {
  SANDBOX_CLIENT_CAN_REUSE_PRESERVED_OWNED_SESSION_SUFFIX,
  SANDBOX_CLIENT_CREATE_SUFFIX,
  SANDBOX_CLIENT_DELETE_SUFFIX,
  SANDBOX_CLIENT_RESUME_SUFFIX,
  SANDBOX_CLIENT_SERIALIZE_SESSION_STATE_SUFFIX,
  SANDBOX_CLIENT_WORKSPACE_ROOT_SUFFIX,
  decodeManifest,
  encodeManifest,
  reviveWorkflowSessionState,
  sandboxSpanName,
  serializeWorkflowSessionState,
  type EncodedManifest,
  type SandboxSessionResult,
  type SerializedSandboxSessionState,
  type TemporalSandboxSessionState,
} from '../common/sandbox-activity-types';
import { TemporalSandboxSession, sessionActivityOptions } from './sandbox-session';
import { getCurrentPluginConfig, workspaceRoots } from './plugin-config-store';
import { maybeTemporalSpan } from './span-helpers';

/**
 * Workflow-side sandbox client. Holds no connection to any sandbox backend —
 * session creation, resumption, and every session operation are dispatched as
 * Activities to the sandbox client provider registered under the same name on
 * the Worker. An operation on a session whose state names a Task Queue, which
 * an `OnHostSandboxClientProvider` records, goes to that queue.
 *
 * Create instances via {@link temporalSandboxClient}.
 */
export class TemporalSandboxClient implements SandboxClient<SandboxClientOptions, TemporalSandboxSessionState> {
  readonly backendId: string;
  /** Manifest environment references are resolved by the sandbox backend on the Worker, never in the Workflow. */
  readonly environmentPreparation = 'client' as const;
  private readonly _name: string;
  private readonly _config: ActivityOptions;

  constructor(name: string, config?: ActivityOptions) {
    this._name = name;
    // Upstream has its own handling for some backend ids, such as `docker`, so this one can't be a provider's name.
    this.backendId = `temporal:${name}`;
    this._config = config ?? { startToCloseTimeout: '5 minutes' };
  }

  async create(
    argsOrManifest?: SandboxClientCreateArgs | Manifest,
    manifestOptions?: SandboxClientOptions
  ): Promise<SandboxSessionLike<TemporalSandboxSessionState>> {
    const args: SandboxClientCreateArgs =
      argsOrManifest instanceof Manifest
        ? { manifest: argsOrManifest, options: manifestOptions }
        : argsOrManifest ?? {};
    let manifest: EncodedManifest | undefined;
    if (args.manifest !== undefined) {
      manifest = encodeManifest(args.manifest instanceof Manifest ? args.manifest : new Manifest(args.manifest));
    }
    const result = await scheduleActivity<SandboxSessionResult>(
      `${this._name}${SANDBOX_CLIENT_CREATE_SUFFIX}`,
      [
        {
          manifest,
          snapshot: args.snapshot,
          options: args.options,
          concurrencyLimits: args.concurrencyLimits,
          archiveLimits: args.archiveLimits,
        },
      ],
      this._config
    );
    const session = this.wrapSession(result);
    if (!manifest || manifest.root === '/workspace') workspaceRoots()[this._name] = session.state.manifest.root;
    return session;
  }

  /**
   * Gives the trusted manifest the root the Worker's backend uses for `/workspace`, so resuming works on
   * backends that move it, such as Daytona and Vercel.
   */
  resolveTrustedManifestForResume(manifest: Manifest): Manifest {
    const root = getCurrentPluginConfig()?.workspaceRoots?.[this._name];
    if (manifest.root !== '/workspace' || root === undefined) return manifest;
    return Object.assign(cloneManifest(manifest), { root });
  }

  /** Before a resume, asks the Worker where its backend puts `/workspace`, unless this Workflow run already knows it. */
  async _fetchWorkspaceRoot(clientOptions?: SandboxClientOptions): Promise<void> {
    const name = `${this._name}${SANDBOX_CLIENT_WORKSPACE_ROOT_SUFFIX}`;
    workspaceRoots()[this._name] ??= await scheduleActivity<string>(name, [clientOptions], this._config);
  }

  async resume(
    state: TemporalSandboxSessionState,
    options?: SandboxClientResumeOptions
  ): Promise<SandboxSessionLike<TemporalSandboxSessionState>> {
    const envelope = serializeWorkflowSessionState(state);
    const { trustedManifest, archiveLimits, clientOptions } = options ?? {};
    const result = await maybeTemporalSpan(
      sandboxSpanName(SANDBOX_CLIENT_RESUME_SUFFIX),
      () =>
        scheduleActivity<SandboxSessionResult>(
          `${this._name}${SANDBOX_CLIENT_RESUME_SUFFIX}`,
          [
            {
              state: envelope,
              trustedManifest: trustedManifest && encodeManifest(trustedManifest),
              archiveLimits,
              clientOptions,
            },
          ],
          sessionActivityOptions(this._config, state)
        ),
      { sessionId: state.sessionId }
    );
    return this.wrapSession(result);
  }

  async delete(state: TemporalSandboxSessionState): Promise<void> {
    await maybeTemporalSpan(
      sandboxSpanName(SANDBOX_CLIENT_DELETE_SUFFIX),
      () =>
        scheduleActivity<void>(
          `${this._name}${SANDBOX_CLIENT_DELETE_SUFFIX}`,
          [{ state: serializeWorkflowSessionState(state) }],
          sessionActivityOptions(this._config, state)
        ),
      { sessionId: state.sessionId }
    );
  }

  async serializeSessionState(
    state: TemporalSandboxSessionState,
    options?: SandboxSessionSerializationOptions
  ): Promise<Record<string, unknown>> {
    const refreshed = await maybeTemporalSpan(
      sandboxSpanName(SANDBOX_CLIENT_SERIALIZE_SESSION_STATE_SUFFIX),
      () =>
        scheduleActivity<SerializedSandboxSessionState>(
          `${this._name}${SANDBOX_CLIENT_SERIALIZE_SESSION_STATE_SUFFIX}`,
          [{ state: serializeWorkflowSessionState(state), options }],
          sessionActivityOptions(this._config, state)
        ),
      { sessionId: state.sessionId }
    );
    const revived = reviveWorkflowSessionState(refreshed);
    if ('snapshot' in revived) state.snapshot = revived.snapshot;
    else delete state.snapshot;
    if ('snapshotFingerprint' in revived) state.snapshotFingerprint = revived.snapshotFingerprint;
    else delete state.snapshotFingerprint;
    if ('snapshotFingerprintVersion' in revived) state.snapshotFingerprintVersion = revived.snapshotFingerprintVersion;
    else delete state.snapshotFingerprintVersion;
    if ('exposedPorts' in revived) state.exposedPorts = revived.exposedPorts;
    else delete state.exposedPorts;
    state.sessionId = revived.sessionId;
    state.manifest = revived.manifest;
    if ('workspaceReady' in revived) state.workspaceReady = revived.workspaceReady;
    else delete state.workspaceReady;
    state.providerState = revived.providerState;
    if (revived.taskQueue !== undefined) state.taskQueue = revived.taskQueue;
    else delete state.taskQueue;
    return {
      sessionId: refreshed.sessionId,
      providerState: refreshed.providerState,
      ...(refreshed.taskQueue !== undefined ? { taskQueue: refreshed.taskQueue } : {}),
    };
  }

  async deserializeSessionState(state: Record<string, unknown>): Promise<TemporalSandboxSessionState> {
    const { sessionId, providerState, manifest, taskQueue, ...rest } = state;
    if (typeof sessionId !== 'string') {
      throw ApplicationFailure.create({
        message:
          'Serialized sandbox session state is missing a sessionId — it was not produced by a Temporal sandbox client.',
        type: 'SandboxSessionStateInvalid',
        nonRetryable: true,
      });
    }
    if (manifest == null) {
      throw ApplicationFailure.create({
        message:
          'Serialized sandbox session state is missing a manifest — it was not produced by a Temporal sandbox client.',
        type: 'SandboxSessionStateInvalid',
        nonRetryable: true,
      });
    }
    if (taskQueue !== undefined && typeof taskQueue !== 'string') {
      throw ApplicationFailure.create({
        message: 'Serialized sandbox session state has a taskQueue that is not a string.',
        type: 'SandboxSessionStateInvalid',
        nonRetryable: true,
      });
    }
    return {
      ...rest,
      sessionId,
      providerState: (providerState ?? {}) as Record<string, unknown>,
      manifest: decodeManifest(manifest as EncodedManifest),
      ...(taskQueue !== undefined ? { taskQueue } : {}),
    };
  }

  /**
   * Asks a Worker whether upstream may reuse a session a paused run kept live. The Worker answers with its
   * backend's own check on the session it holds, or, for a hosted backend without one, whether a mount's
   * credentials changed, and changes nothing on the session.
   */
  async canReusePreservedOwnedSession(
    state: TemporalSandboxSessionState,
    options?: SandboxPreservedSessionReuseOptions
  ): Promise<boolean> {
    const { trustedManifest, clientOptions, revalidateManifestEntries } = options ?? {};
    return maybeTemporalSpan(
      sandboxSpanName(SANDBOX_CLIENT_CAN_REUSE_PRESERVED_OWNED_SESSION_SUFFIX),
      () =>
        scheduleActivity<boolean>(
          `${this._name}${SANDBOX_CLIENT_CAN_REUSE_PRESERVED_OWNED_SESSION_SUFFIX}`,
          [
            {
              state: serializeWorkflowSessionState(state),
              trustedManifest: trustedManifest && encodeManifest(trustedManifest),
              clientOptions,
              revalidateManifestEntries,
            },
          ],
          sessionActivityOptions(this._config, state)
        ),
      { sessionId: state.sessionId }
    );
  }

  private wrapSession(result: SandboxSessionResult): TemporalSandboxSession {
    return new TemporalSandboxSession(
      this._name,
      this._config,
      reviveWorkflowSessionState(result.state),
      result.supportsPty
    );
  }
}

/**
 * Creates a Workflow-side sandbox client for `RunConfig.sandbox`. All sandbox
 * operations are dispatched as Activities to the `SandboxClientProvider` or
 * `OnHostSandboxClientProvider` registered under the same name on the Worker.
 *
 * When `config` is omitted, Activities use a five-minute start-to-close timeout.
 * A supplied `ActivityOptions` object is used as-is without merging defaults,
 * except that an operation on an `OnHostSandboxClientProvider` session goes to
 * the Task Queue of the Worker process that created it.
 *
 * @param name - Provider name; must match the provider registered on the Worker side.
 * @param config - Activity options for every sandbox operation.
 */
export function temporalSandboxClient(name: string, config?: ActivityOptions): TemporalSandboxClient {
  return new TemporalSandboxClient(name, config);
}

/**
 * `sandbox.client` must be a `TemporalSandboxClient`, and a provided `sandbox.session` a
 * `TemporalSandboxSession`, so every sandbox operation is dispatched as an Activity rather than run
 * inline in the Workflow. `where` names the config in the error.
 */
export function validateTemporalSandboxClient(sandbox: SandboxRunConfig, where: string): void {
  if (sandbox.client == null) {
    throw ApplicationFailure.create({
      message:
        `${where}.client must be set to a Temporal sandbox client. ` +
        'Use temporalSandboxClient(name) from @temporalio/openai-agents/workflow.',
      type: 'SandboxConfigurationError',
      nonRetryable: true,
    });
  }
  if (!(sandbox.client instanceof TemporalSandboxClient)) {
    throw ApplicationFailure.create({
      message:
        `${where}.client must be created via temporalSandboxClient(name) ` +
        'from @temporalio/openai-agents/workflow. Do not pass a raw sandbox client directly.',
      type: 'SandboxConfigurationError',
      nonRetryable: true,
    });
  }
  // Upstream runs a provided live session's operations directly, like a raw client's.
  if (sandbox.session != null && !(sandbox.session instanceof TemporalSandboxSession)) {
    throw ApplicationFailure.create({
      message:
        `${where}.session must be a session from a Temporal sandbox client. ` +
        'Do not pass a raw sandbox session directly.',
      type: 'SandboxConfigurationError',
      nonRetryable: true,
    });
  }
}
