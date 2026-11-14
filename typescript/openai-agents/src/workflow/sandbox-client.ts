import type {
  ManifestInput,
  SandboxClient,
  SandboxClientCreateArgs,
  SandboxClientOptions,
  SandboxClientResumeOptions,
  SandboxPreservedSessionReuseOptions,
  SandboxRunConfig,
  SandboxSession,
  SandboxSessionLike,
  SandboxSessionSerializationOptions,
} from '@openai/agents-core/sandbox';
import { Manifest } from '@openai/agents-core/sandbox';
import { ApplicationFailure, type ActivityOptions } from '@temporalio/common';
import { isCancellation, scheduleActivity } from '@temporalio/workflow';
import {
  SANDBOX_CLIENT_CREATE_SUFFIX,
  SANDBOX_CLIENT_DELETE_SUFFIX,
  SANDBOX_CLIENT_RESUME_SUFFIX,
  SANDBOX_CLIENT_SERIALIZE_SESSION_STATE_SUFFIX,
  decodeManifest,
  encodeManifest,
  reviveWorkflowSessionState,
  sandboxSpanName,
  serializeSessionEnvelope,
  type EncodedManifest,
  type SandboxSessionResult,
  type SerializedSandboxSessionState,
  type TemporalSandboxSessionState,
} from '../common/sandbox-activity-types';
import { TemporalSandboxSession } from './sandbox-session';
import { maybeTemporalSpan } from './span-helpers';

/** A session a `TemporalSandboxClient` created or resumed, with the mounts that decide how upstream resumes it. */
export interface LiveSandboxSession {
  session: TemporalSandboxSession;
  inContainerMounts: boolean;
  nonResumableMountAuthority: boolean;
}

/** The sessions this Workflow's `TemporalSandboxClient`s created or resumed, by provider name and session ID. */
const liveSessions = new Map<string, LiveSandboxSession>();

/** The environment, path grants and mount command allowlist of `manifest`, which a live sandbox keeps. */
function manifestAuthority(manifest: ManifestInput | undefined): string | undefined {
  if (manifest === undefined) return undefined;
  const { environment, extraPathGrants, remoteMountCommandAllowlist } = encodeManifest(
    manifest instanceof Manifest ? manifest : new Manifest(manifest)
  );
  return JSON.stringify([
    Object.entries(environment).sort(([a], [b]) => (a < b ? -1 : 1)),
    extraPathGrants,
    remoteMountCommandAllowlist,
  ]);
}

function throwIfCancelled(error: unknown): void {
  if (isCancellation(error)) throw error;
}

/**
 * Workflow-side sandbox client. Holds no connection to any sandbox backend —
 * session creation, resumption, and every session operation are dispatched as
 * Activities to the `SandboxClientProvider` registered under the same name on
 * the Worker.
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
    this.backendId = name;
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
    return this.wrapSession(result);
  }

  async resume(
    state: TemporalSandboxSessionState,
    options?: SandboxClientResumeOptions
  ): Promise<SandboxSessionLike<TemporalSandboxSessionState>> {
    const envelope = serializeSessionEnvelope(state.sessionId, state, state.providerState);
    // Upstream passes the agent's current manifest as authoritative. Its environment, path grants and
    // mount command allowlist replace the persisted ones, so a stale or altered RunState can't pick what
    // the Worker resolves.
    if (options?.trustedManifest !== undefined) {
      const trusted = encodeManifest(options.trustedManifest);
      envelope.manifest = {
        ...envelope.manifest,
        environment: trusted.environment,
        extraPathGrants: trusted.extraPathGrants,
        remoteMountCommandAllowlist: trusted.remoteMountCommandAllowlist,
      };
    }
    const result = await maybeTemporalSpan(
      sandboxSpanName(SANDBOX_CLIENT_RESUME_SUFFIX),
      () =>
        scheduleActivity<SandboxSessionResult>(
          `${this._name}${SANDBOX_CLIENT_RESUME_SUFFIX}`,
          [{ state: envelope, archiveLimits: options?.archiveLimits }],
          this._config
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
          [{ state: serializeSessionEnvelope(state.sessionId, state, state.providerState) }],
          this._config
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
          [{ state: serializeSessionEnvelope(state.sessionId, state, state.providerState), options }],
          this._config
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
    return { sessionId: refreshed.sessionId, providerState: refreshed.providerState };
  }

  async deserializeSessionState(state: Record<string, unknown>): Promise<TemporalSandboxSessionState> {
    const { sessionId, providerState, manifest, ...rest } = state;
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
    return {
      ...rest,
      sessionId,
      providerState: (providerState ?? {}) as Record<string, unknown>,
      manifest: decodeManifest(manifest as EncodedManifest),
    };
  }

  /**
   * Whether upstream may reuse a session a paused run kept live: only while `trustedManifest` gives it the
   * environment, extra path grants and remote mount command allowlist it was created with, which a live
   * session keeps.
   */
  canReusePreservedOwnedSession(
    state: TemporalSandboxSessionState,
    options?: SandboxPreservedSessionReuseOptions
  ): boolean {
    return (
      options?.trustedManifest === undefined ||
      manifestAuthority(state.manifest) === manifestAuthority(options.trustedManifest)
    );
  }

  /** @internal */
  _liveSession(sessionId: string): LiveSandboxSession | undefined {
    return liveSessions.get(JSON.stringify([this._name, sessionId]));
  }

  /**
   * @internal Closes `session` as upstream's cleanup does. If its Worker no longer holds it, deletes it
   * through this client instead; if that fails too, the session is left.
   */
  async _close(session: TemporalSandboxSession): Promise<void> {
    const options = { reason: 'cleanup' };
    for (const step of [(session as SandboxSession).preStop, session.stop, session.shutdown]) {
      await step?.call(session, options).catch(throwIfCancelled);
    }
    await session.delete(options).catch(async (error: unknown) => {
      throwIfCancelled(error);
      await this.delete(session.state).catch(throwIfCancelled);
    });
  }

  private wrapSession(result: SandboxSessionResult): TemporalSandboxSession {
    const session = new TemporalSandboxSession(
      this._name,
      this._config,
      reviveWorkflowSessionState(result.state),
      result.supportsPty
    );
    liveSessions.set(JSON.stringify([this._name, session.state.sessionId]), {
      session,
      inContainerMounts: result.inContainerMounts === true,
      nonResumableMountAuthority: result.nonResumableMountAuthority === true,
    });
    return session;
  }
}

/**
 * Creates a Workflow-side sandbox client for `RunConfig.sandbox`. All sandbox
 * operations are dispatched as Activities to the `SandboxClientProvider`
 * registered under the same name on the Worker.
 *
 * When `config` is omitted, Activities use a five-minute start-to-close timeout.
 * A supplied `ActivityOptions` object is used as-is without merging defaults.
 *
 * @param name - Provider name; must match the `SandboxClientProvider` registered on the Worker side.
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
