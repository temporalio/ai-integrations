import { AsyncLocalStorage } from 'node:async_hooks';
import { randomUUID } from 'node:crypto';
import {
  UserError,
  getCurrentTrace,
  withCustomSpan,
  type ApplyPatchOperation,
  type ApplyPatchResult,
  type Editor,
} from '@openai/agents-core';
import type {
  SandboxClient,
  SandboxClientOptions,
  SandboxClientResumeOptions,
  SandboxSession,
  SandboxSessionState,
} from '@openai/agents-core/sandbox';
import { Manifest, SandboxError, SandboxMountError, isEnvValueReference } from '@openai/agents-core/sandbox';
import {
  captureLiveMountCredentialAuthorityIfAbsent,
  deserializeHostPathGrantRedactionMetadata,
  isStringRecord,
  liveMountEnvironmentAuthorityMatches,
  rebindPersistedMountCredentials,
  rehydratePersistedEnvironmentForRuntime,
  sanitizeEnvironmentForPersistence,
} from '@openai/agents-core/sandbox/internal';
import { ApplicationFailure } from '@temporalio/common';
import { registerWorkerEnvVarContext } from '../common/worker-env-vars';
import {
  SANDBOX_CLIENT_CAN_REUSE_PRESERVED_OWNED_SESSION_SUFFIX,
  SANDBOX_CLIENT_CREATE_SUFFIX,
  SANDBOX_CLIENT_DELETE_SUFFIX,
  SANDBOX_CLIENT_RESUME_SUFFIX,
  SANDBOX_CLIENT_SERIALIZE_SESSION_STATE_SUFFIX,
  SANDBOX_CLIENT_WORKSPACE_ROOT_SUFFIX,
  SANDBOX_EDITOR_CREATE_FILE_SUFFIX,
  SANDBOX_EDITOR_DELETE_FILE_SUFFIX,
  SANDBOX_EDITOR_UPDATE_FILE_SUFFIX,
  SANDBOX_SESSION_APPLY_MANIFEST_SUFFIX,
  SANDBOX_SESSION_DELETE_SUFFIX,
  SANDBOX_SESSION_EXEC_COMMAND_SUFFIX,
  SANDBOX_SESSION_EXEC_SUFFIX,
  SANDBOX_SESSION_HYDRATE_WORKSPACE_SUFFIX,
  SANDBOX_SESSION_LIST_DIR_SUFFIX,
  SANDBOX_SESSION_MATERIALIZE_ENTRY_SUFFIX,
  SANDBOX_SESSION_PATH_EXISTS_SUFFIX,
  SANDBOX_SESSION_PERSIST_WORKSPACE_SUFFIX,
  SANDBOX_SESSION_READ_FILE_SUFFIX,
  SANDBOX_SESSION_RESOLVE_EXPOSED_PORT_SUFFIX,
  SANDBOX_SESSION_RUNNING_SUFFIX,
  SANDBOX_SESSION_SHUTDOWN_SUFFIX,
  SANDBOX_SESSION_START_SUFFIX,
  SANDBOX_SESSION_STOP_SUFFIX,
  SANDBOX_SESSION_VIEW_IMAGE_SUFFIX,
  SANDBOX_SESSION_WRITE_STDIN_SUFFIX,
  decodeEntry,
  decodeManifest,
  encodeManifest,
  encodeToolOutputImage,
  sandboxSpanName,
  serializeSessionEnvelope,
  toSdkStateRecord,
  withTrustedManifest,
  type EncodedManifest,
  type SandboxApplyManifestInput,
  type SandboxCanReusePreservedOwnedSessionInput,
  type SandboxCreateSessionInput,
  type SandboxEditorInput,
  type SandboxExecInput,
  type SandboxHydrateWorkspaceInput,
  type SandboxLifecycleInput,
  type SandboxListDirInput,
  type SandboxMaterializeEntryInput,
  type SandboxPathExistsInput,
  type SandboxPersistWorkspaceInput,
  type SandboxReadFileInput,
  type SandboxResolveExposedPortInput,
  type SandboxResumeSessionInput,
  type SandboxSerializeSessionStateInput,
  type SandboxSessionResult,
  type SandboxSessionStateInput,
  type SandboxViewImageInput,
  type SandboxWriteStdinInput,
  type SerializedSandboxSessionState,
} from '../common/sandbox-activity-types';

type ActivityFunction = (...args: any[]) => Promise<any>;

const workerEnvVarStorage = new AsyncLocalStorage<readonly string[]>();
registerWorkerEnvVarContext(() => workerEnvVarStorage.getStore());

/**
 * Temporal retries every Activity exception by default, so a SandboxError the agents library has
 * classified as terminal (`retryable === false`) is turned into a non-retryable ApplicationFailure. So
 * is any other UserError, such as a backend's refusal to resume, which a retry won't change.
 */
function translateSandboxErrors(fn: ActivityFunction): ActivityFunction {
  return async (...args: unknown[]) => {
    try {
      return await fn(...args);
    } catch (err: unknown) {
      if (err instanceof UserError && (!(err instanceof SandboxError) || err.retryable === false)) {
        throw ApplicationFailure.create({
          message: err.message,
          type: err instanceof SandboxError ? err.code : err.name,
          nonRetryable: true,
          cause: err,
        });
      }
      throw err;
    }
  };
}

function sessionIdOf(input: unknown): string | undefined {
  const sessionId = (input as { state?: { sessionId?: unknown } } | undefined)?.state?.sessionId;
  return typeof sessionId === 'string' ? sessionId : undefined;
}

function summarizeResult(result: unknown): Record<string, number> {
  if (result && typeof result === 'object' && typeof (result as { exitCode?: unknown }).exitCode === 'number') {
    return { exitCode: (result as { exitCode: number }).exitCode };
  }
  if (result instanceof Uint8Array) return { byteLength: result.byteLength };
  if (typeof result === 'string') return { length: result.length };
  if (Array.isArray(result)) return { count: result.length };
  return {};
}

function unsupportedOperation(name: string, operation: string): never {
  throw ApplicationFailure.create({
    message: `Sandbox backend '${name}' does not support ${operation}().`,
    type: 'SandboxOperationUnsupported',
    nonRetryable: true,
  });
}

/**
 * A named sandbox client provider for Temporal Workflows.
 *
 * Wraps a real `SandboxClient` with a unique name so multiple sandbox
 * backends can be registered on a single Temporal Worker. Each provider gets
 * its own set of Activities whose names are prefixed with the provider name.
 * Live sessions are cached by session id. Docker and Unix-local run the sandbox
 * on the Worker's own machine, so they're for local development with a single
 * Worker. With a hosted backend, any Worker can serve a session's Activities
 * because a cache miss is self-healed via `client.resume()`, except for a
 * session whose manifest has an in-container mount, or a mount upstream can't
 * verify after persistence, such as one with `dockerVolumeMountStrategy()` and
 * `driverOptions`: upstream refuses to resume it, so its Activities fail on a
 * cache miss.
 *
 * On the Worker side, pass providers to the plugin:
 *
 * ```ts
 * const plugin = new OpenAIAgentsPlugin({
 *   modelProvider,
 *   sandboxClientProviders: [new SandboxClientProvider('local', new UnixLocalSandboxClient())],
 * });
 * ```
 *
 * On the Workflow side, reference a provider by name via
 * `temporalSandboxClient` from `@temporalio/openai-agents/workflow`:
 *
 * ```ts
 * runConfig: { sandbox: { client: temporalSandboxClient('local') } }
 * ```
 *
 * @experimental Sandbox support is experimental and may change without notice.
 */
export class SandboxClientProvider {
  private readonly _sessions = new Map<string, SandboxSession>();
  private readonly _resuming = new Map<string, Promise<SandboxSession>>();
  private _resolvableWorkerEnvVars: readonly string[] = [];
  private _workerEnvVarsConfigured = false;
  /** @internal */
  _addTemporalSpans = false;

  constructor(
    public readonly name: string,
    private readonly client: SandboxClient<SandboxClientOptions, SandboxSessionState>
  ) {}

  _setResolvableWorkerEnvVars(names: readonly string[]): void {
    if (
      this._workerEnvVarsConfigured &&
      (this._resolvableWorkerEnvVars.length !== names.length ||
        this._resolvableWorkerEnvVars.some((name, index) => name !== names[index]))
    ) {
      throw new Error(
        `Sandbox client provider '${this.name}' cannot be shared by plugins with different Worker environment allowlists.`
      );
    }
    this._resolvableWorkerEnvVars = [...names];
    this._workerEnvVarsConfigured = true;
  }

  private async rehydrateState(payload: SerializedSandboxSessionState): Promise<SandboxSessionState> {
    const record = toSdkStateRecord(payload);
    if (this.client.deserializeSessionState) {
      return await this.client.deserializeSessionState(record);
    }
    // Mirrors `deserializeLocalSandboxSessionStateValues`: Worker-injected keys are not
    // in the manifest and cannot be re-derived, so they sit under the rehydrated set.
    const manifest = decodeManifest(payload.manifest, this._resolvableWorkerEnvVars);
    const persisted = isStringRecord(record.environment) ? { ...record.environment } : {};
    const injected = Object.fromEntries(
      Object.entries(persisted).filter(([key]) => !Object.hasOwn(manifest.environment, key))
    );
    return {
      ...record,
      manifest,
      environment: { ...injected, ...(await rehydratePersistedEnvironmentForRuntime(manifest, persisted)) },
    } as SandboxSessionState;
  }

  /**
   * Backends refuse a resumed state whose mounts or path grant `hostPath`s weren't rebound from a trusted
   * manifest: the one the Workflow sends, or else the envelope's own.
   */
  private async resumableState(
    payload: SerializedSandboxSessionState,
    trustedManifest?: EncodedManifest
  ): Promise<SandboxSessionState> {
    const trusted = decodeManifest(trustedManifest ?? payload.manifest, this._resolvableWorkerEnvVars);
    const saved = trustedManifest ? withTrustedManifest(payload, trustedManifest) : payload;
    // Rebound onto the backend's own object, as upstream's `resumeSandboxSession` does: Cloudflare refuses
    // to resume one its `deserializeSessionState` returned, and a copy would get past that.
    const state = await this.rehydrateState(saved);
    const rebound = rebindPersistedMountCredentials(state, trusted);
    state.manifest = rebound.manifest;
    for (const key of Object.keys(state)) if (!(key in rebound)) delete state[key];
    const grants = new Map(trusted.extraPathGrants.map((grant) => [grant.path, grant]));
    const { extraPathGrants } = state.manifest;
    for (const [index, grant] of extraPathGrants.entries()) extraPathGrants[index] = grants.get(grant.path) ?? grant;
    for (const [key, paths] of Object.entries(deserializeHostPathGrantRedactionMetadata(state))) {
      if ((paths as string[]).every((path) => grants.get(path)?.hostPath !== undefined)) delete state[key];
    }
    return state;
  }

  private async session(
    payload: SerializedSandboxSessionState,
    trustedManifest?: EncodedManifest,
    options?: SandboxClientResumeOptions
  ): Promise<SandboxSession> {
    const existing = this._sessions.get(payload.sessionId);
    if (existing) return existing;
    // The agents run loop executes tools in parallel, so several Activities can
    // cache-miss the same session at once; share one resume and clear the entry
    // in `finally` so a failed resume isn't memoized and a later call can retry.
    const inFlight = this._resuming.get(payload.sessionId);
    if (inFlight) return inFlight;
    if (!this.client.resume) {
      throw ApplicationFailure.create({
        message: `Sandbox backend '${this.name}' does not support resume(); session '${payload.sessionId}' cannot be recovered on this Worker.`,
        type: 'SandboxOperationUnsupported',
        nonRetryable: true,
      });
    }
    const resume = this.client.resume.bind(this.client);
    const resumePromise = (async (): Promise<SandboxSession> => {
      return this.track(payload.sessionId, await resume(await this.resumableState(payload, trustedManifest), options));
    })();
    this._resuming.set(payload.sessionId, resumePromise);
    try {
      return await resumePromise;
    } finally {
      this._resuming.delete(payload.sessionId);
    }
  }

  /**
   * Caches a session the Worker created or resumed, and records its live mount authority, as upstream's
   * runner does for a session it owns, so the reuse check compares the trusted manifest with it.
   */
  private track(sessionId: string, session: SandboxSession): SandboxSession {
    captureLiveMountCredentialAuthorityIfAbsent(session.state.manifest, session.state.environment);
    this._sessions.set(sessionId, session);
    return session;
  }

  private async providerState(state: SandboxSessionState): Promise<Record<string, unknown>> {
    if (this.client.serializeSessionState) {
      return await this.client.serializeSessionState(state);
    }
    // The envelope carries these fields itself; what is left is what the backend added.
    const {
      manifest: _manifest,
      snapshot: _snapshot,
      snapshotFingerprint: _snapshotFingerprint,
      snapshotFingerprintVersion: _snapshotFingerprintVersion,
      workspaceReady: _workspaceReady,
      exposedPorts: _exposedPorts,
      environment,
      ...rest
    } = state;
    return environment === undefined ? rest : { ...rest, environment: sanitizeEnvironmentForPersistence(state) };
  }

  private async sessionResult(sessionId: string, session: SandboxSession): Promise<SandboxSessionResult> {
    return {
      state: serializeSessionEnvelope(sessionId, session.state, await this.providerState(session.state)),
      supportsPty: session.supportsPty?.() ?? false,
    };
  }

  private async editor(input: SandboxEditorInput): Promise<Editor> {
    const session = await this.session(input.state);
    if (!session.createEditor) unsupportedOperation(this.name, 'createEditor');
    return session.createEditor(input.runAs);
  }

  _getActivities(): Record<string, ActivityFunction> {
    const n = this.name;

    const activities: Record<string, ActivityFunction> = {
      [`${n}${SANDBOX_CLIENT_CREATE_SUFFIX}`]: async (
        input: SandboxCreateSessionInput
      ): Promise<SandboxSessionResult> => {
        if (!this.client.create) unsupportedOperation(n, 'create');
        const session = await this.client.create({
          manifest: input.manifest && decodeManifest(input.manifest, this._resolvableWorkerEnvVars),
          snapshot: input.snapshot,
          options: input.options,
          concurrencyLimits: input.concurrencyLimits,
          archiveLimits: input.archiveLimits,
        });
        // At-least-once Activity execution: `SandboxClient.create` takes no
        // idempotency key, so a retry after the backend session is created but
        // before this result is recorded leaks that first session.
        const sessionId = randomUUID();
        return this.sessionResult(sessionId, this.track(sessionId, session));
      },

      [`${n}${SANDBOX_CLIENT_RESUME_SUFFIX}`]: async (
        input: SandboxResumeSessionInput
      ): Promise<SandboxSessionResult> => {
        if (!this.client.resume) unsupportedOperation(n, 'resume');
        const session = await this.client.resume(await this.resumableState(input.state, input.trustedManifest), {
          archiveLimits: input.archiveLimits,
          clientOptions: input.clientOptions,
        });
        return this.sessionResult(input.state.sessionId, this.track(input.state.sessionId, session));
      },

      [`${n}${SANDBOX_CLIENT_DELETE_SUFFIX}`]: async (input: SandboxSessionStateInput): Promise<void> => {
        const state = this._sessions.get(input.state.sessionId)?.state ?? (await this.rehydrateState(input.state));
        try {
          await this.client.delete?.(state);
        } finally {
          this._sessions.delete(input.state.sessionId);
        }
      },

      [`${n}${SANDBOX_CLIENT_SERIALIZE_SESSION_STATE_SUFFIX}`]: async (
        input: SandboxSerializeSessionStateInput
      ): Promise<SerializedSandboxSessionState> => {
        const session = await this.session(input.state);
        const providerState = this.client.serializeSessionState
          ? await this.client.serializeSessionState(session.state, input.options)
          : await this.providerState(session.state);
        return serializeSessionEnvelope(input.state.sessionId, session.state, providerState, false);
      },

      [`${n}${SANDBOX_CLIENT_CAN_REUSE_PRESERVED_OWNED_SESSION_SUFFIX}`]: async (
        input: SandboxCanReusePreservedOwnedSessionInput
      ): Promise<boolean> => {
        let state: SandboxSessionState;
        try {
          ({ state } = await this.session(input.state, input.trustedManifest, { clientOptions: input.clientOptions }));
        } catch (err) {
          // On a cache miss, upstream refuses to resume some sessions, such as one with an in-container mount.
          if (err instanceof SandboxMountError && err.code === 'mount_config_invalid') return false;
          throw err;
        }
        const trustedManifest =
          input.trustedManifest && decodeManifest(input.trustedManifest, this._resolvableWorkerEnvVars);
        // A Docker container keeps the environment it was created with, so it would keep a removed variable.
        // A session rehydrated on a cache miss has the trusted environment, so the persisted one counts too.
        if (
          trustedManifest &&
          this.client.backendId === 'docker' &&
          [...Object.keys(state.manifest.environment), ...Object.keys(input.state.manifest.environment)].some(
            (key) => !(key in trustedManifest.environment)
          )
        ) {
          return false;
        }
        if (this.client.canReusePreservedOwnedSession) {
          const reusable = await this.client.canReusePreservedOwnedSession(state, {
            clientOptions: input.clientOptions,
            revalidateManifestEntries: input.revalidateManifestEntries,
            trustedManifest,
          });
          if (!reusable) return false;
        } else if (trustedManifest) {
          // For a backend with no check of its own, upstream refuses a change to the credentials that mounts
          // take from the environment.
          const environment = await trustedManifest.resolveEnvironment();
          if (!liveMountEnvironmentAuthorityMatches(state.manifest, trustedManifest, environment)) return false;
        }
        // Upstream refuses to resume a live session without a trusted manifest, so there is nothing to refresh.
        if (!trustedManifest) return true;
        const { extraPathGrants, remoteMountCommandAllowlist } = state.manifest;
        extraPathGrants.splice(0, extraPathGrants.length, ...trustedManifest.extraPathGrants);
        remoteMountCommandAllowlist.splice(
          0,
          remoteMountCommandAllowlist.length,
          ...trustedManifest.remoteMountCommandAllowlist
        );
        const environment = { ...state.environment, ...(await trustedManifest.resolveEnvironment()) };
        for (const [key, value] of Object.entries(state.manifest.environment)) {
          if (!(key in trustedManifest.environment) && isEnvValueReference(value)) {
            delete environment[key];
            delete state.manifest.environment[key];
          }
        }
        Object.assign(state.manifest.environment, trustedManifest.environment);
        state.environment = environment;
        return true;
      },

      [`${n}${SANDBOX_CLIENT_WORKSPACE_ROOT_SUFFIX}`]: async (clientOptions?: Record<string, unknown>) =>
        this.client.resolveTrustedManifestForResume?.(new Manifest(), clientOptions).root ?? '/workspace',

      [`${n}${SANDBOX_SESSION_START_SUFFIX}`]: async (input: SandboxLifecycleInput): Promise<void> => {
        const session = await this.session(input.state);
        await session.start?.(input.options);
      },

      [`${n}${SANDBOX_SESSION_RUNNING_SUFFIX}`]: async (input: SandboxSessionStateInput): Promise<boolean> => {
        const session = await this.session(input.state);
        return session.running ? await session.running() : false;
      },

      [`${n}${SANDBOX_SESSION_STOP_SUFFIX}`]: async (input: SandboxLifecycleInput): Promise<void> => {
        const session = await this.session(input.state);
        await session.stop?.(input.options);
      },

      [`${n}${SANDBOX_SESSION_SHUTDOWN_SUFFIX}`]: async (input: SandboxLifecycleInput): Promise<void> => {
        const session = await this.session(input.state);
        await session.shutdown?.(input.options);
      },

      [`${n}${SANDBOX_SESSION_DELETE_SUFFIX}`]: async (input: SandboxLifecycleInput): Promise<void> => {
        const session = await this.session(input.state);
        try {
          if (session.delete) {
            await session.delete(input.options);
          } else if (!session.stop && !session.shutdown) {
            await session.close?.();
          }
        } finally {
          this._sessions.delete(input.state.sessionId);
        }
      },

      [`${n}${SANDBOX_SESSION_EXEC_SUFFIX}`]: async (input: SandboxExecInput) => {
        const session = await this.session(input.state);
        if (!session.exec) unsupportedOperation(n, 'exec');
        return session.exec(input.args);
      },

      [`${n}${SANDBOX_SESSION_EXEC_COMMAND_SUFFIX}`]: async (input: SandboxExecInput): Promise<string> => {
        const session = await this.session(input.state);
        if (!session.execCommand) unsupportedOperation(n, 'execCommand');
        return session.execCommand(input.args);
      },

      [`${n}${SANDBOX_SESSION_WRITE_STDIN_SUFFIX}`]: async (input: SandboxWriteStdinInput): Promise<string> => {
        const session = await this.session(input.state);
        if (!session.writeStdin) unsupportedOperation(n, 'writeStdin');
        return session.writeStdin(input.args);
      },

      [`${n}${SANDBOX_SESSION_VIEW_IMAGE_SUFFIX}`]: async (input: SandboxViewImageInput) => {
        const session = await this.session(input.state);
        if (!session.viewImage) unsupportedOperation(n, 'viewImage');
        return encodeToolOutputImage(await session.viewImage(input.args));
      },

      [`${n}${SANDBOX_SESSION_READ_FILE_SUFFIX}`]: async (
        input: SandboxReadFileInput
      ): Promise<string | Uint8Array> => {
        const session = await this.session(input.state);
        if (!session.readFile) unsupportedOperation(n, 'readFile');
        return session.readFile(input.args);
      },

      [`${n}${SANDBOX_SESSION_LIST_DIR_SUFFIX}`]: async (input: SandboxListDirInput) => {
        const session = await this.session(input.state);
        if (!session.listDir) unsupportedOperation(n, 'listDir');
        return session.listDir(input.args);
      },

      [`${n}${SANDBOX_SESSION_PATH_EXISTS_SUFFIX}`]: async (input: SandboxPathExistsInput): Promise<boolean> => {
        const session = await this.session(input.state);
        if (!session.pathExists) unsupportedOperation(n, 'pathExists');
        return session.pathExists(input.path, input.runAs);
      },

      [`${n}${SANDBOX_SESSION_MATERIALIZE_ENTRY_SUFFIX}`]: async (
        input: SandboxMaterializeEntryInput
      ): Promise<EncodedManifest> => {
        const session = await this.session(input.state);
        if (!session.materializeEntry) unsupportedOperation(n, 'materializeEntry');
        await session.materializeEntry({ path: input.path, entry: decodeEntry(input.entry), runAs: input.runAs });
        return encodeManifest(session.state.manifest);
      },

      [`${n}${SANDBOX_SESSION_APPLY_MANIFEST_SUFFIX}`]: async (
        input: SandboxApplyManifestInput
      ): Promise<EncodedManifest> => {
        const session = await this.session(input.state);
        const manifest = decodeManifest(input.manifest, this._resolvableWorkerEnvVars);
        if (session.applyManifest) {
          await session.applyManifest(manifest, input.runAs);
        } else if (session.materializeEntry) {
          for (const [path, entry] of Object.entries(manifest.entries)) {
            await session.materializeEntry({ path, entry, runAs: input.runAs });
          }
        } else {
          unsupportedOperation(n, 'applyManifest');
        }
        return encodeManifest(session.state.manifest);
      },

      [`${n}${SANDBOX_SESSION_PERSIST_WORKSPACE_SUFFIX}`]: async (
        input: SandboxPersistWorkspaceInput
      ): Promise<Uint8Array> => {
        const session = await this.session(input.state);
        if (!session.persistWorkspace) unsupportedOperation(n, 'persistWorkspace');
        if (input.archiveLimits !== undefined) session.setArchiveLimits?.(input.archiveLimits);
        return session.persistWorkspace();
      },

      [`${n}${SANDBOX_SESSION_HYDRATE_WORKSPACE_SUFFIX}`]: async (
        input: SandboxHydrateWorkspaceInput,
        data: string | Uint8Array
      ): Promise<void> => {
        const session = await this.session(input.state);
        if (!session.hydrateWorkspace) unsupportedOperation(n, 'hydrateWorkspace');
        await session.hydrateWorkspace(
          data,
          input.archiveLimits !== undefined ? { archiveLimits: input.archiveLimits } : undefined
        );
      },

      [`${n}${SANDBOX_SESSION_RESOLVE_EXPOSED_PORT_SUFFIX}`]: async (input: SandboxResolveExposedPortInput) => {
        const session = await this.session(input.state);
        if (!session.resolveExposedPort) unsupportedOperation(n, 'resolveExposedPort');
        return session.resolveExposedPort(input.port);
      },

      [`${n}${SANDBOX_EDITOR_CREATE_FILE_SUFFIX}`]: async (
        input: SandboxEditorInput
      ): Promise<ApplyPatchResult | void> => {
        const editor = await this.editor(input);
        return editor.createFile(input.operation as Extract<ApplyPatchOperation, { type: 'create_file' }>);
      },

      [`${n}${SANDBOX_EDITOR_UPDATE_FILE_SUFFIX}`]: async (
        input: SandboxEditorInput
      ): Promise<ApplyPatchResult | void> => {
        const editor = await this.editor(input);
        return editor.updateFile(input.operation as Extract<ApplyPatchOperation, { type: 'update_file' }>);
      },

      [`${n}${SANDBOX_EDITOR_DELETE_FILE_SUFFIX}`]: async (
        input: SandboxEditorInput
      ): Promise<ApplyPatchResult | void> => {
        const editor = await this.editor(input);
        return editor.deleteFile(input.operation as Extract<ApplyPatchOperation, { type: 'delete_file' }>);
      },
    };

    // Upstream copies manifests, in create() and applyManifest() for example, and each copy rebuilds its
    // Worker environment references, which take the allowlist from this context.
    return Object.fromEntries(
      Object.entries(activities).map(([name, fn]) => [
        name,
        this.withSpan(
          name,
          translateSandboxErrors((...args) => workerEnvVarStorage.run(this._resolvableWorkerEnvVars, () => fn(...args)))
        ),
      ])
    );
  }

  private withSpan(name: string, fn: ActivityFunction): ActivityFunction {
    const spanName = `${sandboxSpanName(name.slice(this.name.length))}:result`;
    return async (...args: unknown[]) => {
      const sessionId = sessionIdOf(args[0]);
      if (!this._addTemporalSpans || !sessionId || !getCurrentTrace()) return fn(...args);
      return withCustomSpan(
        async (span) => {
          const result = await fn(...args);
          Object.assign(span.spanData.data, { sessionId, ...summarizeResult(result) });
          return result;
        },
        { data: { name: spanName, data: {} } }
      );
    };
  }
}
