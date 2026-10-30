import { EnvValueReference, registerEnvValueReference } from '@openai/agents-core/sandbox';

/** A Manifest environment reference that keeps its value in a private field, which only `serialize` exposes. */
export class PrivateEnvValue extends EnvValueReference {
  static readonly type = 'test.private-env-value';
  readonly #value: string;

  constructor(value: string) {
    super();
    this.#value = value;
  }

  serialize(): Record<string, unknown> {
    return { secret: this.#value };
  }

  async resolve(): Promise<string> {
    return this.#value;
  }
}

registerEnvValueReference(PrivateEnvValue, (payload) => new PrivateEnvValue(String(payload.secret)));
