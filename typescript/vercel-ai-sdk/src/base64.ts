import { fromByteArray, toByteArray } from 'base64-js';

function invalidCharacter(): Error {
  return Object.assign(new Error('Invalid character'), { name: 'InvalidCharacterError' });
}

/** Encode a Latin-1 string without depending on Node APIs in the sandbox. */
export function encodeBase64(value: string): string {
  const input = String(value);
  const bytes = new Uint8Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const code = input.charCodeAt(i);
    if (code > 255) throw invalidCharacter();
    bytes[i] = code;
  }
  return fromByteArray(bytes);
}

/** Decode using the forgiving base64 rules of the Web atob API. */
export function decodeBase64(value: string): string {
  let input = String(value).replace(/[\t\n\f\r ]/g, '');
  if (input.length % 4 === 0) input = input.replace(/={1,2}$/, '');
  if (input.length % 4 === 1 || /[^A-Za-z0-9+/]/.test(input)) throw invalidCharacter();
  const padded = input + '='.repeat((4 - (input.length % 4)) % 4);
  return Array.from(toByteArray(padded), (byte) => String.fromCharCode(byte)).join('');
}
