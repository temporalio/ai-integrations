import { createHash } from 'node:crypto';
import test from 'ava';
import { sha256Hex } from '../sha256';

// FIPS 180-2 examples, plus inputs that cross the one- and two-block padding boundaries.
for (const [name, input, digest] of [
  ['empty', '', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'],
  ['abc', 'abc', 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'],
  [
    '448-bit message',
    'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq',
    '248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1',
  ],
] as const) {
  test(`sha256Hex matches the FIPS 180-2 digest: ${name}`, (t) => {
    t.is(sha256Hex(input), digest);
  });
}

test('sha256Hex matches node:crypto across lengths and non-ASCII text', (t) => {
  const inputs = ['é', '日本語', '🦜🔗', 'a'.repeat(1_000_000)];
  for (let length = 0; length <= 130; length++) inputs.push('x'.repeat(length));
  for (const input of inputs) {
    t.is(sha256Hex(input), createHash('sha256').update(input, 'utf8').digest('hex'), `length ${input.length}`);
  }
});
