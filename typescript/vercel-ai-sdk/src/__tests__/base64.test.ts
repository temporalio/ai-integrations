import test from 'ava';
import { decodeBase64, encodeBase64 } from '../base64';

test('sandbox base64 matches native APIs for every byte and padding length', (t) => {
  const bytes = Array.from({ length: 256 }, (_, index) => String.fromCharCode(index)).join('');
  for (let length = 0; length <= bytes.length; length++) {
    const input = bytes.slice(0, length);
    t.is(encodeBase64(input), btoa(input));
    t.is(decodeBase64(btoa(input)), input);
  }
});

test('sandbox base64 accepts whitespace and omitted padding', (t) => {
  for (const input of ['', 'YQ', 'YQ==', 'Y Q\t=\n=', 'YWI', 'YWJj', 'AB', 'ABC']) {
    t.is(decodeBase64(input), atob(input));
  }
});

test('sandbox base64 rejects invalid input with the Web error name', (t) => {
  for (const input of ['A', '=', 'YQ=', 'YQ===', 'YQ==A', '____', 'é']) {
    t.throws(() => decodeBase64(input), { name: 'InvalidCharacterError' });
  }
  t.throws(() => encodeBase64('☃'), { name: 'InvalidCharacterError' });
});
