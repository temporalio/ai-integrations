// Pin the dev copy of each peer dependency to the lowest version the plugin allows: the floor
// of the peer range, or of the dev range when that is higher. Used by `make sync-lowest`.
import { readFileSync, writeFileSync } from 'node:fs';

const compare = (a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
// The lowest x.y.z a range names, so `^1.25.1 || ^2.0.0` and `>=2.0.0 <2.4.0` give their lower bound.
const floor = (range) => {
  const versions = [...range.matchAll(/(\d+)\.(\d+)\.(\d+)/g)].map((match) => match.slice(1).map(Number));
  if (versions.length === 0) throw new Error(`No x.y.z version in range ${JSON.stringify(range)}`);
  return versions.reduce((a, b) => (compare(a, b) <= 0 ? a : b));
};

const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
pkg.devDependencies ??= {};
for (const [name, range] of Object.entries(pkg.peerDependencies ?? {})) {
  const dev = pkg.devDependencies[name];
  const lowest = [floor(range), ...(dev ? [floor(dev)] : [])].reduce((a, b) => (compare(a, b) >= 0 ? a : b));
  pkg.devDependencies[name] = lowest.join('.');
  console.log(`${name}: ${pkg.devDependencies[name]}`);
}
writeFileSync('package.json', `${JSON.stringify(pkg, null, 2)}\n`);
