/** Read npm archives without accepting links, traversal, or duplicate members. */
import { list } from 'tar';

export async function archiveFiles(file: string): Promise<Map<string, Buffer>> {
  const files = new Map<string, Buffer>(), seen = new Set<string>();
  await list({ file, onReadEntry(entry) {
    if (entry.path.startsWith('/') || entry.path.split('/').includes('..') || seen.has(entry.path)) {
      throw new Error(`unsafe or duplicate archive member: ${entry.path}`);
    }
    seen.add(entry.path);
    if (entry.type === 'Directory') { entry.resume(); return; }
    if (entry.type !== 'File') throw new Error(`unexpected archive member type: ${entry.type}`);
    const chunks: Buffer[] = [];
    entry.on('data', (chunk: Buffer) => chunks.push(chunk));
    entry.on('end', () => files.set(entry.path, Buffer.concat(chunks)));
  } });
  return files;
}
