/** Explicit lock updates; ordinary Bazel builds consume committed pnpm locks. */
import { argumentsOf, command, root } from './common';
import { dirname, join, resolve } from 'node:path';
export function main():void {
 const {positionals:[action],options:o}=argumentsOf();
 const project=resolve(root(),o['plugin-dir']);
 if(action==='update') command(process.env.JS_BINARY__NODE_BINARY || process.execPath,[join(dirname(require.resolve('pnpm')), 'bin/pnpm.cjs'),'update','--latest=false'],project);
 else throw new Error('expected update');
}
if(require.main===module)try{main();}catch(e:any){console.error(e.message);process.exitCode=1;}
