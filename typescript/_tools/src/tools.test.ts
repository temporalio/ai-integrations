import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { mkdtempSync, writeFileSync, mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { create } from 'tar';
import { parseTag, checkPolicy, version, verifyArtifact, artifact } from './release';
import { targets } from './package';
import { selected } from './ci';

test('export maps are recursively checked', () => assert.deepEqual(targets({ '.': { require: './lib/index.js', types: './lib/index.d.ts' } }), ['./lib/index.js','./lib/index.d.ts']));
for(const invalid of ['v1.0.0','1.0','01.0.0','1.0.0+build','1.0.0-rc.01','1.0.0\n'])test(`reject noncanonical npm version ${JSON.stringify(invalid)}`,()=>assert.throws(()=>version(invalid)));
for(const [maturity,first] of Object.entries({'pre-release':'0.0.1','public-preview':'0.1.0','generally-available':'1.0.0'}))test(`${maturity} first and forward releases`,()=>{
 assert.equal(checkPolicy(first,maturity,[]),false);
 assert.equal(checkPolicy(first+'-rc.1',maturity,[]),false);
 assert.throws(()=>checkPolicy('2.0.0',maturity,[]));
 assert.equal(checkPolicy(first,maturity,[first]),true);
 assert.throws(()=>checkPolicy(first+'-rc.1',maturity,[first]));
});
for(const invalid of ['python/example/v1.2.3','typescript/../v1.0.0','typescript/example/v1.0.0\n'])test(`invalid scoped tag ${JSON.stringify(invalid)}`,()=>assert.throws(()=>parseTag(invalid)));
test('release identity and SHA512 must match tested bytes',()=>{
 const bytes=Buffer.from('tested tarball'),meta={name:'@temporalio/example',version:'1.0.0',dist:{integrity:'sha512-'+createHash('sha512').update(bytes).digest('base64')}};
 verifyArtifact(meta,bytes,meta.name,meta.version);
 for(const change of [{name:'@other/example'},{version:'1.0.1'},{deprecated:'recalled'},{dist:{integrity:'sha512-wrong'}}])assert.throws(()=>verifyArtifact({...meta,...change},bytes,meta.name,meta.version));
 assert.throws(()=>verifyArtifact(meta,Buffer.from('different bytes'),meta.name,meta.version));
});
test('only exact public tarball identities are publishable',async()=>{
 const temp=mkdtempSync(join(tmpdir(),'npm-test-'));try{
  mkdirSync(join(temp,'package'));writeFileSync(join(temp,'package/package.json'),JSON.stringify({name:'@temporalio/example',version:'1.0.0'}));
  await create({file:join(temp,'tested.tgz'),cwd:temp,gzip:true},['package/package.json']);
  assert.equal(await artifact(temp,'@temporalio/example','1.0.0'),join(temp,'tested.tgz'));
  await assert.rejects(()=>artifact(temp,'@temporalio/example','1.0.1'));
  writeFileSync(join(temp,'package/package.json'),JSON.stringify({name:'@temporalio/example',version:'1.0.0',private:true}));
  await create({file:join(temp,'tested.tgz'),cwd:temp,gzip:true},['package/package.json']);
  await assert.rejects(()=>artifact(temp,'@temporalio/example','1.0.0'));
 }finally{rmSync(temp,{recursive:true,force:true});}
});
test('shared and Bazel changes fan out; plugin changes remain isolated',()=>{
 const names=['one','two'];assert.deepEqual(selected(['typescript/one/src/index.ts'],names),['one']);
 for(const path of ['typescript/_tools/src/ci.ts','MODULE.bazel.lock','.github/workflows/ci.yml','bazel/typescript.bzl'])assert.deepEqual(selected([path],names),names);
 assert.deepEqual(selected(['python/mcp/plugin.toml'],names),[]);
});
