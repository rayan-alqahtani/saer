import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
const require = createRequire(new URL('../frontend/package.json', import.meta.url));
const { transformWithOxc } = await import(pathToFileURL(require.resolve('vite')).href);
const source = await fs.readFile(new URL('../frontend/src/ReferenceApp.tsx', import.meta.url), 'utf8');
async function moduleFrom(path, extra = '') {
  const code = (await fs.readFile(new URL(path, import.meta.url), 'utf8')).split(/\r?\n/).filter(line => !line.trimStart().startsWith('import ')).join('\n') + extra;
  const result = await transformWithOxc(code, 'fixture.ts');
  return import('data:text/javascript;base64,' + Buffer.from(result.code).toString('base64'));
}
function body(marker) {
  let start = source.indexOf('=> {', source.indexOf(marker)) + 4, depth = 1, end = start;
  for (; depth; end++) { if (source[end] === '{') depth++; if (source[end] === '}') depth--; }
  return source.slice(start, end - 1);
}
function action(marker, params, context) {
  return Function('context', `return async function(${params}) { with(context) { ${body(marker)} } }`)(context);
}
function deferred() { let resolve, reject; const promise = new Promise((a,b) => {resolve=a;reject=b}); return {promise,resolve,reject}; }

test('JSON failures stay Arabic and rejected listings cache can retry', async () => {
  const previous = globalThis.fetch;
  const service = await moduleFrom('../frontend/src/services/demoService.ts', '\nexport { allListings };');
  let calls = 0;
  globalThis.fetch = async () => { calls++; return {ok:true,json:async()=>{throw new SyntaxError('Unexpected token <')}}; };
  try {
    await assert.rejects(service.allListings(), /تعذر تحميل بيانات العرض التجريبي/);
    globalThis.fetch = async () => { calls++; return {ok:true,json:async()=>({items:[]})}; };
    assert.deepEqual(await service.allListings(), {items:[]});
    await service.allListings(); assert.equal(calls, 2);
  } finally { globalThis.fetch = previous; }
});

for (const kind of ['picker', 'listings', 'market']) {
  test(`${kind}: older successful response cannot replace latest response`, async () => {
    const first=deferred(),second=deferred(); let calls=0, value, loading;
    const invoke=()=> (++calls===1?first:second).promise;
    const context={requestId:{current:0},uiService:{listingsPage:invoke,market:invoke},filter:{},setLoading:v=>loading=v,setError:()=>{},setItems:v=>value=typeof v==='function'?v([]):v,setTotal:()=>{},setOffset:()=>{},setMarket:v=>value=v};
    const marker=kind==='picker'?'const load = async':kind==='listings'?'const loadListings = async':'const loadMarket = async';
    const load=action(marker,kind==='picker'?'nextOffset,query':kind==='listings'?'next=0':'selected={}',context);
    const a=load(0,''),b=load(0,'new');
    const latest=kind==='market'?{notice:'latest'}:{items:['latest'],total:1};
    second.resolve(latest);await b;first.resolve(kind==='market'?{notice:'old'}:{items:['old'],total:1});await a;
    assert.deepEqual(value,kind==='market'?latest:['latest']);assert.equal(loading,false);
  });
}

test('picker catches rejection with Arabic error', async () => {
  let message;const context={requestId:{current:0},uiService:{listingsPage:async()=>{throw Error('network')}},setLoading:()=>{},setError:v=>message=v};
  await action('const load = async','nextOffset,query',context)(0,'');
  assert.match(message,/تعذر تحميل الإعلانات/);
});

test('prepareImage validates MIME, size and decoding and revokes failed URL', async () => {
  const prepare=await moduleFrom('../frontend/src/services/imageUpload.ts');
  const oldImage=globalThis.Image,oldURL=globalThis.URL;
  let created=0,revoked=[];
  globalThis.URL={createObjectURL:()=>`blob:${++created}`,revokeObjectURL:u=>revoked.push(u)};
  globalThis.Image=class {naturalWidth=10;naturalHeight=10;async decode(){throw Error('invalid')}};
  try {
    await assert.rejects(prepare.prepareImage({type:'text/plain',size:1}),/اختر صورة/);
    await assert.rejects(prepare.prepareImage({type:'image/png',size:12*1024*1024+1}),/12/);
    assert.equal(created,0);
    await assert.rejects(prepare.prepareImage({type:'image/png',size:1}),/تعذّر فتح الصورة/);
    assert.deepEqual(revoked,['blob:1']);
  } finally {globalThis.Image=oldImage;globalThis.URL=oldURL;}
});

test('choose rejects multiple files, allows retry and revokes a superseded asynchronous preview', async () => {
  let calls=0,files,previews,status;const a=deferred(),b=deferred(),revoked=[];
  const context={selectionId:{current:0},prepareImage:()=> (++calls===1?a:b).promise,URL:{revokeObjectURL:u=>revoked.push(u)},setImagePrice:()=>{},setStatus:v=>status=v,setFiles:v=>files=v,setPreviews:v=>previews=v,input:{current:null}};
  const choose=action('const choose = async','selected',context);
  await choose(['first','unused']); assert.equal(calls,0); assert.match(status,/صورة واحدة فقط/);
  const p=choose(['first']),q=choose(['new']);b.resolve('blob:new');await q;a.resolve('blob:old');await p;
  assert.deepEqual(files,['new']);assert.deepEqual(previews,['blob:new']);assert.deepEqual(revoked,['blob:old']);assert.equal(status,'');
});

test('image request follows single/ListFiles schema and always sends one image', async () => {
  const oldClient=globalThis.uploadClient;
  const sent=[];
  const client={config:{dependencies:[{api_name:'predict_image',inputs:[8]}],components:[{id:8,type:'file',props:{file_count:'multiple'}}]},predict:async(name,args)=>{sent.push({name,args});return {data:[{price_sar:123}]};}};
  globalThis.uploadClient=client;
  try {
    const service=await moduleFrom('../frontend/src/services/demoService.ts', `
      const Client={connect:async()=>globalThis.uploadClient};
      const resolveApiBaseUrl=async()=> 'https://example.test';
      const validateImage=file=>{if(!file)throw Error('اختر صورة السيارة أولًا.');};
      const handle_file=file=>({path:file.name,meta:{_type:'gradio.FileData'}});
    `);
    const file={name:'one.webp'};
    await service.estimateFromImage(file);
    assert.equal(sent[0].args.length,1);assert.equal(sent[0].args[0].length,1);
    assert.equal(sent[0].args[0][0].path,'one.webp');
    client.config.components[0].props.file_count='single';
    await service.estimateFromImage({name:'two.webp'});
    assert.equal(sent[1].args.length,1);assert.equal(sent[1].args[0].path,'two.webp');
    await assert.rejects(service.estimateFromImages([file,file]),/صورة واحدة فقط/);
    await assert.rejects(service.estimateFromImages([]),/صورة واحدة فقط/);
    assert.equal(sent.length,2);
    client.config.components[0].props.file_count='directory';
    await assert.rejects(service.estimateFromImage(file),/غير متوافق/);
  } finally {globalThis.uploadClient=oldClient;}
});
