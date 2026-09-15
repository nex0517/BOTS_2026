import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const publicDir = path.join(here, 'public');
const predictionsDir = path.join(root, 'submission', 'organizer', 'development_predictions');
const dataDir = process.env.BRANCHSEED_DATA_ROOT ||
  path.join(process.env.HOME || '', 'Downloads', 'TORALIS CHALLENGE ');
const port = Number(process.env.PORT || 3000);
const mime = {'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json; charset=utf-8','.gz':'application/octet-stream'};
const valid = id => /^subject(?:00[1-9]|01\d|02[0-5])$/.test(id);
const recentRuns = new Map();

async function analysisCase(id) {
  const number = Number(id.slice(-3));
  const folder = path.join(dataDir, id);
  const locate = async stem => {
    for (const extension of ['.nii', '.nii.gz']) {
      const file = path.join(folder, `${stem}${number}${extension}`);
      try { await fs.access(file); return file; } catch { /* try next extension */ }
    }
    throw Object.assign(new Error(`NIfTI input missing for ${id} in ${folder}`), {status:404});
  };
  const image = await locate('orig');
  const mask = await locate('mask');
  const started = performance.now();
  const output = path.join(predictionsDir, `${id}.json`);
  await fs.mkdir(predictionsDir, {recursive:true});
  await new Promise((resolve,reject) => {
    const child = spawn(process.env.PYTHON || 'python3',
      ['run.py','--image',image,'--aorta-mask',mask,'--output',output],
      {cwd:root,env:process.env});
    let errorText='';
    child.stderr.on('data', chunk => errorText += chunk.toString().slice(0,5000));
    child.on('error',reject);
    child.on('close', code => code===0 ? resolve() : reject(new Error(errorText || `Detector exited with code ${code}`)));
  });
  const runtime_s = Math.round((performance.now()-started)/10)/100;
  recentRuns.set(id,runtime_s);
  // The single-file CLI names a case after origN; the organizer dashboard uses
  // the containing subject folder, as self_test.py does for its 25-case run.
  const prediction = JSON.parse(await fs.readFile(output,'utf8'));
  prediction.case_id = id;
  await fs.writeFile(output,JSON.stringify(prediction,null,2)+'\n');
  return {prediction,runtime_s};
}

async function sendFile(res, file, type, compressed=false) {
  const content = await fs.readFile(file);
  res.writeHead(200, {'Content-Type':type,'Content-Length':content.length,
    'Cache-Control':'no-cache',...(compressed?{'Content-Encoding':'gzip'}:{})});
  res.end(content);
}
const server=http.createServer(async(req,res)=>{
  try {
    const url=new URL(req.url || '/',`http://localhost:${port}`);
    const bits=url.pathname.split('/').filter(Boolean);
    if(url.pathname==='/api/cases') {
      const cases=[];
      let report={cases:[]};
      try { report=JSON.parse(await fs.readFile(path.join(root,'submission','organizer','self_test_report.json'),'utf8')); } catch {}
      const runInfo = new Map(report.cases.map(c=>[c.case,c]));
      for(let i=1;i<=25;i++) {
        const id=`subject${String(i).padStart(3,'0')}`;
        const info=runInfo.get(id);
        try {
          const d=JSON.parse(await fs.readFile(path.join(predictionsDir,`${id}.json`),'utf8'));
          cases.push({id,number:i,count:d.daughters.length,runtime_s:recentRuns.get(id)??info?.runtime_s??null});
        } catch { cases.push({id,number:i,count:null,runtime_s:null}); }
      }
      const body=JSON.stringify(cases);
      res.writeHead(200,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-cache'});res.end(body);return;
    }
    if(bits[0]==='api'&&bits[1]==='case'&&valid(bits[2])&&bits[3]==='run'&&bits.length===4&&req.method==='POST') {
      const body=JSON.stringify(await analysisCase(bits[2]));
      res.writeHead(200,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-cache'});res.end(body);return;
    }
    if(bits[0]==='api'&&bits[1]==='case'&&valid(bits[2])&&bits.length===3) {
      await sendFile(res,path.join(predictionsDir,`${bits[2]}.json`),mime['.json']);return;
    }
    if(bits[0]==='api'&&bits[1]==='volume'&&valid(bits[2])&&bits.length===4) {
      if(bits[3]==='meta') await sendFile(res,path.join(publicDir,'volumes',`${bits[2]}.meta.json`),mime['.json']);
      else if(bits[3]==='data') await sendFile(res,path.join(publicDir,'volumes',`${bits[2]}.raw.gz`),mime['.gz'],true);
      else throw Object.assign(new Error('Not found'),{status:404});
      return;
    }
    const rel=url.pathname==='/'?'pitch.html':decodeURIComponent(url.pathname.slice(1));
    const file=path.resolve(publicDir,rel);
    if(!file.startsWith(publicDir+path.sep)&&file!==path.join(publicDir,'index.html')) throw Object.assign(new Error('Forbidden'),{status:403});
    await sendFile(res,file,mime[path.extname(file)]||'application/octet-stream');
  } catch(error) {
    const status=error?.status || (error?.code==='ENOENT'?404:500);
    res.writeHead(status,{'Content-Type':'application/json'});
    res.end(JSON.stringify({error:status===500?'Server error':error.message}));
    if(status===500) console.error(error);
  }
});
server.listen(port,'127.0.0.1',()=>console.log(`Branchseed dashboard: http://127.0.0.1:${port}`));
