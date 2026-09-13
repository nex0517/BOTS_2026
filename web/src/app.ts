type Vec3=[number,number,number];
type Daughter={instance_id:string;parent_instance_id:string;ostium_xyz_mm:Vec3;seed_xyz_mm:Vec3;radius_mm:number;direction_xyz:Vec3};
type Prediction={case_id:string;parent:{instance_id:string};daughters:Daughter[]};
type CaseSummary={id:string;number:number;count:number|null;runtime_s:number|null};
type VolumeMeta={case_id:string;width:number;height:number;depth:number;offset_xyz:Vec3;origin_xyz_mm:Vec3;basis_xyz_mm:number[][];source_shape_zyx:number[];window_hu:number[]};
const el=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const canvas=el<HTMLCanvasElement>('scan-canvas');
const ctx=canvas.getContext('2d')!;
let cases:CaseSummary[]=[];let active='';let prediction:Prediction|null=null;let meta:VolumeMeta|null=null;
let volume:Uint8Array|null=null;let slice=0;let selectedBranch:string|null=null;let requestId=0;
const imageCanvas=document.createElement('canvas');const imageCtx=imageCanvas.getContext('2d')!;
const fmt=(v:number,d=1)=>Number(v).toFixed(d);
const caseLabel=(id:string)=>`Subject ${id.slice(-3)}`;
const base=(n:number,min:number,max:number)=>Math.max(min,Math.min(max,n));
function inverse3(a:number[][]):number[][] {
 const [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]=a;
 const det=a00*(a11*a22-a12*a21)-a01*(a10*a22-a12*a20)+a02*(a10*a21-a11*a20);
 if(Math.abs(det)<1e-10)throw new Error('Invalid volume coordinate transform');
 return [[(a11*a22-a12*a21)/det,(a02*a21-a01*a22)/det,(a01*a12-a02*a11)/det],[(a12*a20-a10*a22)/det,(a00*a22-a02*a20)/det,(a02*a10-a00*a12)/det],[(a10*a21-a11*a20)/det,(a01*a20-a00*a21)/det,(a00*a11-a01*a10)/det]];
}
let inverse:number[][]=[];
function toVoxel(point:Vec3):Vec3 {
 if(!meta)return [0,0,0];
 const v=point.map((n,i)=>n-meta!.origin_xyz_mm[i]);
 const q=inverse.map(row=>row.reduce((s,n,i)=>s+n*v[i],0));
 return [q[0]-meta.offset_xyz[0],q[1]-meta.offset_xyz[1],q[2]-meta.offset_xyz[2]];
}
function drawArrow(x1:number,y1:number,x2:number,y2:number,color:string,strong:boolean) {
 const length=Math.hypot(x2-x1,y2-y1);
 ctx.save();ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=strong?3.5:2.5;ctx.lineCap='round';
 if(length<8){ctx.beginPath();ctx.arc(x1,y1,strong?12:9,0,Math.PI*2);ctx.stroke();}
 else {ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();const angle=Math.atan2(y2-y1,x2-x1);ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-11*Math.cos(angle-.5),y2-11*Math.sin(angle-.5));ctx.lineTo(x2-11*Math.cos(angle+.5),y2-11*Math.sin(angle+.5));ctx.closePath();ctx.fill();}
 ctx.restore();
}
function renderSlice(){
 ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#070e17';ctx.fillRect(0,0,canvas.width,canvas.height);
 if(!meta||!volume)return;
 const {width,height,depth}=meta;const pixels=new Uint8ClampedArray(width*height*4);
 const baseIndex=slice*width*height*2;
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){
   const source=(y*width+x)*2+baseIndex;const target=(y*width+x)*4;
   const gray=volume[source],mask=volume[source+1]>0;
   let edge=false;
   if(mask){const left=x===0?0:volume[source-2+1],right=x===width-1?0:volume[source+2+1],top=y===0?0:volume[source-width*2+1],bottom=y===height-1?0:volume[source+width*2+1];edge=!(left&&right&&top&&bottom);}
   pixels[target]=edge?47:mask?Math.round(gray*.65+18):gray;
   pixels[target+1]=edge?228:mask?Math.round(gray*.7+80):gray;
   pixels[target+2]=edge?218:mask?Math.round(gray*.8+85):gray;
   pixels[target+3]=255;
 }
 imageCanvas.width=width;imageCanvas.height=height;imageCtx.putImageData(new ImageData(pixels,width,height),0,0);
 const scale=Math.min((canvas.width-44)/width,(canvas.height-44)/height);
 const x0=(canvas.width-width*scale)/2,y0=(canvas.height-height*scale)/2;
 ctx.imageSmoothingEnabled=true;ctx.drawImage(imageCanvas,x0,y0,width*scale,height*scale);
 const near=prediction?.daughters.map(d=>({d,o:toVoxel(d.ostium_xyz_mm),s:toVoxel(d.seed_xyz_mm)})).filter(v=>Math.abs(v.o[2]-slice)<=2.5)||[];
 for(const {d,o,s} of near){
   const strong=d.instance_id===selectedBranch;const color=strong?'#ffc271':'#ff876d';
   const sx=x0+o[0]*scale,sy=y0+o[1]*scale;
   const vx=(s[0]-o[0])*scale,vy=(s[1]-o[1])*scale;
   const mag=Math.hypot(vx,vy);const factor=mag<14&&mag>0?14/mag:1;
   drawArrow(sx,sy,sx+vx*factor,sy+vy*factor,color,strong);
   ctx.beginPath();ctx.fillStyle='#fff';ctx.arc(sx,sy,strong?5.5:4.5,0,Math.PI*2);ctx.fill();
   ctx.font='600 13px system-ui';const label=d.instance_id.replace('branch_','BR ')+(Math.abs(s[2]-o[2])>2?` · ${s[2]>o[2]?'↑':'↓'} Z`:'');
   const tx=base(sx+12,8,canvas.width-95),ty=base(sy-15,22,canvas.height-8);
   const tw=ctx.measureText(label).width+16;ctx.fillStyle='rgba(10,22,35,.83)';ctx.fillRect(tx-7,ty-16,tw,23);ctx.fillStyle='#fff';ctx.fillText(label,tx,ty);
 }
 el('slice-badge').textContent=`SLICE ${String(slice+meta.offset_xyz[2]).padStart(3,'0')}`;
 const center:[number,number,number]=[meta.offset_xyz[0]+width/2,meta.offset_xyz[1]+height/2,meta.offset_xyz[2]+slice];
 const physical=meta.origin_xyz_mm.map((v,i)=>v+meta!.basis_xyz_mm[i].reduce((sum,n,j)=>sum+n*center[j],0));
 el('scan-coords').textContent=`Z ${fmt(physical[2])} mm · ${near.length} ostia on slice`;
 el<HTMLInputElement>('slice-slider').value=String(slice);el('slice-position').textContent=`${slice+1} / ${depth}`;
}
function setSlice(next:number){if(!meta)return;slice=base(next,0,meta.depth-1);renderSlice();}
function renderCases(){
 const side=el('sidebar-cases');const select=el<HTMLSelectElement>('case-select');side.replaceChildren();select.replaceChildren();
 for(const item of cases){
   const option=document.createElement('option');option.value=item.id;option.textContent=caseLabel(item.id);select.append(option);
   const button=document.createElement('button');button.className='case-link'+(item.id===active?' active':'');button.innerHTML=`<span>${String(item.number).padStart(2,'0')}</span><b>${caseLabel(item.id)}</b><em>${item.count??'—'}</em>`;
   button.onclick=()=>void loadCase(item.id);side.append(button);
 }
 select.value=active;
}
function renderPrediction(){if(!prediction)return;
 const p=prediction,item=cases.find(v=>v.id===active),count=p.daughters.length;
 el('title').textContent=caseLabel(active);el('breadcrumb-case').textContent=caseLabel(active);
 el('candidate-count').textContent=String(count);el('summary-found').textContent=String(count);el('branch-count').textContent=String(count);
 el('runtime-value').textContent=item?.runtime_s==null?'—':`${fmt(item.runtime_s,2)} s`;
 el('runtime-sub').textContent=item?.runtime_s==null?'No recorded run time':'Latest recorded case run';
 el('count-sub').textContent=count===1?'Predicted direct branch':'Predicted direct branches';
 const bar=el('count-bar'),badge=el('result-badge');
 const maximum=Math.max(1,...cases.map(v=>v.count??0));
 el('bar-label').textContent='Relative candidate count';el('summary-target').textContent=`${maximum} max`;
 bar.style.width=`${Math.min(100,count/maximum*100)}%`;bar.className='progress-fill';
 badge.textContent='PREDICTION';badge.className='badge';
 el('bar-caption').textContent='Bar length shows the candidate count relative to the largest current case.';
 const list=el('branch-list');list.replaceChildren();
 if(!count){const empty=document.createElement('p');empty.className='empty';empty.textContent='No daughter candidates were predicted for this case.';list.append(empty);}
 for(const d of p.daughters){const button=document.createElement('button');button.className='branch-item'+(selectedBranch===d.instance_id?' selected':'');
  const head=document.createElement('div');head.className='branch-item-head';const name=document.createElement('strong');name.textContent=d.instance_id.replace('_',' ');const radius=document.createElement('span');radius.textContent=`${fmt(d.radius_mm)} mm radius`;head.append(name,radius);
  const track=document.createElement('div');track.className='mini-track';const fill=document.createElement('div');fill.style.width=`${Math.min(100,d.radius_mm/8*100)}%`;track.append(fill);
  const coords=document.createElement('small');coords.textContent=`Ostium · ${d.ostium_xyz_mm.map(v=>fmt(v)).join(', ')} mm`;
  button.append(head,track,coords);button.onclick=()=>{selectedBranch=d.instance_id;renderPrediction();const v=toVoxel(d.ostium_xyz_mm);setSlice(Math.round(v[2]));};list.append(button);
 }
 el('json-output').textContent=JSON.stringify(p,null,2);
}
async function loadCase(id:string){
 if(!/^subject(?:00[1-9]|01\d|02[0-5])$/.test(id))return;
 const seq=++requestId;active=id;selectedBranch=null;prediction=null;meta=null;volume=null;
 history.replaceState(null,'',`?case=${id}`);renderCases();el('scan-loading').hidden=false;el('scan-loading').textContent='Loading CT volume…';
 try{
  const [pr,mr,vr]=await Promise.all([fetch(`/api/case/${id}`),fetch(`/api/volume/${id}/meta`),fetch(`/api/volume/${id}/data`)]);
  if(!pr.ok||!mr.ok||!vr.ok)throw new Error('Case data is unavailable.');
  const [p,m,v]=await Promise.all([pr.json() as Promise<Prediction>,mr.json() as Promise<VolumeMeta>,vr.arrayBuffer()]);
  if(seq!==requestId)return;
  prediction=p;meta=m;volume=new Uint8Array(v);inverse=inverse3(m.basis_xyz_mm);
  if(volume.length!==m.width*m.height*m.depth*2)throw new Error('CT volume size does not match metadata.');
  el('slice-count').textContent=String(m.depth);el<HTMLInputElement>('slice-slider').max=String(m.depth-1);
  const first=p.daughters[0];slice=first?base(Math.round(toVoxel(first.ostium_xyz_mm)[2]),0,m.depth-1):Math.floor(m.depth/2);
  el('scan-loading').hidden=true;renderPrediction();renderSlice();
 }catch(error){if(seq!==requestId)return;el('scan-loading').hidden=false;el('scan-loading').textContent=error instanceof Error?error.message:'Could not load case.';}
}
async function main(){
 try{const response=await fetch('/api/cases');if(!response.ok)throw new Error('Unable to load patient list');cases=await response.json() as CaseSummary[];el('sidebar-count').textContent=String(cases.length);
  const id=new URLSearchParams(location.search).get('case');await loadCase(id&&cases.some(v=>v.id===id)?id:cases[0].id);
 }catch(error){el('scan-loading').textContent=error instanceof Error?error.message:'Could not load dashboard.';}
}
el<HTMLSelectElement>('case-select').addEventListener('change',e=>void loadCase((e.target as HTMLSelectElement).value));
el<HTMLInputElement>('slice-slider').addEventListener('input',e=>setSlice(Number((e.target as HTMLInputElement).value)));
el('prev-slice').addEventListener('click',()=>setSlice(slice-1));el('next-slice').addEventListener('click',()=>setSlice(slice+1));
el('scan-stage').addEventListener('wheel',e=>{e.preventDefault();setSlice(slice+(e.deltaY>0?1:-1));},{passive:false});
el('copy-json').addEventListener('click',async()=>{if(!prediction)return;await navigator.clipboard.writeText(JSON.stringify(prediction,null,2));const b=el('copy-json');b.textContent='Copied!';setTimeout(()=>b.textContent='Copy JSON',1500);});
el('download-json').addEventListener('click',()=>{if(!prediction)return;const blob=new Blob([JSON.stringify(prediction,null,2)+'\n'],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`${prediction.case_id}.json`;a.click();URL.revokeObjectURL(url);});
el('run-analysis').addEventListener('click',async()=>{
 if(!active)return;
 const id=active,button=el<HTMLButtonElement>('run-analysis');button.disabled=true;button.textContent='Running detector…';
 try {
  const response=await fetch(`/api/case/${id}/run`,{method:'POST'});
  const result=await response.json();
  if(!response.ok)throw new Error(result.error || 'Detector failed');
  const entry=cases.find(v=>v.id===id);
  if(entry){entry.count=result.prediction.daughters.length;entry.runtime_s=result.runtime_s;}
  if(active===id){prediction=result.prediction;selectedBranch=null;renderCases();renderPrediction();renderSlice();}
  button.textContent='Analysis updated ✓';setTimeout(()=>button.textContent='Re-run analysis ↗',1800);
 } catch(error) {button.textContent='Run failed';alert(error instanceof Error?error.message:'Detector failed');}
 finally {button.disabled=false;}
});
void main();
