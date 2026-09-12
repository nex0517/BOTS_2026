const scenarios = {
  "low-contrast": {
    case_id: "subject024", runtime_s: 38.4, peak_memory_gb: 1.26,
    note: "Per-case calibration recovered low-enhancement lumen at 1.5 mm spacing.",
    daughters: [
      {instance_id:"branch_001", ostium_xyz_mm:[-11.2,43.8,202.5], seed_xyz_mm:[-6.7,45.4,201.1], radius_mm:2.4, direction_xyz:[.918,.326,-.228], confidence:.94, map:[-102,29], path:6.8, evidence:[.96,.91,.89,.95,.88], logic:"No competing path shares this wall-opening component."},
      {instance_id:"branch_002", ostium_xyz_mm:[8.6,40.1,185.2], seed_xyz_mm:[12.8,42.4,183.8], radius_mm:2.1, direction_xyz:[.842,.461,-.279], confidence:.88, map:[74,48], path:7.4, evidence:[.91,.86,.84,.88,.79], logic:"Separated from the nearest origin by 8.2 mm."},
      {instance_id:"branch_003", ostium_xyz_mm:[-13.7,38.2,164.9], seed_xyz_mm:[-18.0,40.6,164.2], radius_mm:1.8, direction_xyz:[-.856,.482,-.139], confidence:.82, map:[-132,71], path:5.9, evidence:[.85,.81,.76,.86,.78], logic:"Path remained contrast-filled for 5.9 mm before tapering."},
      {instance_id:"branch_004", ostium_xyz_mm:[11.9,34.1,149.0], seed_xyz_mm:[16.7,35.2,148.0], radius_mm:1.7, direction_xyz:[.963,.221,-.154], confidence:.79, map:[136,85], path:6.2, evidence:[.81,.79,.73,.84,.76], logic:"Low contrast; accepted because boundary closure remained stable."},
      {instance_id:"branch_005", ostium_xyz_mm:[-9.2,31.0,127.4], seed_xyz_mm:[-13.0,33.9,125.8], radius_mm:1.5, direction_xyz:[-.754,.579,-.318], confidence:.72, map:[-68,104], path:5.4, evidence:[.75,.73,.69,.77,.66], logic:"Review band: score stays above threshold under ±6% intensity perturbation."},
      {instance_id:"branch_006", ostium_xyz_mm:[14.4,29.6,111.8], seed_xyz_mm:[18.1,32.4,109.9], radius_mm:1.6, direction_xyz:[.737,.566,-.369], confidence:.69, map:[112,120], path:5.1, evidence:[.71,.70,.66,.75,.63], logic:"Review band: shortest accepted path; cap distance is safe."}
    ]
  },
  "close-origins": {
    case_id:"subject011", runtime_s:29.7, peak_memory_gb:1.08,
    note:"Disconnected wall-opening components preserve two nearby direct daughters.",
    daughters:[
      {instance_id:"branch_001",ostium_xyz_mm:[-8.1,50.4,214.2],seed_xyz_mm:[-12.4,52.8,213.4],radius_mm:2.7,direction_xyz:[-.859,.477,-.158],confidence:.96,map:[-91,25],path:8.2,evidence:[.97,.95,.92,.96,.91],logic:"Disconnected openings: retained despite a 2.7 mm centroid distance."},
      {instance_id:"branch_002",ostium_xyz_mm:[-5.7,51.3,213.3],seed_xyz_mm:[-2.1,54.4,211.8],radius_mm:2.0,direction_xyz:[.714,.622,-.306],confidence:.91,map:[-78,27],path:7.1,evidence:[.93,.90,.87,.92,.84],logic:"Disconnected openings: retained despite a 2.7 mm centroid distance."},
      {instance_id:"branch_003",ostium_xyz_mm:[10.2,43.2,180.8],seed_xyz_mm:[14.6,45.2,179.4],radius_mm:1.9,direction_xyz:[.881,.401,-.249],confidence:.87,map:[118,53],path:6.9,evidence:[.9,.86,.83,.88,.8],logic:"One lower-scoring duplicate path was suppressed."},
      {instance_id:"branch_004",ostium_xyz_mm:[-12.5,36.8,151.6],seed_xyz_mm:[-16.9,38.7,150.2],radius_mm:1.8,direction_xyz:[-.879,.386,-.276],confidence:.83,map:[-126,81],path:6.1,evidence:[.86,.82,.78,.85,.79],logic:"Distinct wall opening and low proximal path overlap."},
      {instance_id:"branch_005",ostium_xyz_mm:[13.9,30.4,124.9],seed_xyz_mm:[17.8,33.0,123.2],radius_mm:1.7,direction_xyz:[.777,.519,-.337],confidence:.78,map:[139,107],path:5.7,evidence:[.8,.79,.73,.83,.74],logic:"No competing path shares this wall-opening component."}
    ]
  },
  "common-trunk": {
    case_id:"subject018", runtime_s:34.1, peak_memory_gb:1.19,
    note:"A shared proximal path is counted once and stops at the first stable split.",
    daughters:[
      {instance_id:"branch_001",ostium_xyz_mm:[-7.6,47.3,208.4],seed_xyz_mm:[-11.7,49.9,207.1],radius_mm:2.9,direction_xyz:[-.817,.521,-.257],confidence:.95,map:[-104,31],path:7.0,evidence:[.96,.94,.9,.95,.9],logic:"Common trunk: two distal continuations share one wall opening; counted once."},
      {instance_id:"branch_002",ostium_xyz_mm:[9.8,40.7,181.2],seed_xyz_mm:[14.0,43.1,179.9],radius_mm:2.2,direction_xyz:[.844,.482,-.261],confidence:.9,map:[95,56],path:7.8,evidence:[.92,.89,.86,.91,.85],logic:"No competing path shares this wall-opening component."},
      {instance_id:"branch_003",ostium_xyz_mm:[-12.9,35.6,154.1],seed_xyz_mm:[-17.3,37.6,152.9],radius_mm:1.9,direction_xyz:[-.877,.399,-.24],confidence:.84,map:[-132,83],path:6.3,evidence:[.88,.83,.79,.86,.78],logic:"A short returning hypothesis was rejected as parent re-entry."},
      {instance_id:"branch_004",ostium_xyz_mm:[12.7,29.8,126.0],seed_xyz_mm:[16.5,32.5,124.2],radius_mm:1.6,direction_xyz:[.761,.54,-.358],confidence:.74,map:[124,108],path:5.5,evidence:[.78,.75,.7,.8,.68],logic:"Review band: stable opening, but lower tracker margin."}
    ]
  }
};

const scoreLabels = ["Lumen continuity","Boundary closure","Radius stability","Tracker margin","Duplicate margin"];
let current = scenarios["low-contrast"];
let selectedIndex = 0;
const $ = (id) => document.getElementById(id);
const fmt = (arr, signed=false) => arr.map(v => `${signed && v >= 0 ? "+" : ""}${Number(v).toFixed(signed?3:1)}`).join(", ");
const clamp = (v,a,b) => Math.min(b,Math.max(a,v));

function scorerOutput(){
  return {case_id:current.case_id,parent:{instance_id:"aorta"},daughters:current.daughters.map(({instance_id,ostium_xyz_mm,seed_xyz_mm,radius_mm,direction_xyz})=>({instance_id,parent_instance_id:"aorta",ostium_xyz_mm,seed_xyz_mm,radius_mm,direction_xyz}))};
}

function updateSummary(){
  $("caseId").textContent=current.case_id; $("branchCount").textContent=current.daughters.length;
  $("runtime").textContent=`${current.runtime_s.toFixed(1)} s`; $("memory").textContent=`${current.peak_memory_gb.toFixed(2)} GB`;
  $("visibleCount").textContent=`${current.daughters.length} shown`; $("jsonOutput").textContent=JSON.stringify(scorerOutput(),null,2);
}

function renderMap(){
  const g=$("mapDots"); g.innerHTML="";
  current.daughters.forEach((d,i)=>{
    const x=344+(d.map?.[0]??(i*53-120))*1.7; const y=62+(d.map?.[1]??(i*17+30))*2.05;
    const r=9+d.radius_mm*2.2; const cls=d.confidence>=.8?"high":"medium";
    const node=document.createElementNS("http://www.w3.org/2000/svg","g");
    node.setAttribute("class",`map-dot ${cls} ${i===selectedIndex?"selected":""}`); node.setAttribute("transform",`translate(${clamp(x,86,602)} ${clamp(y,55,308)})`); node.setAttribute("tabindex","0"); node.setAttribute("role","button"); node.setAttribute("aria-label",`${d.instance_id}, confidence ${Math.round(d.confidence*100)} percent`);
    node.innerHTML=`<circle class="halo" r="${r+8}"></circle><circle class="core" r="${r}"></circle><text text-anchor="middle" dy="4">${i+1}</text>`;
    node.addEventListener("click",()=>selectBranch(i)); node.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();selectBranch(i)}}); g.appendChild(node);
  });
}

function renderModel(){
  const g=$("modelBranches"); g.innerHTML="";
  current.daughters.forEach((d,i)=>{
    const y=92+i*(282/Math.max(1,current.daughters.length-1)); const left=d.direction_xyz[0]<0; const x0=left?191:296; const x1=left?95:395; const rise=d.direction_xyz[1]*34; const width=7+d.radius_mm*2;
    const p=document.createElementNS("http://www.w3.org/2000/svg","path"); p.setAttribute("d",`M${x0} ${y} Q${(x0+x1)/2} ${y-rise*.55} ${x1} ${y-rise}`); p.setAttribute("class",`model-branch ${i===selectedIndex?"selected":""}`); p.setAttribute("stroke-width",width); p.dataset.i=i; p.addEventListener("click",()=>selectBranch(i)); g.appendChild(p);
    const c=document.createElementNS("http://www.w3.org/2000/svg","circle"); c.setAttribute("cx",left?145:345); c.setAttribute("cy",y-rise*.6); c.setAttribute("r",4); c.setAttribute("class",`model-seed ${i===selectedIndex?"selected":""}`); g.appendChild(c);
  });
}

function renderList(filter=""){
  const list=$("branchList"); list.innerHTML=""; let shown=0;
  current.daughters.forEach((d,i)=>{if(!d.instance_id.toLowerCase().includes(filter.toLowerCase()))return;shown++;const b=document.createElement("button");b.className=`branch-row ${i===selectedIndex?"selected":""}`;b.type="button";b.setAttribute("role","listitem");b.innerHTML=`<span class="branch-num">${i+1}</span><span class="branch-copy"><strong>${d.instance_id}</strong><small>${d.path.toFixed(1)} mm tracked</small></span><span class="branch-meta"><strong>${Math.round(d.confidence*100)}%</strong><small>r ${d.radius_mm.toFixed(1)}</small></span>`;b.addEventListener("click",()=>selectBranch(i));list.appendChild(b)});
  $("visibleCount").textContent=`${shown} shown`;
}

function drawSlice(id, seed, orientation, cross=true){
  const c=$(id),ctx=c.getContext("2d"),w=c.width,h=c.height; const img=ctx.createImageData(w,h); let s=(Math.abs(seed[0]*97+seed[1]*43+seed[2]*11)+orientation*131)|0;
  const rand=()=>{s=(s*1664525+1013904223)>>>0;return s/4294967296};
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){const dx=(x-w*.5)/w,dy=(y-h*.5)/h;const body=Math.exp(-(dx*dx*3.1+dy*dy*2.3));const spine=Math.exp(-((dx+.01)**2*90+(dy-.25)**2*160));const organ=.22*Math.sin(x*.045+orientation)+.18*Math.cos(y*.052-orientation);const noise=(rand()-.5)*.24;let v=(.11+body*.38+spine*.28+organ*.16+noise)*255;const k=(y*w+x)*4;img.data[k]=v*.86;img.data[k+1]=v*.9;img.data[k+2]=v*.92;img.data[k+3]=255}ctx.putImageData(img,0,0);
  const cx=w*.5+(seed[0]%9)*2.2,cy=h*.51+(seed[1]%7)*1.6;ctx.strokeStyle="#ff8a4c";ctx.lineWidth=2;ctx.beginPath();ctx.arc(cx,cy,17,0,Math.PI*2);ctx.stroke();ctx.fillStyle="rgba(92,225,230,.22)";ctx.beginPath();ctx.arc(cx+(orientation-1)*10,cy-8,6,0,Math.PI*2);ctx.fill();if(cross){ctx.strokeStyle="#5ce1e6";ctx.lineWidth=1;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(cx-34,cy);ctx.lineTo(cx+34,cy);ctx.moveTo(cx,cy-34);ctx.lineTo(cx,cy+34);ctx.stroke();ctx.setLineDash([])}
}

function drawRadius(d){
  const c=$("radiusPlane"),ctx=c.getContext("2d"),w=c.width,h=c.height;ctx.fillStyle="#050a0e";ctx.fillRect(0,0,w,h);const grad=ctx.createRadialGradient(w/2,h/2,4,w/2,h/2,88);grad.addColorStop(0,"#d7e2e2");grad.addColorStop(.26,"#89999c");grad.addColorStop(.47,"#303b3f");grad.addColorStop(1,"#080d10");ctx.fillStyle=grad;ctx.fillRect(0,0,w,h);const r=24+d.radius_mm*8;ctx.beginPath();ctx.arc(w/2,h/2,r,0,Math.PI*2);ctx.strokeStyle="#5ce1e6";ctx.lineWidth=3;ctx.stroke();ctx.beginPath();ctx.moveTo(w/2,h/2);ctx.lineTo(w/2+r,h/2);ctx.strokeStyle="#ff8a4c";ctx.lineWidth=2;ctx.stroke();ctx.fillStyle="#ff8a4c";ctx.font="700 12px ui-monospace";ctx.fillText(`${d.radius_mm.toFixed(1)} mm`,w/2+12,h/2-8)
}

function renderEvidence(){
  const d=current.daughters[selectedIndex]; if(!d)return;
  $("branchTitle").textContent=d.instance_id.replace("_"," ").replace(/\b\w/g,x=>x.toUpperCase());$("selectedId").textContent=d.instance_id;$("selectedPath").textContent=`${d.path.toFixed(1)} mm proven`;
  $("confidenceBadge").innerHTML=`<span></span><strong>${Math.round(d.confidence*100)}%</strong> ${d.confidence>=.8?"high":"review"}`;$("confidenceBadge").classList.toggle("review",d.confidence<.8);
  $("ostium").textContent=fmt(d.ostium_xyz_mm);$("seed").textContent=fmt(d.seed_xyz_mm);$("direction").textContent=fmt(d.direction_xyz,true);$("axialZ").textContent=`z ${d.ostium_xyz_mm[2].toFixed(1)} mm`;$("radiusLabel").textContent=`r ${d.radius_mm.toFixed(1)} mm`;
  drawSlice("axial",d.ostium_xyz_mm,0);drawSlice("coronal",d.ostium_xyz_mm,1);drawSlice("sagittal",d.ostium_xyz_mm,2);drawRadius(d);
  $("logicTitle").textContent=d.logic.startsWith("Common")?"Common-trunk rule passed":d.logic.startsWith("Disconnected")?"Close-origin rule passed":"Deduplication passed";$("logicText").textContent=d.logic;
  $("scoreBars").innerHTML=scoreLabels.map((l,i)=>`<div class="score-row"><span>${l}</span><div class="bar"><i style="width:${Math.round(d.evidence[i]*100)}%"></i></div><strong>${Math.round(d.evidence[i]*100)}</strong></div>`).join("");
}

function selectBranch(i){selectedIndex=i;renderMap();renderModel();renderList($("branchSearch").value);renderEvidence()}
function loadScenario(key){current=scenarios[key];selectedIndex=0;updateSummary();renderMap();renderModel();renderList();renderEvidence();toast(current.note)}
function toast(msg){const t=$("toast");t.textContent=msg;t.classList.add("show");clearTimeout(toast.timer);toast.timer=setTimeout(()=>t.classList.remove("show"),2800)}

$("caseSelect").addEventListener("change",e=>loadScenario(e.target.value));
$("branchSearch").addEventListener("input",e=>renderList(e.target.value));
$("uploadBtn").addEventListener("click",()=>$("fileInput").click());
$("fileInput").addEventListener("change",async e=>{const f=e.target.files[0];if(!f)return;try{const data=JSON.parse(await f.text());if(!Array.isArray(data.daughters))throw new Error("Missing daughters array");data.daughters=data.daughters.map((d,i)=>({...d,confidence:d.confidence??.78,map:d.map??[-115+i*48,34+i*18],path:d.path??5,evidence:d.evidence??[.78,.76,.72,.75,.71],logic:d.logic??"Imported scorer output; diagnostic evidence was not included."}));current={case_id:data.case_id||"imported_case",runtime_s:data.runtime_s||0,peak_memory_gb:data.peak_memory_gb||0,note:"Imported local prediction JSON.",daughters:data.daughters};selectedIndex=0;updateSummary();renderMap();renderModel();renderList();renderEvidence();toast("Prediction loaded locally")}catch(err){toast(`Could not open JSON: ${err.message}`)}finally{e.target.value=""}});
$("downloadBtn").addEventListener("click",()=>{const blob=new Blob([JSON.stringify(scorerOutput(),null,2)],{type:"application/json"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=`${current.case_id}_prediction.json`;a.click();URL.revokeObjectURL(a.href);toast("Strict prediction JSON exported")});
$("jsonBtn").addEventListener("click",()=>$("jsonDialog").showModal());$("closeDialog").addEventListener("click",()=>$("jsonDialog").close());
$("animateBtn").addEventListener("click",()=>{const p=document.querySelector(".model-branch.selected");if(!p)return;const len=p.getTotalLength();p.style.strokeDasharray=len;p.style.strokeDashoffset=len;p.animate([{strokeDashoffset:len},{strokeDashoffset:0}],{duration:1100,easing:"cubic-bezier(.2,.8,.2,1)"});toast("Tracing selected proximal path: 0 → 10 mm")});
window.addEventListener("resize",()=>renderEvidence());
loadScenario("low-contrast");
