"use strict";
const $ = id => document.getElementById(id);
let state = null, selectedStage = null, selectedFrame = 0, playing = false, sourceKey = '', sourceImage = null, productKey = '';
const fmt = (x, digits=3) => Number.isFinite(x) ? x.toFixed(digits) : '--';
const text = (id, value) => { $(id).textContent = value; };
const heat = t => {t=Math.max(0,Math.min(1,t)); const stops=[[12,20,42],[89,43,118],[187,59,103],[244,133,88],[252,243,166]];const i=Math.min(3,Math.floor(t*4)), f=t*4-i;return stops[i].map((v,k)=>Math.round(v+(stops[i+1][k]-v)*f));};

function empty(canvas, message='No observation yet') {
  const c=canvas.getContext('2d');c.fillStyle='#0b1521';c.fillRect(0,0,canvas.width,canvas.height);
  c.fillStyle='#91a6ba';c.font='13px system-ui';c.textAlign='center';c.fillText(message,canvas.width/2,canvas.height/2);
}
function drawGrid(id, grid, {mode='heat', lo=0, hi=null}={}) {
  const canvas=$(id);if(!grid?.length || !grid[0]?.length){empty(canvas);return;}
  const h=grid.length,w=grid[0].length, flat=grid.flat().filter(Number.isFinite);
  if(hi===null)hi=flat.reduce((m,v)=>Math.max(m,v),1);if(hi<=lo)hi=lo+1;
  const tile=document.createElement('canvas');tile.width=w;tile.height=h;
  const tc=tile.getContext('2d'), data=tc.createImageData(w,h);
  for(let r=0;r<h;r++)for(let c=0;c<w;c++){
    const v=grid[r][c];let rgb=[69,85,103];
    if(Number.isFinite(v)){
      const t=Math.max(0,Math.min(1,(v-lo)/(hi-lo)));
      if(mode==='status')rgb=[[69,85,103],[91,173,149],[220,166,91],[188,67,104]][Math.round(v)]||rgb;
      else if(mode==='adaptive')rgb=[[126,139,152],[233,181,84],[98,191,167],[124,129,183],[200,70,98]][Math.round(v)]||rgb;
      else if(mode==='gray')rgb=[t*255,t*255,t*255];
      else if(mode==='residual'){const x=Math.max(-1,Math.min(1,v/Math.max(Math.abs(lo),Math.abs(hi),1e-9)));rgb=x<0?[230*(1+x),230*(1+x),230]:[230,230*(1-x),230*(1-x)];}
      else rgb=heat(t);
    }
    const off=(r*w+c)*4;data.data.set([...rgb,255],off);
  }
  tc.putImageData(data,0,0);const ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;
  ctx.fillStyle='#0b1521';ctx.fillRect(0,0,canvas.width,canvas.height);
  const scale=Math.min(canvas.width/w,canvas.height/h),dw=w*scale,dh=h*scale;
  ctx.drawImage(tile,(canvas.width-dw)/2,(canvas.height-dh)/2,dw,dh);
}

function drawSource(){
  const canvas=$('source'),ctx=canvas.getContext('2d');if(!sourceImage||!state?.inputs){empty(canvas,'Waiting for aligned image frames');return;}
  const {image_shape:shape,test_pixel:target}=state.inputs;
  const scale=Math.min(canvas.width/shape[1],canvas.height/shape[0]),w=shape[1]*scale,h=shape[0]*scale;
  const ox=(canvas.width-w)/2,oy=(canvas.height-h)/2;
  ctx.fillStyle='#0b1521';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.imageSmoothingEnabled=false;ctx.drawImage(sourceImage,ox,oy,w,h);
  function cross(row,col,color,size){ctx.strokeStyle=color;ctx.lineWidth=1.5;const x=ox+(col+.5)*scale,y=oy+(row+.5)*scale;ctx.beginPath();ctx.moveTo(x-size,y);ctx.lineTo(x+size,y);ctx.moveTo(x,y-size);ctx.lineTo(x,y+size);ctx.stroke();}
  if(target)cross(target[0],target[1],'white',8);
  const fit=state.snapshot?.fit;
  if(fit){cross(fit.root_row_px,fit.root_col_px,'#69dfd0',10);const sz=(fit.patch_size_px||fit.observed?.[0]?.length||25)*scale;ctx.strokeStyle='#69dfd0';ctx.strokeRect(ox+(fit.col_px+.5)*scale-sz/2,oy+(fit.row_px+.5)*scale-sz/2,sz,sz);}
}
function renderFrames(){
  const inputs=state.inputs, frames=inputs?.frames||[], select=$('frames');
  if(select.options.length!==frames.length || select.dataset.ids!==JSON.stringify(frames.map(f=>f.pid))){
    select.replaceChildren(...frames.map(f=>{const o=document.createElement('option');o.value=f.index;o.textContent=`${f.index+1}. ${f.pid}`;return o;}));
    select.dataset.ids=JSON.stringify(frames.map(f=>f.pid));
  }
  if(!frames.length){empty($('source'),'Waiting for input images');return;}
  selectedFrame=Math.min(selectedFrame,frames.length-1);select.value=selectedFrame;
  const f=frames[selectedFrame], key=`${inputs.demo}:${f.image}:${f.pid}`;
  if(key!==sourceKey){sourceKey=key;const image=new Image();image.onload=()=>{if(sourceKey===key){sourceImage=image;drawSource();}};image.onerror=()=>{sourceKey='';};image.src='/input/'+encodeURIComponent(f.image);}
  else drawSource();
  text('frame-id',f.pid);text('azimuth',fmt(f.azimuth_deg,2)+'°');text('elevation',fmt(f.elevation_deg,3)+'°');text('posting',fmt(inputs.pixel_m,2)+' m/pixel');text('frame-number',`${selectedFrame+1} / ${frames.length}`);
  text('source-note',inputs.meaning);$('demo').hidden=!inputs.demo;
  const c=$('sun').getContext('2d');c.clearRect(0,0,200,140);c.strokeStyle='#35495d';c.beginPath();c.arc(100,64,42,0,Math.PI*2);c.stroke();
  c.fillStyle='#91a6ba';c.font='10px system-ui';c.textAlign='center';c.fillText('map up',100,12);
  const az=f.azimuth_deg*Math.PI/180,x=100+Math.sin(az)*36,y=64-Math.cos(az)*36;
  c.strokeStyle='#efc16f';c.lineWidth=2;c.beginPath();c.moveTo(100,64);c.lineTo(x,y);c.stroke();c.fillStyle='#efc16f';c.beginPath();c.arc(x,y,5,0,Math.PI*2);c.fill();
  c.fillStyle='#91a6ba';c.fillText('Sun direction / map coordinates',100,130);
}

function renderStages(){
  const list=$('stages');list.replaceChildren();
  for(const r of state.stages){const li=document.createElement('li'),b=document.createElement('button'),strong=document.createElement('strong'),small=document.createElement('small');
    b.className=r.status.toLowerCase()+(state.selected_stage===r.id?' selected':'');b.setAttribute('aria-pressed',String(selectedStage===r.id));
    strong.textContent=r.id.startsWith('software-')?r.id.slice(9).replaceAll('_',' '):r.id==='maps'?'Three-map replay':r.id+' · '+r.title;
    small.textContent=r.status.toLowerCase().replaceAll('_',' ');b.append(strong,small);b.title=r.reason||r.title;
    b.onclick=()=>{selectedStage=r.id;updateFollow();poll();};li.append(b);list.append(li);}
  const done=state.stages.filter(r=>!['PENDING','RUNNING'].includes(r.status)).length;
  text('completed',`${done} / ${state.stages.length}`);
}
function updateFollow(){const follow=selectedStage===null;$('follow').classList.toggle('active',follow);$('follow').setAttribute('aria-pressed',String(follow));}

function renderTerrain(){
  const t=state.terrain; $('terrain-panel').hidden=!t; if(!t)return;
  drawGrid('terrain-slope',t.slope);drawGrid('terrain-rms',t.rms);
  text('terrain-context',`Fixed coordinate: row ${t.row_px}, column ${t.col_px} · native DEM ${fmt(t.native_pixel_m,2)} m/pixel · plane diameter ${fmt(t.diameter_m,2)} m. ${t.footprint_resolved?'Footprint spans the configured minimum native support.':'Footprint is finer than the supported native window.'}`);
  const metrics=$('terrain-metrics');metrics.replaceChildren();
  for(const [key,label,unit] of [['slope_deg','Slope','°'],['rms_height_m','Surface RMS',' m'],['positive_relief_m','Positive relief',' m'],['negative_relief_m','Negative relief',' m']]){
    const m=t.sample[key], dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=`${fmt(m.value)}${unit} / limit ${m.limit}${unit} → ${fmt(m.ratio)}×`;metrics.append(dt,dd);
  }
  text('terrain-index',`Local terrain index = max(ratios) / (1 + max(ratios)) = ${fmt(t.index)}. Before navigation buffering and fusion. Limits: ${t.configuration_label.replaceAll('_',' ')}. Missing measurements remain unavailable.`);
}

function renderCalculation(){
  const s=state.snapshot,fit=s?.fit;
  text('calculation-title',fit?`Last ${s.kind==='adaptive'?'adaptive':'regional'} fit · ${s.subrun}`:s?.subrun||s?.message||'Waiting for a calculation');
  const observedAt=fit?s.regional_updated:s?.updated;
  const age=observedAt?Math.max(0,(Date.now()-Date.parse(observedAt))/1000):null;
  text('snapshot-time',age===null?'No snapshot':`Snapshot ${Math.round(age)}s ago`);
  text('fit-description',fit?`Cell row ${fit.row_px}, column ${fit.col_px}. ${fit.meaning}`:'Snapshots show intermediate scientific work. Completed regional fits remain in their stage; use the sequence to inspect them.');
  const idx=fit?.frames?.indexOf(selectedFrame)??-1;
  if(fit&&idx>=0){
    const range=state.inputs?.display_range||[0,1];drawGrid('observed',fit.observed[idx],{mode:'gray',lo:range[0],hi:range[1]});
    drawGrid('template',fit.template[idx],{mode:'gray',lo:0,hi:1});
    const max=Math.max(1,...fit.residual[idx].flat().filter(Number.isFinite).map(Math.abs));drawGrid('residual',fit.residual[idx],{mode:'residual',lo:-max,hi:max});
  }else{for(const id of ['observed','template','residual'])empty($(id),fit?'Frame not eligible in this fit':'No current regional fit');}
  text('delta',fit?fmt(fit.null_energy-fit.fitted_energy):'--');text('score',fmt(fit?.score));text('index',fmt(fit?.index));
  text('index-formula',s?.kind==='adaptive'?'Experimental pass score; baseline hazard maps remain separate':`index = score / (score + ${fit?.score_scale??'scale'})`);
  const metrics=$('metrics');metrics.replaceChildren();
  if(fit){for(const [label,value] of [['Bank height',fmt(fit.height_m,2)+' m'],['Bank width',fmt(fit.width_m,2)+' m'],['Contrast',fmt(fit.contrast)],['Common support',fmt(fit.common_fraction*100,1)+'%'],['Identifiability',fmt(fit.identifiability)],['Eligible frames',String(fit.frames.length)],['Endpoint',fit.endpoint_censored?'Censored':'Inside support'],['Null energy',fmt(fit.null_energy,1)]]){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;metrics.append(dt,dd);}}
  if(fit && s.geometry_order?.some((v,i)=>v!==i)){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent='Stress model Sun';dd.textContent=`${fmt(s.model_azimuths[selectedFrame],2)}° az / ${fmt(s.model_elevations[selectedFrame],3)}° el`;metrics.append(dt,dd);}
  const bars=$('frame-bars');bars.replaceChildren();
  if(fit){const max=Math.max(1,...fit.frame_delta_chi2.map(Math.abs));fit.frames.forEach((f,i)=>{const row=document.createElement('div');row.className='frame-row';const label=document.createElement('span'),track=document.createElement('div'),bar=document.createElement('div'),value=document.createElement('span');label.textContent=`Frame ${f+1}`;track.className='track';bar.className='bar'+(fit.frame_delta_chi2[i]<0?' negative':'');bar.style.width=(Math.abs(fit.frame_delta_chi2[i])/max*100)+'%';value.textContent=fmt(fit.frame_delta_chi2[i],2);track.append(bar);row.append(label,track,value);bars.append(row);});}
}

function renderMaps(){
  const s=state.snapshot;
  text('map-title',s?.kind==='field'?s.field_title:s?.kind==='adaptive'?'Adaptive context':s?.kind==='regional'?'Regional search':'Most recent regional scan');
  if(s?.kind==='field'){drawGrid('map-score',s.field,{hi:s.field_title.includes('index')||s.field_title.includes('illumination')?1:null});empty($('map-support'),'Coverage shown during regional search');text('score-caption',s.field_title);}
  else{drawGrid('map-score',s?.score);drawGrid('map-support',s?.assessment,{mode:s?.kind==='adaptive'?'adaptive':'status',hi:s?.kind==='adaptive'?4:3});text('score-caption',s?.kind==='adaptive'?'Last-scale experimental score; different windows are not comparable significances':'Raw shadow score · colour scale spans this snapshot');}
  const total=s?.cells_total,visited=s?.cells_visited;
  $('cells').max=total||1;$('cells').value=visited||0;
  text('cell-count',total?`${visited.toLocaleString()} / ${total.toLocaleString()} cells visited`:'Waiting for regional scan');
  text('map-caption',s?.kind==='adaptive'?'Grey: not requested or unavailable · Amber: queued · Green: context supported, unvalidated · Purple: low evidence, unqualified · Pink: unresolved.':
    'Grey: unvisited/border · Green: assessed · Amber: unavailable · Pink: nonidentifiable. Partial maps are not final results.'+(s?.regional_updated&&s.kind!=='regional'?` Scan snapshot: ${new Date(s.regional_updated).toLocaleTimeString()}.`:''));
  let extra='';
  if(s?.kind==='controls'&&s.control){const c=s.control;extra=`Synthetic control: ${c.kind} · height ${c.height_m??'--'} m${Number.isFinite(c.location)?` · location ${c.location+1}`:''} · seed ${c.seed}\nAssessment: ${c.status} · maximum score ${fmt(c.maximum_score)} · recovered: ${c.recovered===null?'unknown':String(c.recovered)}`;}
  if(s?.kind==='adaptive'&&s.adaptive){const a=s.adaptive;extra=`Pass ${a.scale}× · patch ${2*a.radius_px+1} pixels · fitting radius ${a.support_px} pixels · ${a.status}\nHeight compatibility: ${a.height_range_m?.join('–')??'unavailable'} m · width: ${a.width_range_m?.join('–')??'unavailable'} m. No calibrated confidence level.\nEndpoint support by frame: ${(a.endpoint_reasons||[]).join(', ')}.`;}
  if(s?.kind==='prediction'&&s.prediction){const p=s.prediction;extra=`Withheld illumination at row ${p.row_px}, column ${p.col_px}: ${p.trials} declared trials · ${p.last_status}. Training fixes the object before the withheld frame is evaluated.`;}
  if(s?.kind==='height'&&s.height_profile){const h=s.height_profile;extra=`Height profile at row ${h.row_px}, column ${h.col_px} · support ${h.support_px} pixels · ${h.status}\nHeights (m): ${h.heights_m.join(', ')}\nScores: ${h.scores.map(v=>fmt(v,2)).join(', ')}\nDescriptive compatibility set: ${h.delta_set_m.join(', ')} m. This is not a calibrated confidence interval.`;}
  if(s?.kind==='registration'&&s.registration){const r=s.registration;extra=`Local registration: ${r.tile_px}-pixel tiles · ${r.pairs} similar-illumination pairs · ${r.measured_tiles} measured tiles. Offsets do not modify the input images.`;}
  if(s?.kind==='field'&&s.metrics)extra=Object.entries(s.metrics).map(([k,v])=>`${k.replaceAll('_',' ')}: ${typeof v==='number'?fmt(v,3):v}`).join(' · ');
  text('extra',extra);
}
const SIZING_PHASES={'relief check':'Relief check on every detection','planted rocks':'Sizing the planted rocks','detections':'Sizing every detection'};
const HEIGHT_TOP=1.5;
const human=v=>v===null||v===undefined?'--':String(v).replaceAll('_',' ');
const rampColor=b=>{const [r,g,b2]=heat(.28+.72*Math.min(1,Math.max(0,b)/HEIGHT_TOP));return `rgb(${r},${g},${b2})`;};
let sizingPoints=[];
function sizingColumns(s){return Object.fromEntries((s.columns||[]).map((k,i)=>[k,i]));}
function distanceTo(s,row,col){return s.touchdown?Math.hypot(row-s.touchdown[0],col-s.touchdown[1])*s.pixel_m:null;}
function currentCell(s){const a=state.snapshot?.kind==='adaptive'?state.snapshot.adaptive:null;return a?.centre||(s.current?[s.current.row_px,s.current.col_px]:null);}
function drawSizingMap(s){
  const canvas=$('sizing-map'),ctx=canvas.getContext('2d'),[H,W]=s.image_shape,col=sizingColumns(s);
  const scale=Math.min(canvas.width/W,canvas.height/H),w=W*scale,h=H*scale,ox=(canvas.width-w)/2,oy=(canvas.height-h)/2;
  ctx.fillStyle='#0b1521';ctx.fillRect(0,0,canvas.width,canvas.height);
  if(sourceImage){ctx.globalAlpha=.42;ctx.imageSmoothingEnabled=false;ctx.drawImage(sourceImage,ox,oy,w,h);ctx.globalAlpha=1;}
  sizingPoints=[];
  for(const r of s.casters){
    const x=ox+(r[col.col_px]+.5)*scale,y=oy+(r[col.row_px]+.5)*scale,b=r[col.height_lower_bound_m];
    ctx.beginPath();
    if(Number.isFinite(b)){ctx.arc(x,y,2.4+2.4*Math.min(1,b/HEIGHT_TOP),0,2*Math.PI);ctx.fillStyle=rampColor(b);ctx.fill();ctx.lineWidth=1.2;ctx.strokeStyle='#0b1521';ctx.stroke();}
    else{ctx.arc(x,y,2.2,0,2*Math.PI);ctx.lineWidth=1;ctx.strokeStyle='#7d8fa1';ctx.stroke();}
    sizingPoints.push({x,y,r});
  }
  if(s.touchdown){
    const tx=ox+(s.touchdown[1]+.5)*scale,ty=oy+(s.touchdown[0]+.5)*scale;
    ctx.setLineDash([5,4]);ctx.lineWidth=1.3;ctx.strokeStyle='rgba(255,255,255,.8)';ctx.beginPath();ctx.arc(tx,ty,20/s.pixel_m*scale,0,2*Math.PI);ctx.stroke();ctx.setLineDash([]);
    ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(tx-7,ty);ctx.lineTo(tx+7,ty);ctx.moveTo(tx,ty-7);ctx.lineTo(tx,ty+7);ctx.strokeStyle='white';ctx.stroke();
  }
  const cur=currentCell(s);
  if(cur){ctx.lineWidth=2;ctx.strokeStyle='#69dfd0';ctx.beginPath();ctx.arc(ox+(cur[1]+.5)*scale,oy+(cur[0]+.5)*scale,9,0,2*Math.PI);ctx.stroke();}
  const lx=ox+12,ly=oy+h-26,lw=150;
  for(let i=0;i<lw;i++){ctx.fillStyle=rampColor(i/(lw-1)*HEIGHT_TOP);ctx.fillRect(lx+i,ly,1,8);}
  ctx.fillStyle='#c5d7e6';ctx.font='11px system-ui';ctx.textAlign='left';ctx.fillText('0 m',lx,ly+21);ctx.textAlign='right';ctx.fillText(`${HEIGHT_TOP} m+ lower bound`,lx+lw+64,ly+21);
}
function sizingTip(event){
  const s=state?.snapshot?.sizing,tip=$('sizing-tip');if(!s){tip.hidden=true;return;}
  const canvas=$('sizing-map'),rect=canvas.getBoundingClientRect(),mx=(event.clientX-rect.left)*canvas.width/rect.width,my=(event.clientY-rect.top)*canvas.height/rect.height;
  let best=null,bestD=10;for(const p of sizingPoints){const d=Math.hypot(p.x-mx,p.y-my);if(d<bestD){best=p;bestD=d;}}
  if(!best){tip.hidden=true;return;}
  const col=sizingColumns(s),r=best.r,b=r[col.height_lower_bound_m],hgt=r[col.height_m],dist=r[col.distance_to_touchdown_m];
  tip.textContent=`Row ${r[col.row_px]}, column ${r[col.col_px]}${Number.isFinite(dist)?` · ${fmt(dist,1)} m from the touchdown`:''}\n`+
    (Number.isFinite(b)?`Height lower bound ${fmt(b,2)} m${r[col.censored]?' (shadow leaves the window)':''}`:'No warning evidence at any scale')+
    `\nHeight ${Number.isFinite(hgt)?fmt(hgt,2)+' m':'not resolved'} · score ${fmt(r[col.score],1)}\nState: ${human(r[col.state])} · relief check: ${human(r[col.relief_check])}`;
  tip.hidden=false;const x=(event.clientX-rect.left)+14,y=(event.clientY-rect.top)+14;
  tip.style.left=Math.min(x,rect.width-270)+'px';tip.style.top=Math.min(y,rect.height-90)+'px';
}
function drawSizingHistogram(s){
  const canvas=$('sizing-hist'),ctx=canvas.getContext('2d'),col=sizingColumns(s);
  const bounds=s.casters.map(r=>r[col.height_lower_bound_m]).filter(Number.isFinite);
  ctx.fillStyle='#0b1521';ctx.fillRect(0,0,canvas.width,canvas.height);
  if(!bounds.length){ctx.fillStyle='#91a6ba';ctx.font='12px system-ui';ctx.textAlign='center';ctx.fillText('No height bounds yet',canvas.width/2,canvas.height/2);return;}
  const top=Math.max(HEIGHT_TOP,Math.ceil(Math.max(...bounds)*10)/10),bins=Math.round(top/.1),counts=new Array(bins).fill(0);
  for(const b of bounds)counts[Math.min(bins-1,Math.floor(b/.1))]++;
  const left=34,right=12,bottom=24,topPad=12,pw=canvas.width-left-right,ph=canvas.height-bottom-topPad,max=Math.max(...counts),bw=pw/bins;
  ctx.strokeStyle='#293a4d';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,topPad+ph+.5);ctx.lineTo(left+pw,topPad+ph+.5);ctx.stroke();
  counts.forEach((n,i)=>{if(!n)return;const bh=n/max*ph,x=left+i*bw+1,y=topPad+ph-bh;ctx.fillStyle=rampColor((i+.5)*.1);ctx.beginPath();ctx.roundRect(x,y,Math.max(1,bw-2),bh,[3,3,0,0]);ctx.fill();});
  ctx.fillStyle='#91a6ba';ctx.font='10px system-ui';ctx.textAlign='center';
  for(let v=0;v<=top+1e-9;v+=.3)ctx.fillText(v.toFixed(1),left+v/.1*bw,canvas.height-8);
  ctx.textAlign='right';ctx.fillText(String(max),left-6,topPad+8);ctx.fillText('0',left-6,topPad+ph);
  const cx=left+s.clearance_m/.1*bw;ctx.setLineDash([4,3]);ctx.strokeStyle='rgba(255,255,255,.7)';ctx.beginPath();ctx.moveTo(cx,topPad);ctx.lineTo(cx,topPad+ph);ctx.stroke();ctx.setLineDash([]);
  ctx.textAlign='left';ctx.fillStyle='#c5d7e6';ctx.fillText(`${s.clearance_m} m`,cx+4,topPad+9);
}
function renderSizing(){
  const s=state.snapshot?.sizing;$('sizing-panel').hidden=!s;if(!s)return;
  text('sizing-title',SIZING_PHASES[s.phase]||human(s.phase));
  $('sizing-progress').max=s.total||1;$('sizing-progress').value=s.done||0;
  const elapsed=(Date.parse(s.updated)-Date.parse(s.started))/1000,rate=elapsed>=10&&s.done>=3?s.done/elapsed:0;
  const left=rate>0?(s.total-s.done)/rate:null,eta=left===null?'':left<90?` · about ${Math.round(left)} s left`:` · about ${Math.round(left/60)} min left`;
  text('sizing-count',`${s.done.toLocaleString()} / ${s.total.toLocaleString()} cells${rate>0?` · ${(rate*60).toFixed(rate*60<10?1:0)} per min`:''}${s.done<s.total?eta:' · done'}`);
  const cur=currentCell(s),a=state.snapshot?.kind==='adaptive'?state.snapshot.adaptive:null;
  let now='';
  if(cur){const d=distanceTo(s,cur[0],cur[1]);now=`${a?'Now measuring':'Last cell'}: row ${cur[0]}, column ${cur[1]}${d!==null?`, ${fmt(d,1)} m from the touchdown`:''}.`;
    if(a)now+=` Pass ${a.scale}×, fitting radius ${a.support_px} px, ${human(a.status)}. Height compatibility ${a.height_range_m?.map(v=>fmt(v,2)).join('–')??'not yet'} m.`;
    else if(s.current?.relief_check&&s.phase==='relief check')now+=` Relief check: ${human(s.current.relief_check)}.`;}
  text('sizing-now',now||'Waiting for the first cell.');
  drawSizingMap(s);drawSizingHistogram(s);
  const counts=$('sizing-counts');counts.replaceChildren();
  const add=(k,v)=>{const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=k;dd.textContent=v;counts.append(dt,dd);};
  const c=s.counts;
  if(s.phase!=='relief check'){add('Cells measured',`${c.sized.toLocaleString()} of ${s.total.toLocaleString()}`);add('With warning evidence',c.with_warning_evidence.toLocaleString());
    add('Height estimated',c.context_supported.toLocaleString());add(`Lower bound ≥ ${s.clearance_m} m`,c.exceeding_clearance.toLocaleString());}
  const rel=s.relief||{},checked=Object.values(rel).reduce((m,v)=>m+v,0);
  if(checked)add('Relief check',`${rel.rock_like||0} rock-like · ${rel.ambiguous||0} ambiguous · ${rel.relief_like||0} relief-like · ${rel.none||0} none`);
  const col=sizingColumns(s),body=$('sizing-nearest');body.replaceChildren();
  const near=s.casters.filter(r=>Number.isFinite(r[col.distance_to_touchdown_m])).sort((x,y)=>x[col.distance_to_touchdown_m]-y[col.distance_to_touchdown_m]).slice(0,8);
  for(const r of near){const tr=document.createElement('tr');for(const v of [fmt(r[col.distance_to_touchdown_m],1)+' m',Number.isFinite(r[col.height_lower_bound_m])?fmt(r[col.height_lower_bound_m],2)+' m':'--',Number.isFinite(r[col.height_m])?fmt(r[col.height_m],2)+' m':'--',human(r[col.state]),human(r[col.relief_check])]){const td=document.createElement('td');td.textContent=v;tr.append(td);}body.append(tr);}
  if(!near.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=5;td.textContent=s.phase==='relief check'?'Measurements start after the relief check.':'No measured cells yet.';tr.append(td);body.append(tr);}
}
function renderProducts(){
  const select=$('products'), previous=select.value, items=state.artifacts||[];
  const ids=JSON.stringify(items.map(a=>a.path));
  if(select.dataset.ids!==ids){select.replaceChildren(...items.map(a=>{const o=document.createElement('option');o.value=a.path;o.textContent=a.path.replace('stages/','');return o;}));select.dataset.ids=ids;select.value=items.some(a=>a.path===previous)?previous:(items.find(a=>a.path.endsWith('/three_maps.png'))?.path||items[0]?.path||'');}
  const p=items.find(a=>a.path===select.value), key=p?`${p.path}:${p.version}`:'';
  $('product-empty').hidden=!!p;$('product').hidden=!p;
  if(p&&productKey!==key){productKey=key;const url='/artifact/'+p.path.split('/').map(encodeURIComponent).join('/')+'?v='+p.version;$('product').src=url;$('product-link').href=url;}
  text('verdict',state.verdict?`Execution: ${state.verdict.execution} · Scientific verdict: ${state.verdict.scientific_verdict}\n${(state.verdict.findings||[]).join('\n')}`:'Verdict is assembled after the campaign finishes.');
}
function render(){
  text('run-name',state.run_name);text('activity',state.snapshot?.message||state.stages.find(r=>r.id===state.selected_stage)?.title||'Waiting for the campaign to start');
  const label={live:'Live',quiet:'Running · quiet step',stale:'No recent heartbeat',finished:'Run finished',interrupted:'Run interrupted',no_heartbeat:'Saved output / no heartbeat'}[state.health]||state.health;
  text('health',label);$('health').className='badge '+state.health;
  const inferred=state.heartbeat_source==='inferred',age=state.heartbeat_age_seconds;
  text('heartbeat',age===null?'No heartbeat from this runner':inferred?`Last activity ${age<120?Math.round(age)+'s':Math.round(age/60)+' min'} ago · stages started directly`:`Last heartbeat ${Math.round(age)}s ago`);
  const notice=state.health==='quiet'?`No new snapshot for ${Math.round(age/60)} min. Long steps, such as the shape-from-shading solve, report only when they finish.`:
    state.health==='stale'?(inferred?'No stage has written a snapshot or log line for 15 minutes. The displayed results are saved snapshots; the process may have stopped.':'The runner has stopped sending heartbeats. The displayed results are saved snapshots; its process may have stopped or lost access to the output folder.'):
    state.health==='no_heartbeat'?'This run has no live heartbeat. Saved images and logs are available; live calculation snapshots require the updated runner.':state.input_error?`Input preview unavailable: ${state.input_error}`:'';
  $('notice').hidden=!notice;text('notice',notice);
  renderStages();renderFrames();renderTerrain();renderCalculation();renderSizing();renderMaps();renderProducts();
  const log=$('log'), atBottom=log.scrollHeight-log.scrollTop-log.clientHeight<40;log.textContent=state.log||'Waiting for output…';if(atBottom)log.scrollTop=log.scrollHeight;
}
let fetching=false;
async function poll(){if(fetching)return;fetching=true;try{const res=await fetch('/api/state'+(selectedStage?'?stage='+encodeURIComponent(selectedStage):''),{cache:'no-store'});if(!res.ok)throw new Error('Viewer server unavailable');state=await res.json();render();}catch(error){text('health','Viewer disconnected');$('health').className='badge offline';text('notice','The viewer cannot reach its local server. The calculation runs independently; the last displayed values are frozen.');$('notice').hidden=false;}finally{fetching=false;}}
$('frames').onchange=()=>{selectedFrame=Number($('frames').value);renderFrames();renderCalculation();};
$('play').onclick=()=>{playing=!playing;$('play').setAttribute('aria-pressed',String(playing));text('play',playing?'Pause frames':'Play frames');};
$('follow').onclick=()=>{selectedStage=null;updateFollow();poll();};
$('products').onchange=renderProducts;
$('sizing-map').onmousemove=sizingTip;$('sizing-map').onmouseleave=()=>{$('sizing-tip').hidden=true;};
setInterval(()=>{if(playing&&state?.inputs?.frames.length){selectedFrame=(selectedFrame+1)%state.inputs.frames.length;renderFrames();renderCalculation();}},1000);
setInterval(poll,2000);poll();
// Browsers throttle timers in background tabs; catch up as soon as the tab is shown.
document.addEventListener('visibilitychange',()=>{if(!document.hidden)poll();});
