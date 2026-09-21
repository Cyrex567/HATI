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
  if(fit){cross(fit.root_row_px,fit.root_col_px,'#69dfd0',10);const sz=(fit.observed?.[0]?.length||25)*scale;ctx.strokeStyle='#69dfd0';ctx.strokeRect(ox+(fit.col_px+.5)*scale-sz/2,oy+(fit.row_px+.5)*scale-sz/2,sz,sz);}
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
  text('calculation-title',fit?`Last regional fit · ${s.subrun}`:s?.subrun||s?.message||'Waiting for a calculation');
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
  text('index-formula',`index = score / (score + ${fit?.score_scale??'scale'})`);
  const metrics=$('metrics');metrics.replaceChildren();
  if(fit){for(const [label,value] of [['Bank height',fmt(fit.height_m,2)+' m'],['Bank width',fmt(fit.width_m,2)+' m'],['Contrast',fmt(fit.contrast)],['Common support',fmt(fit.common_fraction*100,1)+'%'],['Identifiability',fmt(fit.identifiability)],['Eligible frames',String(fit.frames.length)],['Endpoint',fit.endpoint_censored?'Censored':'Inside support'],['Null energy',fmt(fit.null_energy,1)]]){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;metrics.append(dt,dd);}}
  if(fit && s.geometry_order?.some((v,i)=>v!==i)){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent='Stress model Sun';dd.textContent=`${fmt(s.model_azimuths[selectedFrame],2)}° az / ${fmt(s.model_elevations[selectedFrame],3)}° el`;metrics.append(dt,dd);}
  const bars=$('frame-bars');bars.replaceChildren();
  if(fit){const max=Math.max(1,...fit.frame_delta_chi2.map(Math.abs));fit.frames.forEach((f,i)=>{const row=document.createElement('div');row.className='frame-row';const label=document.createElement('span'),track=document.createElement('div'),bar=document.createElement('div'),value=document.createElement('span');label.textContent=`Frame ${f+1}`;track.className='track';bar.className='bar'+(fit.frame_delta_chi2[i]<0?' negative':'');bar.style.width=(Math.abs(fit.frame_delta_chi2[i])/max*100)+'%';value.textContent=fmt(fit.frame_delta_chi2[i],2);track.append(bar);row.append(label,track,value);bars.append(row);});}
}

function renderMaps(){
  const s=state.snapshot;
  text('map-title',s?.kind==='field'?s.field_title:s?.kind==='regional'?'Regional search':'Most recent regional scan');
  if(s?.kind==='field'){drawGrid('map-score',s.field,{hi:s.field_title.includes('index')||s.field_title.includes('illumination')?1:null});empty($('map-support'),'Coverage shown during regional search');text('score-caption',s.field_title);}
  else{drawGrid('map-score',s?.score);drawGrid('map-support',s?.assessment,{mode:'status',hi:3});text('score-caption','Raw shadow score · colour scale spans this snapshot');}
  const total=s?.cells_total,visited=s?.cells_visited;
  $('cells').max=total||1;$('cells').value=visited||0;
  text('cell-count',total?`${visited.toLocaleString()} / ${total.toLocaleString()} cells visited`:'Waiting for regional scan');
  text('map-caption','Grey: unvisited/border · Green: assessed · Amber: unavailable · Pink: nonidentifiable. Partial maps are not final results.'+(s?.regional_updated&&s.kind!=='regional'?` Scan snapshot: ${new Date(s.regional_updated).toLocaleTimeString()}.`:''));
  let extra='';
  if(s?.kind==='controls'&&s.control){const c=s.control;extra=`Synthetic control: ${c.kind} · height ${c.height_m} m · location ${c.location+1} · seed ${c.seed}\nAssessment: ${c.status} · maximum score ${fmt(c.maximum_score)} · recovered: ${c.recovered===null?'unknown':String(c.recovered)}`;}
  if(s?.kind==='height'&&s.height_profile){const h=s.height_profile;extra=`Height profile at row ${h.row_px}, column ${h.col_px} · support ${h.support_px} pixels · ${h.status}\nHeights (m): ${h.heights_m.join(', ')}\nScores: ${h.scores.map(v=>fmt(v,2)).join(', ')}\nDescriptive compatibility set: ${h.delta_set_m.join(', ')} m. This is not a calibrated confidence interval.`;}
  if(s?.kind==='registration'&&s.registration){const r=s.registration;extra=`Local registration: ${r.tile_px}-pixel tiles · ${r.pairs} similar-illumination pairs · ${r.measured_tiles} measured tiles. Offsets do not modify the input images.`;}
  if(s?.kind==='field'&&s.metrics)extra=Object.entries(s.metrics).map(([k,v])=>`${k.replaceAll('_',' ')}: ${typeof v==='number'?fmt(v,3):v}`).join(' · ');
  text('extra',extra);
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
  const label={live:'Live',stale:'No recent heartbeat',finished:'Run finished',interrupted:'Run interrupted',no_heartbeat:'Saved output / no heartbeat'}[state.health]||state.health;
  text('health',label);$('health').className='badge '+state.health;
  text('heartbeat',state.heartbeat_age_seconds===null?'No heartbeat from this runner':`Last heartbeat ${Math.round(state.heartbeat_age_seconds)}s ago`);
  const notice=state.health==='stale'?'The runner has stopped sending heartbeats. The displayed results are saved snapshots; its process may have stopped or lost access to the output folder.':state.health==='no_heartbeat'?'This run has no live heartbeat. Saved images and logs are available; live calculation snapshots require the updated runner.':state.input_error?`Input preview unavailable: ${state.input_error}`:'';
  $('notice').hidden=!notice;text('notice',notice);
  renderStages();renderFrames();renderTerrain();renderCalculation();renderMaps();renderProducts();
  const log=$('log'), atBottom=log.scrollHeight-log.scrollTop-log.clientHeight<40;log.textContent=state.log||'Waiting for output…';if(atBottom)log.scrollTop=log.scrollHeight;
}
let fetching=false;
async function poll(){if(fetching)return;fetching=true;try{const res=await fetch('/api/state'+(selectedStage?'?stage='+encodeURIComponent(selectedStage):''),{cache:'no-store'});if(!res.ok)throw new Error('Viewer server unavailable');state=await res.json();render();}catch(error){text('health','Viewer disconnected');$('health').className='badge offline';text('notice','The viewer cannot reach its local server. The calculation runs independently; the last displayed values are frozen.');$('notice').hidden=false;}finally{fetching=false;}}
$('frames').onchange=()=>{selectedFrame=Number($('frames').value);renderFrames();renderCalculation();};
$('play').onclick=()=>{playing=!playing;$('play').setAttribute('aria-pressed',String(playing));text('play',playing?'Pause frames':'Play frames');};
$('follow').onclick=()=>{selectedStage=null;updateFollow();poll();};
$('products').onchange=renderProducts;
setInterval(()=>{if(playing&&state?.inputs?.frames.length){selectedFrame=(selectedFrame+1)%state.inputs.frames.length;renderFrames();renderCalculation();}},1000);
setInterval(poll,2000);poll();
