/* eslint react-refresh/only-export-components: off */
import React from 'react';
import { fmt } from '@/lib/utils';

export function MetricsViewer({ showLR=false }){
  const [text,setText]=React.useState('');
  React.useEffect(()=>{
    let alive=true;
    const tick=()=>{ fetch('/api/run-output', { cache: 'no-store' }).then(r=>r.text()).then(t=>{ if(alive) setText(t) }).catch(()=>{}) };
    tick(); const id=setInterval(tick, 1200); return ()=>{ alive=false; clearInterval(id) };
  },[]);
  const metrics=parseMetrics(text);
  return (
    <div>
      <div className="grid grid-cols-2 gap-3">
        <LossesChart title="Loss (Train/Val)" trainData={metrics.train_loss||[]} valData={metrics.val_loss||[]} trainColor="#2563eb" valColor="#dc2626" />
        <MiniChart title="Val Acc" data={metrics.val_acc||[]} color="#16a34a"/>
        {showLR && <MiniChart title="Learning Rate" data={metrics.lr||[]} color="#8b5cf6" yLabel="lr"/>}
        <MiniChart title="Avg Epoch Time (s)" data={metrics.avg_epoch_time_sec||[]} color="#0ea5e9" yLabel="sec"/>
      </div>
    </div>
  );
}

export function MetricsSummary(){
  const [text,setText]=React.useState('');
  React.useEffect(()=>{
    let alive=true;
    const tick=()=>{ fetch('/api/run-output', { cache: 'no-store' }).then(r=>r.text()).then(t=>{ if(alive) setText(t) }).catch(()=>{}) };
    tick(); const id=setInterval(tick, 1500); return ()=>{ alive=false; clearInterval(id) };
  },[]);
  const m=parseMetrics(text);
  const last=(k)=> (m[k] && m[k].length>0) ? m[k][m[k].length-1].y : undefined;
  return (
    <div className="text-sm space-y-1">
      <div>Best Val Acc: {fmt(last('best_val_acc'))}</div>
      <div>Final Test Acc: {fmt(last('test_acc'))}</div>
      <div>Final Test Loss: {fmt(last('test_loss'))}</div>
      <div>Avg Epoch Time: {fmt(last('avg_epoch_time_sec'))} s</div>
    </div>
  );
}

export function parseMetrics(text){
  const lines=String(text||'').split(/\n/);
  const push=(obj,key,x,epoch)=>{ if(!obj[key]) obj[key]=[]; obj[key].push({ x:epoch, y:parseFloat(x) }); };
  const out={};
  lines.forEach(line=>{
  const epm=line.match(/METRIC:\s*epoch=(\d+)/);
  const ep = epm ? parseInt(epm[1],10) : NaN;
    const base=line.match(/METRIC:\s*epoch=\d+\s+train_loss=([0-9.]+)\s+val_loss=([0-9.]+)\s+val_acc=([0-9.]+)/);
    if(base){ push(out,'train_loss',base[1],ep); push(out,'val_loss',base[2],ep); push(out,'val_acc',base[3],ep); }
  const ext=line.match(/epoch_time_sec=([^\s]+)\s+avg_epoch_time_sec=([^\s]+)\s+gpu_mem_mb=([^\s]+)\s+rss_mem_mb=([^\s]+)/);
    if(ext){ push(out,'epoch_time_sec',ext[1],ep); push(out,'avg_epoch_time_sec',ext[2],ep); push(out,'gpu_mem_mb',ext[3],ep); push(out,'rss_mem_mb',ext[4],ep); }
  const b=line.match(/BEST:\s*val_acc=([0-9.]+)/); if(b){ push(out,'best_val_acc',b[1], NaN); }
  // LR from METRIC line (preferred)
  const lra=line.match(/METRIC:\s*epoch=(\d+).*?\blr=([0-9.eE+-]+)/);
  if(lra){ const e=parseInt(lra[1],10); push(out,'lr', lra[2], e); }
  // legacy fallback: step-level prints "LR: ..."
  const lr=line.match(/LR:\s*([0-9.eE+-]+)/); if(lr && ep && !lra){ push(out,'lr', lr[1], ep); }
    const t=line.match(/TEST:\s*acc=([0-9.]+)\s+loss=([0-9.]+)/); if(t){ push(out,'test_acc',t[1], NaN); push(out,'test_loss',t[2], NaN); }
  });
  return out;
}

export function MiniChart({ title, data, color, xLabel='epoch', yLabel }){
  const w=280, h=140;
  const padL=36, padR=8, padT=8, padB=22;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const xs = data.map(p=>p.x).filter(x=>Number.isFinite(x));
  const ys = data.map(p=>p.y).filter(y=>Number.isFinite(y));
  const xmin = xs.length? Math.min(...xs) : 0; const xmax= xs.length? Math.max(...xs):1;
  let ymin = ys.length? Math.min(...ys) : 0; let ymax= ys.length? Math.max(...ys):1;
  if(ymin===ymax){ ymax = ymin + (ymin===0 ? 1 : Math.abs(ymin)*0.1); }
  const scaleX=(x)=> padL + plotW * (xs.length? (x - xmin) / Math.max(1e-6, (xmax - xmin)) : 0);
  const scaleY=(y)=> padT + plotH - plotH * (ys.length? (y - ymin) / Math.max(1e-6, (ymax - ymin)) : 0);
  const path = data.filter(p=>Number.isFinite(p.x) && Number.isFinite(p.y)).map((p,i)=> (i? 'L':'M') + scaleX(p.x) + ',' + scaleY(p.y)).join(' ');
  const isAcc = title.toLowerCase().includes('acc');
  const best = ys.length ? (isAcc ? Math.max(...ys) : Math.min(...ys)) : undefined;
  const bestPoint = (()=>{
    if(!Number.isFinite(best)) return null;
    let idx=-1;
    for(let i=0;i<data.length;i++){ const p=data[i]; if(Number.isFinite(p.y) && p.y===best){ idx=i; break; } }
    return idx>=0 ? data[idx] : null;
  })();
  const xTicks = 4; const yTicks = 4;
  const xTickVals = Array.from({length: xTicks+1}, (_,i)=> xmin + (xmax - xmin) * (xTicks? i/xTicks : 0));
  const yTickVals = Array.from({length: yTicks+1}, (_,i)=> ymin + (ymax - ymin) * (yTicks? i/yTicks : 0));
  return (
    <div className="border rounded-md p-2 bg-white">
      <div className="text-xs mb-1 flex items-center justify-between">
        <span>{title}</span>
        {Number.isFinite(best) && (
          <span className="text-[11px] text-neutral-700">{isAcc? 'max' : 'min'}: {fmt(best)}</span>
        )}
      </div>
      <svg width={w} height={h}>
        {/* gridlines */}
        {yTickVals.map((v,i)=> (
          <line key={`y${i}`} x1={padL} y1={scaleY(v)} x2={w-padR} y2={scaleY(v)} stroke="#f3f4f6" strokeWidth="1" />
        ))}
        {xTickVals.map((v,i)=> (
          <line key={`x${i}`} x1={scaleX(v)} y1={padT} x2={scaleX(v)} y2={h-padB} stroke="#f3f4f6" strokeWidth="1" />
        ))}
        {/* axes */}
        <line x1={padL} y1={h-padB} x2={w-padR} y2={h-padB} stroke="#e5e7eb" strokeWidth="1"/>
        <line x1={padL} y1={padT} x2={padL} y2={h-padB} stroke="#e5e7eb" strokeWidth="1"/>
        {/* data */}
        <path d={path} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round"/>
        {bestPoint && (
          <g>
            <circle cx={scaleX(bestPoint.x)} cy={scaleY(bestPoint.y)} r={3} fill={color} />
          </g>
        )}
        {/* tick labels */}
        {yTickVals.map((v,i)=> {
          const label = (yLabel==='lr' || title.toLowerCase().includes('learning rate'))
            ? (Number.isFinite(v)? v.toExponential(1) : '')
            : fmt(v);
          return (
            <text key={`yl${i}`} x={padL-6} y={scaleY(v)+3} textAnchor="end" fontSize="10" fill="#6b7280">{label}</text>
          );
        })}
        {xTickVals.map((v,i)=> (
          <text key={`xl${i}`} x={scaleX(v)} y={h-padB+12} textAnchor="middle" fontSize="10" fill="#6b7280">{Number.isFinite(v)? Math.round(v) : ''}</text>
        ))}
        {/* axis labels */}
        <text x={w - padR} y={h - 4} textAnchor="end" fontSize="10" fill="#6b7280">{xLabel}</text>
        <text x={padL+2} y={padT+10} fontSize="10" fill="#6b7280">{yLabel || (isAcc? 'acc' : 'loss')}</text>
      </svg>
    </div>
  );
}

export function ConfusionMatrix(){
  const [data,setData] = React.useState(null);
  React.useEffect(()=>{
    let alive=true;
    const tick=()=>{
      fetch(`/checkpoints/confusion.json?t=${Date.now()}`, { cache:'no-store' })
        .then(r=> r.ok ? r.json() : null)
        .then(j=>{ if(alive && j) setData(j); })
        .catch(()=>{});
    };
    tick(); const id=setInterval(tick, 2000);
    return ()=>{ alive=false; clearInterval(id); };
  },[]);
  const counts = data?.counts; const norm = data?.normalized;
  const n = Array.isArray(norm)? norm.length : 0;
  const size = 240; const pad=30; const cell = n? (size - pad) / n : 0;
  const color = (v)=>{ // v in [0,1]
    const t = Math.max(0, Math.min(1, Number(v)||0));
    const r = Math.round(255 * (1 - t));
    const g = Math.round(255 * (1 - 0.5*t));
    const b = Math.round(255 * (1 - 0.9*t));
    return `rgb(${r},${g},${b})`;
  };
  return (
    <div className="border rounded-md p-2 bg-white">
      <div className="text-xs mb-1 flex items-center justify-between">
        <span>Confusion Matrix</span>
        {!!n && <span className="text-[11px] text-neutral-600">{n} classes</span>}
      </div>
      {!n ? (
        <div className="text-xs text-neutral-600">No confusion matrix yet. It will appear after testing completes.</div>
      ) : (
        <svg width={size} height={size}>
          {/* axes labels */}
          <text x={pad + (size-pad)/2} y={size-4} textAnchor="middle" fontSize="10" fill="#6b7280">Predicted</text>
          <text x={10} y={pad/2} fontSize="10" fill="#6b7280" transform={`rotate(-90 10 ${pad/2})`}>True</text>
          {/* grid */}
          {norm.map((row, i)=> row.map((v,j)=> (
            <g key={`c${i}-${j}`}>
              <rect x={pad + j*cell} y={pad + i*cell} width={cell} height={cell} fill={color(v)} stroke="#e5e7eb" />
              <text x={pad + j*cell + cell/2} y={pad + i*cell + cell/2 + 3} textAnchor="middle" fontSize="9" fill="#111827">{Math.round((v||0)*100)}%</text>
            </g>
          )))}
          {/* ticks */}
          {Array.from({length:n}, (_,i)=> (
            <text key={`xl${i}`} x={pad + i*cell + cell/2} y={pad-4} textAnchor="middle" fontSize="8" fill="#6b7280">{i}</text>
          ))}
          {Array.from({length:n}, (_,i)=> (
            <text key={`yl${i}`} x={pad-10} y={pad + i*cell + cell/2 + 3} textAnchor="end" fontSize="8" fill="#6b7280">{i}</text>
          ))}
        </svg>
      )}
      {counts && (
        <div className="mt-1 text-[11px] text-neutral-600">Counts shown via hover not implemented; numbers reflect row-normalized percentages.</div>
      )}
    </div>
  );
}

function LossesChart({ title, trainData, valData, trainColor, valColor, xLabel='epoch' }){
  const w=280, h=140;
  const padL=36, padR=8, padT=8, padB=22;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const xs = [...(trainData||[]), ...(valData||[])].map(p=>p.x).filter(x=>Number.isFinite(x));
  const ys = [...(trainData||[]), ...(valData||[])].map(p=>p.y).filter(y=>Number.isFinite(y));
  const xmin = xs.length? Math.min(...xs) : 0; const xmax= xs.length? Math.max(...xs):1;
  const ymin = ys.length? Math.min(...ys) : 0; const ymax= ys.length? Math.max(...ys):1;
  const scaleX=(x)=> padL + plotW * (xs.length? (x - xmin) / Math.max(1e-6, (xmax - xmin)) : 0);
  const scaleY=(y)=> padT + plotH - plotH * (ys.length? (y - ymin) / Math.max(1e-6, (ymax - ymin)) : 0);
  const mkPath = (arr)=> (arr||[]).filter(p=>Number.isFinite(p.x) && Number.isFinite(p.y)).map((p,i)=> (i? 'L':'M') + scaleX(p.x) + ',' + scaleY(p.y)).join(' ');
  const pathTrain = mkPath(trainData);
  const pathVal = mkPath(valData);
  const minTrain = (trainData||[]).map(p=>p.y).filter(Number.isFinite).reduce((a,b)=> Math.min(a,b), +Infinity);
  const minVal   = (valData||[]).map(p=>p.y).filter(Number.isFinite).reduce((a,b)=> Math.min(a,b), +Infinity);
  const findPoint = (arr, y)=>{ if(!Number.isFinite(y)) return null; for(const p of (arr||[])){ if(Number.isFinite(p.y) && p.y===y) return p; } return null; };
  const ptMinTrain = findPoint(trainData, minTrain);
  const ptMinVal = findPoint(valData, minVal);
  const xTicks = 4; const yTicks = 4;
  const xTickVals = Array.from({length: xTicks+1}, (_,i)=> xmin + (xmax - xmin) * (xTicks? i/xTicks : 0));
  const yTickVals = Array.from({length: yTicks+1}, (_,i)=> ymin + (ymax - ymin) * (yTicks? i/yTicks : 0));
  return (
    <div className="border rounded-md p-2 bg-white">
      <div className="text-xs mb-1 flex items-center justify-between">
        <span>{title}</span>
        <span className="flex items-center gap-2">
          {Number.isFinite(minTrain) && isFinite(minTrain) && <span className="text-[11px]" style={{color: trainColor}}>train min: {fmt(minTrain)}</span>}
          {Number.isFinite(minVal) && isFinite(minVal) && <span className="text-[11px]" style={{color: valColor}}>val min: {fmt(minVal)}</span>}
        </span>
      </div>
      <svg width={w} height={h}>
        {/* gridlines */}
        {yTickVals.map((v,i)=> (
          <line key={`y${i}`} x1={padL} y1={scaleY(v)} x2={w-padR} y2={scaleY(v)} stroke="#f3f4f6" strokeWidth="1" />
        ))}
        {xTickVals.map((v,i)=> (
          <line key={`x${i}`} x1={scaleX(v)} y1={padT} x2={scaleX(v)} y2={h-padB} stroke="#f3f4f6" strokeWidth="1" />
        ))}
        {/* axes */}
        <line x1={padL} y1={h-padB} x2={w-padR} y2={h-padB} stroke="#e5e7eb" strokeWidth="1"/>
        <line x1={padL} y1={padT} x2={padL} y2={h-padB} stroke="#e5e7eb" strokeWidth="1"/>
        {/* data */}
        <path d={pathTrain} fill="none" stroke={trainColor} strokeWidth="2" strokeLinecap="round"/>
        <path d={pathVal} fill="none" stroke={valColor} strokeWidth="2" strokeLinecap="round"/>
        {ptMinTrain && <circle cx={scaleX(ptMinTrain.x)} cy={scaleY(ptMinTrain.y)} r={3} fill={trainColor} />}
        {ptMinVal && <circle cx={scaleX(ptMinVal.x)} cy={scaleY(ptMinVal.y)} r={3} fill={valColor} />}
        {/* tick labels */}
        {yTickVals.map((v,i)=> (
          <text key={`yl${i}`} x={padL-6} y={scaleY(v)+3} textAnchor="end" fontSize="10" fill="#6b7280">{fmt(v)}</text>
        ))}
        {xTickVals.map((v,i)=> (
          <text key={`xl${i}`} x={scaleX(v)} y={h-padB+12} textAnchor="middle" fontSize="10" fill="#6b7280">{Number.isFinite(v)? Math.round(v) : ''}</text>
        ))}
        {/* axis labels */}
        <text x={w - padR} y={h - 4} textAnchor="end" fontSize="10" fill="#6b7280">{xLabel}</text>
        <text x={padL+2} y={padT+10} fontSize="10" fill="#6b7280">loss</text>
        {/* legend */}
        <g transform={`translate(${w - padR - 90}, ${padT + 8})`}>
          <rect x={0} y={-10} width={90} height={28} rx={4} ry={4} fill="#ffffff" stroke="#e5e7eb" />
          <circle cx={10} cy={0} r={4} fill={trainColor} />
          <text x={20} y={3} fontSize="10" fill="#374151">train</text>
          <circle cx={10} cy={14} r={4} fill={valColor} />
          <text x={20} y={17} fontSize="10" fill="#374151">val</text>
        </g>
      </svg>
    </div>
  );
}
