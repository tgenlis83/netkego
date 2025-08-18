import React from 'react';
import { fmt } from '@/lib/utils';

export function MetricsViewer(){
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
        <MiniChart title="Train Loss" data={metrics.train_loss||[]} color="#2563eb"/>
        <MiniChart title="Val Loss" data={metrics.val_loss||[]} color="#dc2626"/>
        <MiniChart title="Val Acc" data={metrics.val_acc||[]} color="#16a34a"/>
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
    </div>
  );
}

export function parseMetrics(text){
  const lines=String(text||'').split(/\n/);
  const push=(obj,key,x,epoch)=>{ if(!obj[key]) obj[key]=[]; obj[key].push({ x:epoch, y:parseFloat(x) }); };
  const out={};
  lines.forEach(line=>{
    const m=line.match(/METRIC:\s*epoch=(\d+)\s+train_loss=([0-9.]+)\s+val_loss=([0-9.]+)\s+val_acc=([0-9.]+)/);
    if(m){ const ep=parseInt(m[1],10); push(out,'train_loss',m[2],ep); push(out,'val_loss',m[3],ep); push(out,'val_acc',m[4],ep); }
    const b=line.match(/BEST:\s*val_acc=([0-9.]+)/); if(b){ push(out,'best_val_acc',b[1], NaN); }
    const t=line.match(/TEST:\s*acc=([0-9.]+)\s+loss=([0-9.]+)/); if(t){ push(out,'test_acc',t[1], NaN); push(out,'test_loss',t[2], NaN); }
  });
  return out;
}

export function MiniChart({ title, data, color, xLabel='epoch', yLabel }){
  const w=280, h=120, pad=8;
  const xs = data.map(p=>p.x).filter(x=>Number.isFinite(x));
  const ys = data.map(p=>p.y).filter(y=>Number.isFinite(y));
  const xmin = xs.length? Math.min(...xs) : 0; const xmax= xs.length? Math.max(...xs):1;
  const ymin = ys.length? Math.min(...ys) : 0; const ymax= ys.length? Math.max(...ys):1;
  const scaleX=(x)=> pad + (w-2*pad) * (xs.length? (x - xmin) / Math.max(1e-6, (xmax - xmin)) : 0);
  const scaleY=(y)=> h - pad - (h-2*pad) * (ys.length? (y - ymin) / Math.max(1e-6, (ymax - ymin)) : 0);
  const path = data.filter(p=>Number.isFinite(p.x) && Number.isFinite(p.y)).map((p,i)=> (i? 'L':'M') + scaleX(p.x) + ',' + scaleY(p.y)).join(' ');
  return (
    <div className="border rounded-md p-2 bg-white">
      <div className="text-xs mb-1">{title}</div>
      <svg width={w} height={h}>
  {/* axes */}
  <line x1={pad} y1={h-pad} x2={w-pad} y2={h-pad} stroke="#e5e7eb" strokeWidth="1"/>
  <line x1={pad} y1={pad} x2={pad} y2={h-pad} stroke="#e5e7eb" strokeWidth="1"/>
        <path d={path} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round"/>
  {/* labels */}
  <text x={w - pad} y={h - 2} textAnchor="end" fontSize="10" fill="#6b7280">{xLabel}</text>
  <text x={pad+2} y={pad+10} fontSize="10" fill="#6b7280">{yLabel || (title.toLowerCase().includes('acc')? 'acc' : 'loss')}</text>
      </svg>
    </div>
  );
}
