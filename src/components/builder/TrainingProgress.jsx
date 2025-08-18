import React from 'react';

// Parses /api/run-output to display a per-epoch tqdm-like progress bar without raw text
export default function TrainingProgress(){
  const [epoch,setEpoch]=React.useState(0);
  const [totalEpochs,setTotalEpochs]=React.useState(null);
  const [percent,setPercent]=React.useState(0);
  const [elapsedSec, setElapsedSec] = React.useState(0);
  const [etaSec, setEtaSec] = React.useState(null);
  const startRef = React.useRef(null);
  const lastEpochRef = React.useRef(0);

  React.useEffect(()=>{
    let alive=true;
    const tick=()=>{
      fetch('/api/run-output', { cache: 'no-store' })
        .then(r=>r.text())
        .then(t=>{ if(!alive) return; parse(t); })
        .catch(()=>{});
    };
    tick();
    const id=setInterval(tick, 800);
    return ()=>{ alive=false; clearInterval(id) };
  },[]);

  function parse(t){
    const s=String(t||'');
    // EPOCH line examples: "EPOCH: 3/50" or "EPOCH: 3 /50" or "EPOCH 3 / 50"
    const epochLines = [...s.matchAll(/EPOCH[^0-9]*(\d+)[^0-9]+(\d+)/g)];
    if(epochLines.length){
      const m = epochLines[epochLines.length-1];
      const cur = parseInt(m[1],10);
      const tot = parseInt(m[2],10);
      setEpoch(cur);
      setTotalEpochs(tot);
      // new epoch detected -> reset timers
      if(cur !== lastEpochRef.current){
        lastEpochRef.current = cur;
        startRef.current = Date.now();
        setElapsedSec(0);
        setEtaSec(null);
        setPercent(0);
      }
    }
    // tqdm lines: " 42%|█████...| 123/290 [..]"
    const m2 = s.match(/(\d+)%\|/g);
    if(m2 && m2.length){
      const last = m2[m2.length-1];
      const p = parseInt(last,10);
      if(!Number.isNaN(p)) setPercent(p);
    }
    // tqdm timing chunk, e.g. "[00:08<00:12, 100.00it/s]"
    const timeMatches = [...s.matchAll(/\[(\d{2}:\d{2})(?:\.\d+)?<(\d{2}:\d{2})(?:\.\d+)?/g)];
    if(timeMatches.length){
      const mm = timeMatches[timeMatches.length-1];
      const e = toSeconds(mm[1]);
      const r = toSeconds(mm[2]);
      if(Number.isFinite(e)) setElapsedSec(e);
      if(Number.isFinite(r)) setEtaSec(r);
    } else if(startRef.current){
      // fallback: estimate from percent
      const e = (Date.now() - startRef.current)/1000;
      setElapsedSec(e);
      if((percent||0) > 0){
        const eta = e * (100/(percent||1) - 1);
        setEtaSec(eta);
      }
    }
  }

  function toSeconds(mmss){
    const parts = (mmss||'').split(':').map(x=>parseInt(x,10));
    if(parts.length===2){ return parts[0]*60 + parts[1]; }
    if(parts.length===3){ return parts[0]*3600 + parts[1]*60 + parts[2]; }
    return NaN;
  }
  function fmtSeconds(s){
    if(s==null || !Number.isFinite(s)) return '--:--';
    const sec = Math.max(0, Math.floor(s));
    const h = Math.floor(sec/3600);
    const m = Math.floor((sec%3600)/60);
    const ss = sec%60;
    const mm = String(m).padStart(2,'0');
    const s2 = String(ss).padStart(2,'0');
    return h>0 ? `${h}:${mm}:${s2}` : `${mm}:${s2}`;
  }

  const pct = Math.min(100, Math.max(0, percent||0));
  return (
    <div className="space-y-1">
      <div className="text-xs text-neutral-700">{totalEpochs? `Epoch ${epoch}/${totalEpochs}` : (epoch>0? `Epoch ${epoch}`: 'Training progress')}</div>
      <div className="w-full h-2 bg-neutral-200 rounded-md overflow-hidden">
        <div className="h-full bg-blue-600" style={{ width: pct+'%' }} />
      </div>
      <div className="text-[11px] text-neutral-600 flex items-center justify-between">
        <span>{pct}%</span>
        <span>elapsed {fmtSeconds(elapsedSec)} • ETA {fmtSeconds(etaSec)}</span>
      </div>
    </div>
  );
}
