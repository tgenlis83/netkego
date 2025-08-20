import React from 'react';

// Parses /api/run-output to display a per-epoch tqdm-like progress bar without raw text
export default function TrainingProgress(){
  const [epoch,setEpoch]=React.useState(0);
  const [totalEpochs,setTotalEpochs]=React.useState(null);
  const [percent,setPercent]=React.useState(0);
  const [elapsedSec, setElapsedSec] = React.useState(0);
  const [etaSec, setEtaSec] = React.useState(null);
  const [avgEpochSec, setAvgEpochSec] = React.useState(null);
  const [phase, setPhase] = React.useState('');
  const [lr, setLr] = React.useState(null);
  const startRef = React.useRef(null);
  const trainStartRef = React.useRef(null);
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
  if (!trainStartRef.current && cur>=1){ trainStartRef.current = startRef.current; }
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
  // phase: look for " - train" or " - val" descriptors in tqdm labels
  const phaseMatch = s.match(/-\s*(train|val)/i);
  if(phaseMatch){ setPhase(phaseMatch[1].toLowerCase()); }
    // LR: prefer per-epoch from METRIC line: "lr=..."; fall back to recent LR: ... if present
    const metricLR = [...s.matchAll(/\blr=([0-9.eE+-]+)/g)];
    if(metricLR.length){
      const m = metricLR[metricLR.length-1];
      const v = parseFloat(m[1]);
      if(Number.isFinite(v)) setLr(v);
    } else {
      const lrLines = [...s.matchAll(/LR:\s*([0-9.eE+-]+)/g)];
      if(lrLines.length){
        const m = lrLines[lrLines.length-1];
        const v = parseFloat(m[1]);
        if(Number.isFinite(v)) setLr(v);
      }
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
    // METRIC line has avg epoch time: avg_epoch_time_sec=...
    const avgMatch = [...s.matchAll(/avg_epoch_time_sec=([^\s]+)/g)];
    if(avgMatch.length){
      const a = parseFloat(avgMatch[avgMatch.length-1][1]);
      if(Number.isFinite(a)) setAvgEpochSec(a);
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
  // Compute full training ETA: current-epoch ETA + remaining whole epochs * avg epoch time
  let fullEta = null;
  if(Number.isFinite(etaSec) && Number.isFinite(totalEpochs) && Number.isFinite(epoch)){
    const remWhole = Math.max(0, (totalEpochs||0) - (epoch||0));
    const avg = Number.isFinite(avgEpochSec) ? avgEpochSec : etaSec;
    fullEta = etaSec + remWhole * avg;
  } else if (trainStartRef.current && totalEpochs && epoch>=1){
    // fallback: progress-based
    const elapsedTotal = (Date.now() - trainStartRef.current)/1000;
    const progress = Math.max(0, Math.min(1, ((epoch-1) + (pct/100)) / Math.max(1, totalEpochs)));
    if(progress>0){ fullEta = elapsedTotal * (1-progress) / progress; }
  }
  // Overall training progress percent across all epochs
  let overallPct = null;
  if (Number.isFinite(totalEpochs) && totalEpochs>0){
    const doneEpochs = Math.max(0, (epoch||0) - 1);
    const prog = Math.max(0, Math.min(1, (doneEpochs + (pct/100)) / totalEpochs));
    overallPct = Math.round(prog * 100);
  }
  return (
    <div className="space-y-1">
      {/* Full training progress */}
      {Number.isFinite(overallPct) && (
        <div className="space-y-1">
          <div className="text-[11px] text-neutral-700 flex items-center justify-between">
            <span>Training overall</span>
            <span>{overallPct}% {Number.isFinite(fullEta) ? `• ETA ${fmtSeconds(fullEta)}` : ''}</span>
          </div>
          <div className="w-full h-1.5 bg-neutral-200 rounded-md overflow-hidden">
            <div className="h-full bg-emerald-600" style={{ width: overallPct+"%" }} />
          </div>
        </div>
      )}
      <div className="text-xs text-neutral-700 mt-1">
        {totalEpochs? `Epoch ${epoch}/${totalEpochs}` : (epoch>0? `Epoch ${epoch}`: 'Training progress')}
        {phase && <span> • {phase}</span>}
        {Number.isFinite(lr) && <span> • LR {lr}</span>}
      </div>
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
