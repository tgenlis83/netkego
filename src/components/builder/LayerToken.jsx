import React from 'react';
import { LAYERS } from '@/lib/constants';

const CAT_BG = {
  Convolution: 'bg-blue-500',
  Normalization: 'bg-violet-500',
  Activation: 'bg-amber-600',
  Pooling: 'bg-teal-600',
  Attention: 'bg-rose-600',
  Residual: 'bg-slate-500',
  Regularization: 'bg-emerald-600',
  Linear: 'bg-fuchsia-600',
  Meta: 'bg-neutral-500',
};

const ABBR = {
  conv: 'C', pwconv: 'PW', dwconv: 'DW', grpconv: 'GC', dilconv: 'DI', deform: 'DF',
  bn: 'BN', gn: 'GN', ln: 'LN',
  relu: 'RL', gelu: 'GE', silu: 'SI', hswish: 'HS', prelu: 'PR',
  maxpool: 'MP', avgpool: 'AP', gap: 'GAP',
  se: 'SE', eca: 'ECA', cbam: 'CB', mhsa: 'SA', winattn: 'WA',
  add: 'R+', concat: 'CT',
  linear: 'FC',
  dropout: 'DO', droppath: 'DP',
};

function baseAbbr(id){
  return ABBR[id] || (id?.slice(0,3) || '').toUpperCase();
}

function capLen(str, max){
  if(!str) return '';
  return str.length <= max ? str : str.slice(0, max);
}

function summarizeParams(id, cfg = {}, maxLen = 6){
  // Build a compact param string, trimmed to fit inside the token
  const parts = [];
  const push = (label, val, cond = true)=>{ if(cond && (val!==undefined && val!==null)){ parts.push(`${label}${val}`); } };
  if (id === 'conv' || id === 'grpconv' || id === 'dilconv' || id === 'deform' || id === 'dwconv' || id === 'pwconv'){
    // Priorities: C, k, s, g, d (omit defaults that equal 1)
    push('C', cfg.outC);
    push('k', cfg.k);
    push('s', cfg.s, cfg.s && cfg.s !== 1);
    push('g', cfg.g, cfg.g && cfg.g !== 1);
    push('d', cfg.d, cfg.d && cfg.d !== 1);
  } else if (id === 'maxpool' || id === 'avgpool'){
    push('k', cfg.k ?? 2);
    push('s', cfg.s ?? 2, (cfg.s ?? 2) !== 1);
  } else if (id === 'gn'){
    push('g', cfg.groups ?? 32);
  } else if (id === 'se'){
    push('r', cfg.r ?? 16);
  } else if (id === 'linear'){
    push('F', cfg.outF);
  } else if (id === 'dropout' || id === 'droppath'){
    const p = typeof cfg.p === 'number' ? cfg.p : (id==='dropout' ? 0.5 : 0.1);
    const str = p < 0.1 ? p.toFixed(2) : p.toFixed(1);
    push('p', str);
  }
  // Join and clip to maxLen
  let out = '';
  for(const part of parts){
    const next = out ? out + part : part;
    if(next.length <= maxLen){ out = next; } else { break; }
  }
  return out;
}

export default function LayerToken({ id, cfg, size='md', title, showHelper=false }){
  const l = LAYERS.find(x=>x.id===id);
  const bg = CAT_BG[l?.category] || CAT_BG.Meta;
  const effectiveCfg = cfg && Object.keys(cfg).length ? cfg : (l?.defaults || {});
  // Rectangular width only for extended types (conv family, linear); keep others stacked even with tips
  const isExtended = id==='conv' || id==='pwconv' || id==='dwconv' || id==='grpconv' || id==='dilconv' || id==='deform' || id==='linear' || id==='dp' || id==='do';
  const squareW = size==='sm' ? 'w-6' : 'w-7';
  const height = size==='sm' ? 'h-6' : 'h-7';
  const topFs = size==='sm' ? 'text-[10px]' : 'text-[11px]';
  const botFs = size==='sm' ? 'text-[8px]' : 'text-[9px]';
  const base = baseAbbr(id);
  const maxParamLen = (showHelper && isExtended) ? (size==='sm' ? 64 : 128) : (size==='sm' ? 6 : 7);
  const params = showHelper ? summarizeParams(id, effectiveCfg, maxParamLen) : '';
  const hasParams = Boolean(params);
  // Only extended tokens (conv family, linear) should grow horizontally; others stay square
  const widthClass = (showHelper && hasParams && isExtended) ? 'px-2' : squareW;
  return (
  <span className={`inline-flex items-center justify-center ${widthClass} ${height} rounded-md ${bg} text-white font-semibold`} title={title || l?.name || id}>
      {(hasParams && isExtended) ? (
  <span className={`leading-none ${topFs} px-1 text-center`}>{base} - {params}</span>
      ) : hasParams ? (
        <span className="h-full w-full flex flex-col items-center justify-center leading-tight text-center">
          <div className={`${topFs}`}>{base}</div>
          <div className={`${botFs} opacity-90`}>{params}</div>
        </span>
      ) : (
        <span className={`${topFs}`}>{base}</span>
      )}
    </span>
  );
}
