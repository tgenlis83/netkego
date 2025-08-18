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

function tokenText(id, cfg = {}){
  const base = ABBR[id] || (id?.slice(0,3) || '').toUpperCase();
  // pick a single key numeric to show (optional)
  if (id === 'conv' || id === 'dwconv' || id === 'dilconv' || id === 'maxpool' || id === 'avgpool'){
    const k = cfg.k || (id.includes('pool') ? 2 : 3);
    return `${base}${k}`;
  }
  if (id === 'se'){
    const r = cfg.r || 16; return `${base}${r}`;
  }
  // keep short for others
  return base;
}

export default function LayerToken({ id, cfg, size='md', title }){
  const l = LAYERS.find(x=>x.id===id);
  const bg = CAT_BG[l?.category] || CAT_BG.Meta;
  const text = tokenText(id, cfg);
  const dim = size==='sm' ? 'w-6 h-6 text-[10px]' : 'w-7 h-7 text-[11px]';
  return (
    <span className={`inline-flex items-center justify-center ${dim} rounded-md ${bg} text-white font-semibold`} title={title || l?.name || id}>
      {text}
    </span>
  );
}
