import React from "react";
import { Input } from "@/components/ui/input";
import ColorChip from "@/components/builder/ColorChip";
import { LAYERS } from "@/lib/constants";

export default function LayerConfig({ selected, onChange, selectedIdx, block, stats, addContext }){
  const l = LAYERS.find(x=>x.id===selected?.id) || { id: selected?.id, name: selected?.id || 'Layer', category: 'Meta', defaults: {} };
  const cfg = { ...((l && l.defaults) || {}), ...((selected && selected.cfg) || {}) };
  const inS = stats?.inShapes?.[selectedIdx];
  const outS = stats?.outShapes?.[selectedIdx];
  const set = (k,v)=> onChange({ ...cfg, [k]:v });
  // When inspecting model-level Add, compute effective resolved source from flattened steps
  const effectiveFrom = React.useMemo(()=>{
    if(addContext?.scope === 'model' && l.id==='add'){
      const s = block?.[selectedIdx];
      const f = s?.cfg?.from;
      if(typeof f === 'number') return f;
    }
    return null;
  }, [addContext?.scope, l.id, block, selectedIdx]);
  return (
    <div className="space-y-2">
      <div className="text-base font-semibold flex items-center gap-2">{l.name} <ColorChip category={l.category||'Meta'}/></div>
      <div className="border rounded-xl p-2 text-xs bg-white">
        <div className="font-medium mb-1">Dimensions</div>
        <div>Input: ({inS?.C ?? "?"}, {inS?.H ?? "?"}, {inS?.W ?? "?"})</div>
        <div>Output: ({outS?.C ?? "?"}, {outS?.H ?? "?"}, {outS?.W ?? "?"})</div>
      </div>
      {l.category==="Convolution" && l.id!=="dwconv" && (
        <div className="grid grid-cols-2 gap-2">
          <div><div className="text-xs">Out channels</div><Input type="number" value={cfg.outC||64} onChange={e=>set('outC',parseInt(e.target.value||"64",10))}/></div>
          <div><div className="text-xs">Kernel</div><Input type="number" value={cfg.k||3} onChange={e=>set('k',parseInt(e.target.value||"3",10))}/></div>
          <div><div className="text-xs">Stride</div><Input type="number" value={cfg.s||1} onChange={e=>set('s',parseInt(e.target.value||"1",10))}/></div>
          {l.id==="grpconv" && <div><div className="text-xs">Groups</div><Input type="number" value={cfg.g||32} onChange={e=>set('g',parseInt(e.target.value||"32",10))}/></div>}
          {l.id==="dilconv" && <div><div className="text-xs">Dilation</div><Input type="number" value={cfg.d||2} onChange={e=>set('d',parseInt(e.target.value||"2",10))}/></div>}
        </div>
      )}
      {l.id==="dwconv" && (
        <div className="grid grid-cols-2 gap-2">
          <div><div className="text-xs">Kernel</div><Input type="number" value={cfg.k||3} onChange={e=>set('k',parseInt(e.target.value||"3",10))}/></div>
          <div><div className="text-xs">Stride</div><Input type="number" value={cfg.s||1} onChange={e=>set('s',parseInt(e.target.value||"1",10))}/></div>
          <div className="col-span-2 text-[11px] text-neutral-600">Depthwise uses groups=inC and keeps channels.</div>
        </div>
      )}
      {l.id==="pwconv" && (
        <div className="grid grid-cols-2 gap-2">
          <div><div className="text-xs">Out channels</div><Input type="number" value={cfg.outC||64} onChange={e=>set('outC',parseInt(e.target.value||"64",10))}/></div>
        </div>
      )}
      {l.id==="se" && (
        <div className="grid grid-cols-2 gap-2">
          <div><div className="text-xs">Reduction r</div><Input type="number" value={cfg.r||16} onChange={e=>set('r',parseInt(e.target.value||"16",10))}/></div>
        </div>
      )}
      {(l.id==="maxpool"||l.id==="avgpool") ? (
        <div className="grid grid-cols-2 gap-2">
          <div><div className="text-xs">Kernel</div><Input type="number" value={cfg.k||2} onChange={e=>set('k',parseInt(e.target.value||"2",10))}/></div>
          <div><div className="text-xs">Stride</div><Input type="number" value={cfg.s||2} onChange={e=>set('s',parseInt(e.target.value||"2",10))}/></div>
        </div>
      ) : null}
      {l.id==="linear" && (
        <div className="grid grid-cols-2 gap-2">
          <div><div className="text-xs">Out features</div><Input type="number" value={cfg.outF||1000} onChange={e=>set('outF',parseInt(e.target.value||"1000",10))}/></div>
        </div>
      )}
      {(l.id==="dropout" || l.id==="droppath") && (
        <div className="grid grid-cols-2 gap-2">
          <div><div className="text-xs">Drop prob p</div><Input type="number" step="0.01" min="0" max="1" value={cfg.p ?? (l.id==='dropout'?0.5:0.1)} onChange={e=>set('p', Math.max(0, Math.min(1, parseFloat(e.target.value||"0"))) )}/></div>
          <div className="text-[11px] text-neutral-600 col-span-2">
            {l.id==='droppath' ? 'Stochastic Depth randomly drops the residual branch with probability p (train only).' : 'Dropout zeros features with probability p (train only).'}
          </div>
        </div>
      )}
      {l.id==="mhsa" && (
        <div className="grid grid-cols-2 gap-2">
          <div><div className="text-xs">Heads</div><Input type="number" value={cfg.heads ?? 8} onChange={e=>set('heads', Math.max(1, parseInt(e.target.value||"8",10)))} /></div>
          <div><div className="text-xs">Attn Drop</div><Input type="number" step="0.01" min="0" max="1" value={cfg.attnDrop ?? 0.0} onChange={e=>set('attnDrop', Math.max(0, Math.min(1, parseFloat(e.target.value||"0")) ))} /></div>
          <div><div className="text-xs">Proj Drop</div><Input type="number" step="0.01" min="0" max="1" value={cfg.projDrop ?? 0.0} onChange={e=>set('projDrop', Math.max(0, Math.min(1, parseFloat(e.target.value||"0")) ))} /></div>
          <div className="text-[11px] text-neutral-600 col-span-2">Tip: heads should divide channels C; use LayerNorm before attention (pre-norm) like the ViT paper.</div>
        </div>
      )}
      {l.id==="add" && (
        <div className="grid grid-cols-1 gap-2">
          <div>
            <div className="text-xs">Skip from step</div>
            {addContext?.scope === 'model' ? (
              <select
                className="w-full border rounded-md text-sm p-2"
                value={(cfg.fromGlobal!=null) ? String(cfg.fromGlobal) : (effectiveFrom!=null ? String(effectiveFrom) : (cfg.from!=null ? String(cfg.from) : ""))}
                onChange={(e)=>{
                  const g = parseInt(e.target.value,10);
                  onChange({ ...cfg, fromGlobal: g });
                }}
              >
                <option value="" disabled>{"Choose a previous layer from the model"}</option>
                {addContext?.flattenedMeta?.filter(m=> m.g < selectedIdx).map(m=> (
                  <option key={m.g} value={String(m.g)}>
                    #{m.g} {m.blockName ? `[${m.blockName}] `: ''}{m.layerName}
                  </option>
                ))}
              </select>
            ) : (
              <select className="w-full border rounded-md text-sm p-2" value={cfg.from!==null && cfg.from!==undefined ? String(cfg.from) : ""} onChange={(e)=>onChange({ ...cfg, from: parseInt(e.target.value,10) })}>
                <option value="" disabled>{selectedIdx>0?"Choose a previous step":"No previous steps"}</option>
                {block.map((b,idx)=> idx<selectedIdx ? (
                  <option key={idx} value={String(idx)}>#{idx} {LAYERS.find(t=>t.id===b.id)?.name || b.id}</option>
                ) : null)}
              </select>
            )}
            <div className="text-[11px] text-neutral-600 mt-1">Tip: residual requires matching (C,H,W). Choose a source before size-changing ops to form an identity path.</div>
          </div>
        </div>
      )}
      <div className="text-[11px] text-neutral-600">Tip: Use <b>Residual Add</b> around a stack to form a block.</div>
    </div>
  );
}
