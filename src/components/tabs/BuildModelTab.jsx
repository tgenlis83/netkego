import React from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Library, Blocks, Boxes, Box, ArrowUp, ArrowDown, X, Copy, CheckCircle, AlertTriangle } from 'lucide-react'
import LayerToken from '@/components/builder/LayerToken'
import ColorChip from '@/components/builder/ColorChip'
import { LAYERS, PRESETS, MODEL_PRESETS, CAT_COLORS } from '@/lib/constants'
import { renderStepSummary } from '@/lib/builderUtils'

export default function BuildModelTab({
  // state
  model, setModelSelIdx,
  // stats/context
  modelStats, H, W, Cin, hp,
  // dataset/memory helpers
  inputMode, datasetId, datasetPct, estimateMemoryMB, formatMem,
  // actions
  duplicateModelIdx, moveModelIdx, removeModelIdx, editBlockInBuilder,
  addSavedBlockToModel, addPresetToModel, applyModelPreset, appendModelPreset,
  // saved & presets toggles
  savedBlocks, showSaved, setShowSaved,
  showPresetBlocks, setShowPresetBlocks,
  showPresetModels, setShowPresetModels,
  // flattened graph/meta
  flattenedSteps, modelIdxForFlattened,
  // inspector node
  inspector,
}){
  // Detect invalid residual adds (no valid source index)
  const invalidAddModelIdx = React.useMemo(()=>{
    const bad = new Set();
    flattenedSteps.forEach((s, gIdx)=>{
      if (s.id === 'add'){
        const from = s?.cfg?.from;
        const valid = (typeof from === 'number') && from >= 0 && from < gIdx;
        if (!valid){
          const mi = modelIdxForFlattened(model, gIdx);
          if (mi >= 0) bad.add(mi);
        }
      }
    });
    return bad;
  }, [flattenedSteps, model]);
  return (
    <div className="grid grid-cols-5 gap-3">
      <div className="col-span-3">
        <Card className="mb-3">
          <CardHeader><CardTitle>Inspector & Config</CardTitle></CardHeader>
          <CardContent className="text-sm">
            {inspector}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 w-full">
              <span>Current Model</span>
              <span className="ml-auto" />
              <Button size="sm" variant="destructive" onClick={()=>{ if(window.confirm('Clear entire model?')){ const evt = new CustomEvent('netkego:clear-model'); window.dispatchEvent(evt); } }}>Clear</Button>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="border border-dashed border-neutral-300 rounded-xl p-2 bg-white/60 text-xs text-neutral-700">Input • ({Cin}, {H}, {W}) — mandatory</div>
            {model.length===0 && <div className="text-neutral-600 text-sm">Add saved blocks, presets, or layers from the right panel.</div>}
            {model.map((el,i)=>{
              if(el.type==='layer'){
                const l = LAYERS.find(x=>x.id===el.id) || { id: el.id, name: el.id, category: 'Meta' };
                const invalidAdd = (el.id==='add') && invalidAddModelIdx.has(i);
                return (
                  <div key={`L${i}`} className={`border rounded-xl p-2 backdrop-blur-sm flex items-center justify-between shadow-sm hover:shadow-md transition ${invalidAdd ? 'bg-rose-50 border-rose-300' : 'bg-white/90 border-neutral-200'} ${CAT_COLORS[l.category]?.ring||''}`}>
                    <div className="flex items-start gap-2">
                      <LayerToken id={el.id} cfg={el.cfg} size="md" showHelper={true} />
                      <div>
                        <div className="text-sm font-medium flex items-center gap-2">{l.name}<ColorChip category={l.category||'Meta'}/></div>
                        <div className="text-xs text-neutral-600">{renderStepSummary(l, el)}</div>
                        {invalidAdd && (
                          <div className="text-[11px] text-rose-700 mt-0.5">No residual source selected. Pick a valid previous layer in the inspector.</div>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <Button size="icon" variant="ghost" onClick={()=>duplicateModelIdx(i)} title="Duplicate"><Copy className="w-4 h-4"/></Button>
                      <Button size="icon" variant="ghost" onClick={()=>moveModelIdx(i,-1)}><ArrowUp className="w-4 h-4"/></Button>
                      <Button size="icon" variant="ghost" onClick={()=>moveModelIdx(i,1)}><ArrowDown className="w-4 h-4"/></Button>
                      <Button size="icon" variant="ghost" onClick={()=>{ setModelSelIdx(i); }}>Edit</Button>
                      <Button size="icon" variant="ghost" onClick={()=>removeModelIdx(i)}><X className="w-4 h-4"/></Button>
                    </div>
                  </div>
                );
              }
              // block element
              return (
                <div key={`B${i}`} className={`border border-neutral-200 rounded-xl p-2 bg-neutral-100 backdrop-blur-sm flex items-center justify-between shadow-sm hover:shadow-md transition`}>
                  <div className="min-w-0">
                    <div className="text-sm font-medium flex items-center gap-2"><Box className="w-4 h-4"/>{el.name}<span className="px-2 py-0.5 rounded-md text-[11px] bg-neutral-200 text-neutral-800">Block</span></div>
                    <div className="mt-1 flex flex-wrap gap-1.5 items-center">
                      {el.steps.map((s, idx)=> (
                        <LayerToken key={idx} id={s.id} cfg={s.cfg} size="md" showHelper={true} />
                      ))}
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button size="icon" variant="ghost" onClick={()=>duplicateModelIdx(i)} title="Duplicate"><Copy className="w-4 h-4"/></Button>
                    <Button size="icon" variant="ghost" onClick={()=>moveModelIdx(i,-1)}><ArrowUp className="w-4 h-4"/></Button>
                    <Button size="icon" variant="ghost" onClick={()=>moveModelIdx(i,1)}><ArrowDown className="w-4 h-4"/></Button>
                    <Button size="icon" variant="ghost" onClick={()=>editBlockInBuilder(i)}>Edit</Button>
                    <Button size="icon" variant="ghost" onClick={()=>removeModelIdx(i)}><X className="w-4 h-4"/></Button>
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>

        <Card className="mt-3">
          <CardHeader><CardTitle>Rough Totals</CardTitle></CardHeader>
          <CardContent className="text-xs space-y-1">
            <div>Output shape: ({modelStats.outC}, {modelStats.H}, {modelStats.W})</div>
            <div>Params: {(modelStats.params/1e6).toFixed(3)} M (rough)</div>
            <div>FLOPs: {(modelStats.flops/1e9).toFixed(3)} GFLOPs @ {H}×{W}</div>
            <div>Est. memory (batch {hp.batchSize}): {formatMem(estimateMemoryMB(
              modelStats,
              hp.batchSize,
              hp.precision,
              hp.optimizer,
              { Cin, H, W, datasetId, datasetPct, valSplit: hp.valSplit, numWorkers: hp.numWorkers, ema: hp.ema, inputMode }
            ))}</div>
            <div>Issues: {modelStats.issues.length===0 ? <span className="inline-flex items-center gap-1 text-emerald-700"><CheckCircle className="w-3.5 h-3.5"/>None</span> : <span className="inline-flex items-center gap-1 text-rose-700"><AlertTriangle className="w-3.5 h-3.5"/>{modelStats.issues.length}</span>}</div>
          </CardContent>
        </Card>

        <Card className="mt-3">
          <CardHeader><CardTitle>Model Dimension Checker</CardTitle></CardHeader>
          <CardContent className="space-y-2 text-xs">
            {modelStats.issues.length===0 && (
              <div className="text-emerald-700 flex items-center gap-1"><CheckCircle className="w-3.5 h-3.5"/>No dimensional conflicts detected.</div>
            )}
            {modelStats.issues.map((it, idx)=> (
              <div key={idx} className="border border-neutral-200 rounded-md p-2 bg-white/90 backdrop-blur-sm shadow-sm">
                <div className="font-medium flex items-center gap-2">
                  {(it.type==="add_mismatch" || it.type==="add_invalid_from") ? <AlertTriangle className="w-3.5 h-3.5 text-rose-700"/> : null}
                  Step #{it.step}: {it.msg}
                </div>
                {it.type==="add_mismatch" && it.ref && (
                  <div className="mt-1 text-[11px] text-neutral-700">Suggestion: insert a <b>1×1 Conv</b> (projection) or adjust <b>stride</b>/padding so both paths yield the same (C,H,W).</div>
                )}
                {it.type==="linear_spatial" && (
                  <div className="mt-1 text-[11px] text-neutral-700">Tip: add a <b>Global Avg Pool</b> before Linear, or rely on generator safety (auto-GAP) now enabled.</div>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      <div className="col-span-2">
        <Card>
          <CardHeader><CardTitle>Model Builder</CardTitle></CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="text-xs text-neutral-600">Add blocks or layers. Input is implicit from the left settings.</div>
            <div className="border rounded-xl p-2">
              <div className="font-medium text-sm mb-1 flex items-center gap-2"><Library className="w-4 h-4"/>Saved Blocks<Button size="sm" variant="ghost" className="ml-auto" onClick={()=>setShowSaved(v=>!v)}>{showSaved? 'Hide':'Show'}</Button></div>
              {!showSaved ? null : savedBlocks.length===0 ? (
                <div className="text-xs text-neutral-600">Save blocks from the Build Block tab to appear here.</div>
              ) : (
                <div className="space-y-2 max-h-[24vh] overflow-auto pr-1">
                  {savedBlocks.map(sb=> (
                    <div key={sb.name} className="border border-neutral-200 rounded-md p-2 bg-white/90 flex items-center justify-between">
                      <div>
                        <div className="text-sm font-medium">{sb.name}</div>
                        <div className="text-[11px] text-neutral-600 truncate">{sb.steps.map(s=>LAYERS.find(x=>x.id===s.id)?.name||s.id).join(' → ')}</div>
                      </div>
                      <Button size="sm" variant="outline" onClick={()=>addSavedBlockToModel(sb.name)}>Add</Button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="border rounded-xl p-2">
              <div className="font-medium text-sm mb-1 flex items-center gap-2"><Blocks className="w-4 h-4"/>Preset Blocks<Button size="sm" variant="ghost" className="ml-auto" onClick={()=>setShowPresetBlocks(v=>!v)}>{showPresetBlocks? 'Hide':'Show'}</Button></div>
              {!showPresetBlocks ? null : (
                <div className="space-y-2 max-h-[24vh] overflow-auto pr-1">
                  {PRESETS.map(p=> (
                    <div key={p.id} className="border border-neutral-200 rounded-md p-2 bg-white/90">
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <div className="text-sm font-medium">{p.name}</div>
                          <div className="text-[11px] text-neutral-600">{p.family}</div>
                        </div>
                        <Button size="sm" variant="outline" onClick={()=>addPresetToModel(p)}>Add</Button>
                      </div>
                      <div className="mt-2 flex flex-wrap gap-1.5 items-center">
                        {p.composition.map((id, idx)=> (
                          <LayerToken key={idx} id={id} size="md" showHelper={true} />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Palette removed here; use the left-side palette instead */}
            <div className="text-xs text-neutral-600 border rounded-xl p-2">
              Use the Layer Palette on the left. It will add to
              {" "}
              <b>Build Model</b>
              {" "}
              depending on the active tab.
            </div>

            <div className="border rounded-xl p-2">
              <div className="font-medium text-sm mb-1 flex items-center gap-2"><Boxes className="w-4 h-4"/>Preset Models<Button size="sm" variant="ghost" className="ml-auto" onClick={()=>setShowPresetModels(v=>!v)}>{showPresetModels? 'Hide':'Show'}</Button></div>
              {!showPresetModels ? null : (
                <div className="space-y-2 max-h-[24vh] overflow-auto pr-1">
                  {MODEL_PRESETS.map(mp=> (
                    <div key={mp.id} className="border border-neutral-200 rounded-md p-2 bg-white/90 flex items-center justify-between">
                      <div>
                        <div className="text-sm font-medium">{mp.name}</div>
                        <div className="text-[11px] text-neutral-600 truncate">{mp.family} — {mp.description}</div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button size="sm" variant="secondary" onClick={()=>applyModelPreset(mp)}>Use</Button>
                        <Button size="sm" variant="outline" onClick={()=>appendModelPreset(mp)}>Append</Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Graph view */}
            <Card className="mt-3">
              <CardHeader><CardTitle>Model Graph</CardTitle></CardHeader>
              <CardContent>
                {(() => {
                  const N = model.length;
                  const rowH = 48; const width = 420; const height = Math.max(60, N*rowH + 20);
                  const nodes = model.map((el, i)=>({ i, y: 20 + i*rowH + 12, label: el.type==='block'? el.name : (LAYERS.find(l=>l.id===el.id)?.name || el.id), type: el.type }));
                  const adds = flattenedSteps.map((s, g)=>({s,g})).filter(o=> o.s.id==='add' && typeof o.s.cfg?.from==='number');
                  const edges = adds.map(({s,g})=> ({ from: modelIdxForFlattened(model, s.cfg.from), to: modelIdxForFlattened(model, g) })).filter(e=> e.from>=0 && e.to>=0);
                  return (
                    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="bg-white/70 rounded-md border">
                      <defs>
                        <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto" markerUnits="strokeWidth">
                          <path d="M0,0 L0,6 L6,3 z" fill="#64748b" />
                        </marker>
                      </defs>
                      {nodes.slice(0,-1).map((n,i)=> (
                        <line key={`seq-${i}`} x1={80} y1={n.y} x2={80} y2={nodes[i+1].y} stroke="#cbd5e1" strokeWidth="2" />
                      ))}
                      {nodes.map(n=> {
                        const isInvalidAdd = model[n.i]?.type==='layer' && model[n.i]?.id==='add' && invalidAddModelIdx.has(n.i);
                        return (
                          <g key={`n-${n.i}`}>
                            <circle cx={80} cy={n.y} r={8} fill={n.type==='block'? '#2563eb':'#10b981'} />
                            {isInvalidAdd && (
                              <g stroke="#ef4444" strokeWidth="2">
                                <line x1={74} y1={n.y-6} x2={86} y2={n.y+6} />
                                <line x1={86} y1={n.y-6} x2={74} y2={n.y+6} />
                              </g>
                            )}
                            <text x={96} y={n.y+4} fontSize="12" fill="#334155">{n.label}</text>
                          </g>
                        );
                      })}
                      {edges.map((e,idx)=>{
                        const aY = nodes[e.from]?.y; const bY = nodes[e.to]?.y;
                        if(aY==null || bY==null) return null;
                        const x1 = 80, y1 = aY, x2 = 80, y2 = bY;
                        const dx = 120 + (idx%3)*20;
                        const path = `M ${x1} ${y1} C ${x1+dx} ${y1}, ${x2+dx} ${y2}, ${x2} ${y2}`;
                        return <path key={`res-${idx}`} d={path} fill="none" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow)"/>;
                      })}
                    </svg>
                  );
                })()}
                <div className="text-[11px] text-neutral-600 mt-1">Blocks are blue nodes; layers are green nodes. Curved lines show residual adds (source → add location).</div>
              </CardContent>
            </Card>
          </CardContent>
        </Card>

        {/* Inspector moved to top */}
      </div>
    </div>
  )
}

// local helper removed (no longer needed)
