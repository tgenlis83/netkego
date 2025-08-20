import React from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Link2, AlertTriangle, CheckCircle, Save, Copy, ArrowUp, ArrowDown, X } from 'lucide-react'
import LayerToken from '@/components/builder/LayerToken'
// Palette lives on the left column in parent
import ColorChip from '@/components/builder/ColorChip'
import { LAYERS, PRESETS, CAT_COLORS } from '@/lib/constants'
import { renderStepSummary } from '@/lib/builderUtils'

export default function BuildBlocksTab({
  block, stats,
  setSelectedIdx,
  saveName, setSaveName, saveCurrentBlock,
  importPreset, appendPresetToBlock,
  showPresetBlocksBuild, setShowPresetBlocksBuild,
  editingModelBlockIdx, setEditingModelBlockIdx,
  moveIdx, removeIdx, duplicateIdx,
  inspector,
  synergy,
}){
  return (
    <div className="grid grid-cols-5 gap-3">
      <div className="col-span-3">
        {editingModelBlockIdx!=null && (
          <div className="mb-2 p-2 rounded-md bg-blue-50 border border-blue-200 text-blue-800 flex items-center justify-between">
            <span>Editing model block. Changes sync automatically.</span>
            <Button size="sm" variant="secondary" onClick={()=>setEditingModelBlockIdx(null)}>Done</Button>
          </div>
        )}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <CardTitle className="flex items-center gap-2 w-full">
                <span>Current Block</span>
                <span className="ml-auto" />
                <Button size="sm" variant="destructive" onClick={()=>{ if(window.confirm('Clear current block?')){ const evt = new CustomEvent('netkego:clear-block'); window.dispatchEvent(evt); } }}>Clear</Button>
              </CardTitle>
              <div className="ml-auto flex items-center gap-2">
                <Input className="h-8 w-48" placeholder="Name (optional)" value={saveName} onChange={(e)=>setSaveName(e.target.value)} />
                <Button size="sm" onClick={saveCurrentBlock} disabled={block.length===0} title="Save block"><Save className="w-4 h-4 mr-1"/>Save</Button>
              </div>
            </div>
            <div className="text-xs text-neutral-600 mt-1">Use the Layer Palette on the left, or pick a Preset Block below.</div>
          </CardHeader>
          <CardContent className="space-y-2">
            {block.length===0 && <div className="text-neutral-600 text-sm">Add layers from the left palette or import a preset.</div>}
            {block.map((s,i)=>{
              const l = LAYERS.find(x=>x.id===s.id) || { id: s.id, name: s.id, category: 'Meta' };
              const addIssue = stats.issues.find(it => it.step===i && (it.type==="add_invalid_from" || it.type==="add_mismatch"));
              return (
                <div key={i} className={`border border-neutral-200 rounded-xl p-2 bg-white/90 backdrop-blur-sm flex items-center justify-between shadow-sm hover:shadow-md transition ${CAT_COLORS[l.category]?.ring||''}`}>
                  <div className="flex items-start gap-2">
                    <LayerToken id={s.id} cfg={s.cfg} size="md" showHelper={true} />
                    <div>
                      <div className="text-sm font-medium flex items-center gap-2">{l.name}<ColorChip category={l.category||'Meta'}/></div>
                      <div className="text-xs text-neutral-600">{renderStepSummary(l,s)}</div>
                      {l.id==="add" && (
                        <div className="text-[11px] mt-0.5 flex items-center gap-2">
                          <Link2 className="w-3.5 h-3.5"/>
                          {typeof s.cfg?.from === 'number' ? (
                            <>
                              <span>From: #{s.cfg.from} {block[s.cfg.from] ? (LAYERS.find(v=>v.id===block[s.cfg.from].id)?.name || block[s.cfg.from].id) : '?'}</span>
                              {addIssue ? (
                                <span className="inline-flex items-center gap-1 text-rose-700"><AlertTriangle className="w-3.5 h-3.5"/>Mismatch</span>
                              ) : (
                                <span className="inline-flex items-center gap-1 text-emerald-700"><CheckCircle className="w-3.5 h-3.5"/>OK</span>
                              )}
                              <Button size="sm" variant="outline" onClick={()=>setSelectedIdx(s.cfg.from)}>Go</Button>
                            </>
                          ) : (
                            <span className="text-neutral-500">Select a source in the inspector</span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button size="icon" variant="ghost" onClick={()=>duplicateIdx(i)} title="Duplicate"><Copy className="w-4 h-4"/></Button>
                    <Button size="icon" variant="ghost" onClick={()=>moveIdx(i,-1)}><ArrowUp className="w-4 h-4"/></Button>
                    <Button size="icon" variant="ghost" onClick={()=>moveIdx(i,1)}><ArrowDown className="w-4 h-4"/></Button>
                    <Button size="icon" variant="ghost" onClick={()=>{ setSelectedIdx(i); }}>Edit</Button>
                    <Button size="icon" variant="ghost" onClick={()=>removeIdx(i)}><X className="w-4 h-4"/></Button>
                  </div>
                </div>
              )
            })}
          </CardContent>
        </Card>

        <Card className="mt-3">
          <CardHeader><CardTitle>Synergy Tips</CardTitle></CardHeader>
          <CardContent className="flex flex-wrap gap-2 text-xs">
            {synergy}
          </CardContent>
        </Card>
      </div>

      <div className="col-span-2 space-y-3">
        <Card>
          <CardHeader><CardTitle>Inspector & Config</CardTitle></CardHeader>
          <CardContent className="text-sm space-y-3">
            {inspector}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">Preset Blocks
              <Button size="sm" variant="ghost" className="ml-auto" onClick={()=>setShowPresetBlocksBuild(v=>!v)}>{showPresetBlocksBuild? 'Hide':'Show'}</Button>
            </CardTitle>
          </CardHeader>
          {showPresetBlocksBuild && (
            <CardContent className="space-y-2">
              {PRESETS.map(p=> (
                <div key={p.id} className="border border-neutral-200 rounded-md p-2 bg-white/90">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <div className="text-sm font-medium">{p.name}</div>
                      <div className="text-[11px] text-neutral-600">{p.family}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button size="sm" variant="secondary" onClick={()=>importPreset(p)}>Use</Button>
                      <Button size="sm" variant="outline" onClick={()=>appendPresetToBlock(p)}>Append</Button>
                    </div>
                  </div>
                  <div className="mt-2 flex flex-wrap gap-1.5 items-center">
                    {p.composition.map((id, idx)=> (
                      <LayerToken key={idx} id={id} size="md" showHelper={true} />
                    ))}
                  </div>
                </div>
              ))}
            </CardContent>
          )}
        </Card>
      </div>
    </div>
  )
}
