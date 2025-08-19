import React, { useMemo, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Wand2 } from 'lucide-react';
import { PRESETS } from '@/lib/constants';
import BlockLayersPreview from './BlockLayersPreview';
import LayerToken from './LayerToken';

export default function PresetBlocksPanel({ onUse, onAppend }){
  const [q,setQ]=useState('');
  const list = useMemo(()=> PRESETS.filter(p=> [p.name, p.family, p.composition.join(' ')].join(' ').toLowerCase().includes(q.toLowerCase())), [q]);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Preset Blocks</CardTitle>
        <div className="text-xs text-neutral-600 mt-1">Search and add a preset block to the current block. You can edit layers after adding.</div>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <Input placeholder="Search preset blocks" value={q} onChange={e=>setQ(e.target.value)} />
        <div className="space-y-2 max-h-[38vh] overflow-auto pr-1">
          {list.map(p=> (
            <div key={p.id} className="border border-neutral-200 rounded-md p-2 bg-white/90">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="text-sm font-medium">{p.name}</div>
                  <div className="text-[11px] text-neutral-600">{p.family}</div>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <Button size="sm" variant="secondary" onClick={()=>onUse(p)}><Wand2 className="w-4 h-4 mr-1"/>Use</Button>
                  <Button size="sm" variant="outline" onClick={()=>onAppend(p)}>Append</Button>
                </div>
              </div>
              <div className="mt-2 flex flex-wrap gap-1.5 items-center">
                {p.composition.map((id, i)=> (
                  <LayerToken key={i} id={id} size="md" showHelper={true} />
                ))}
              </div>
            </div>
          ))}
          {list.length===0 && (
            <div className="text-xs text-neutral-600">No results.</div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
