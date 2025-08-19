import React, { useMemo, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Plus } from 'lucide-react';
import { LAYERS, CAT_COLORS } from '@/lib/constants';
import ColorChip from './ColorChip';
import LayerToken from './LayerToken';

export default function Palette({ addLayer, mode, compat }){
  const [cat,setCat]=useState('All');
  const [q,setQ]=useState('');
  const list = useMemo(()=> LAYERS.filter(l=> (cat==='All'||l.category===cat) && (q==='' || [l.name,l.role,l.op].join(' ').toLowerCase().includes(q.toLowerCase())) ), [cat,q]);
  return (
    <Card className="flex-1">
      <CardHeader>
        <CardTitle>Layer Palette</CardTitle>
        {mode && <div className="text-xs text-neutral-600 mt-1">Target: <b>{mode === 'model' ? 'Build Model' : 'Build Block'}</b></div>}
        <div className="text-[11px] text-neutral-600 mt-1">Pins: <span className="text-emerald-700">● Fits</span> · <span className="text-amber-700">● Needs config</span> · <span className="text-rose-700">● Mismatch</span></div>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <div className="grid grid-cols-2 gap-2">
          <Select value={cat} onValueChange={setCat}>
            <SelectTrigger className="w-full"><SelectValue/></SelectTrigger>
            <SelectContent>
              {['All',...new Set(LAYERS.map(l=>l.category))].map(c=> <SelectItem key={c} value={c}>{c}</SelectItem>)}
            </SelectContent>
          </Select>
          <Input placeholder="Search layers" value={q} onChange={(e)=>setQ(e.target.value)}/>
        </div>
        <div className="max-h-[50vh] overflow-auto space-y-2 pr-1">
          {list.map(l=> (
            <div key={l.id} className={`border border-neutral-200 rounded-xl p-2 bg-white/90 backdrop-blur-sm flex items-center justify-between shadow-sm hover:shadow-md transition ${CAT_COLORS[l.category]?.ring||''}`}>
              <div className="flex items-center gap-2">
                <LayerToken id={l.id} size="md" />
                <div>
                  <div className="font-medium text-sm flex items-center gap-2">{l.name}<ColorChip category={l.category}/></div>
                <div className="text-[11px] text-neutral-600">{l.role}</div>
                {l.dimReq && <div className="text-[11px] text-neutral-700 mt-0.5">Req: {l.dimReq}</div>}
                {compat && compat[l.id] && (
                  <div className="text-[11px] mt-1 flex items-center gap-2">
                    <span
                      className={`inline-block w-2.5 h-2.5 rounded-full ${compat[l.id].status==='ok' ? 'bg-emerald-600' : compat[l.id].status==='warn' ? 'bg-amber-600' : 'bg-rose-600'}`}
                      title={compat[l.id].reason || ''}
                    />
                    <span className={`${compat[l.id].status==='ok' ? 'text-emerald-700' : compat[l.id].status==='warn' ? 'text-amber-700' : 'text-rose-700'}`}>{compat[l.id].label}</span>
                    {compat[l.id].synergy && (
                      <span className="px-1.5 py-0.5 rounded-md bg-emerald-50 text-emerald-700 border border-emerald-200">{compat[l.id].synergy}</span>
                    )}
                  </div>
                )}
                </div>
              </div>
              <Button size="sm" variant="outline" onClick={()=>addLayer(l.id)}><Plus className="w-4 h-4 mr-1"/>Add</Button>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
