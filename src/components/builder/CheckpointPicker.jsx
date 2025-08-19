import React from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

export default function CheckpointPicker({ open, onClose, onPick }){
  const [items, setItems] = React.useState([]);
  const [path, setPath] = React.useState('checkpoints/best.pt');
  const [mode, setMode] = React.useState('full'); // 'full' | 'weights'

  React.useEffect(()=>{
    if(!open) return;
    let alive=true;
    const tick=()=>{
      fetch('/api/run-output', { cache: 'no-store' })
        .then(r=>r.text())
        .then(t=>{ if(!alive) return; setItems(parseCkpts(t)); })
        .catch(()=>{});
    };
    tick();
    const id=setInterval(tick, 1200);
    return ()=>{ alive=false; clearInterval(id) };
  }, [open]);

  function parseCkpts(s){
    const out=[];
    const lines = String(s||'').split(/\n/);
    for(const line of lines){
      // CKPT: type=epoch path=checkpoints/epoch_005_val0.7231.pt epoch=5 val_acc=0.7231
      const m = line.match(/^CKPT:\s*type=(\w+)\s+path=(\S+)(?:\s+epoch=(\d+))?(?:\s+val_acc=([0-9.]+))?/);
      if(m){
        out.push({ type:m[1], path:m[2], epoch: m[3]? parseInt(m[3],10): null, val_acc: m[4]? parseFloat(m[4]): null });
      }
    }
    // newest last; sort by epoch if available else stable
    out.sort((a,b)=> (b.epoch||0) - (a.epoch||0));
    return out;
  }

  if(!open) return null;
  return (
    <div className="fixed inset-0 z-50 bg-black/30">
      <div className="absolute inset-0 bg-white flex flex-col p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="text-base font-semibold">Resume from checkpoint</div>
          <Button variant="ghost" onClick={onClose}>Close</Button>
        </div>
        <div className="space-y-3 text-sm flex-1 overflow-auto">
          <div>
            <div className="text-xs mb-1">Known checkpoints (parsed)</div>
            <div className="max-h-[60vh] overflow-auto border rounded-md">
              {items.length===0 ? (
                <div className="p-2 text-xs text-neutral-600">No checkpoints detected yet. You can still enter a path manually.</div>
              ) : (
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-neutral-50 text-neutral-700"><th className="text-left px-2 py-1">File</th><th className="text-left px-2 py-1">Type</th><th className="text-left px-2 py-1">Epoch</th><th className="text-left px-2 py-1">Val acc</th><th></th></tr>
                  </thead>
                  <tbody>
                    {items.map((it,idx)=> (
                      <tr key={idx} className="border-t">
                        <td className="px-2 py-1 font-mono truncate max-w-[240px]" title={it.path}>{it.path}</td>
                        <td className="px-2 py-1">{it.type}</td>
                        <td className="px-2 py-1">{it.epoch??'-'}</td>
                        <td className="px-2 py-1">{it.val_acc!=null? it.val_acc.toFixed(4): '-'}</td>
                        <td className="px-2 py-1 text-right"><Button size="sm" variant="outline" onClick={()=>setPath(it.path)}>Use</Button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
          <div>
            <div className="text-xs mb-1">Checkpoint path</div>
            <Input value={path} onChange={(e)=>setPath(e.target.value)} placeholder="checkpoints/best.pt"/>
            <div className="text[11px] text-neutral-600 mt-1">Enter an absolute or project-relative path. Default save dir is ./checkpoints/.</div>
          </div>
          <div>
            <div className="text-xs mb-1">Load mode</div>
            <Select value={mode} onValueChange={setMode}>
              <SelectTrigger className="w-full"><SelectValue/></SelectTrigger>
              <SelectContent>
                <SelectItem value="full">Full state (weights + optimizer/scheduler + epoch)</SelectItem>
                <SelectItem value="weights">Weights only (fresh optimizer/scheduler)</SelectItem>
              </SelectContent>
            </Select>
          </div>
  </div>
  <div className="mt-3 flex items-center justify-end gap-2">
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button onClick={()=>{ onPick({ path, mode }); onClose(); }}>Resume</Button>
        </div>
      </div>
    </div>
  );
}
