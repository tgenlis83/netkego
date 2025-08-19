import React from 'react';
import { LAYERS } from '@/lib/constants';
import ColorChip from './ColorChip';
import LayerToken from './LayerToken';

export default function BlockLayersPreview({ composition, className='' }){
  const items = composition || [];
  return (
    <div className={`rounded-md border border-neutral-200 bg-white/70 p-2 ${className}`}>
      <div className="flex flex-wrap gap-1.5 items-center">
        {items.map((id, idx)=>{
          const l = LAYERS.find(x=>x.id===id);
          return (
            <div key={idx} className="flex items-center gap-1">
              <LayerToken id={id} size="md" title={l?.name || id} showHelper={true} />
            </div>
          );
        })}
        {items.length===0 && (
          <div className="text-[11px] text-neutral-500">No layers</div>
        )}
      </div>
    </div>
  );
}
