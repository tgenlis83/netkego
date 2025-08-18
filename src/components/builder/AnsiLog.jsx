import React from 'react';
import { ansiToHtml } from '@/lib/utils';

export default function AnsiLog({ url }){
  const [text,setText]=React.useState('');
  React.useEffect(()=>{
    let alive=true;
    const tick=()=>{
      fetch(url, { cache: 'no-store' })
        .then(r=>r.text())
        .then(t=>{ if(alive) setText(t) })
        .catch(()=>{})
    };
    tick();
    const id = setInterval(tick, 1000);
    return ()=>{ alive=false; clearInterval(id) };
  },[url]);
  return (
    <pre className="h-[40vh] overflow-auto rounded-xl border border-neutral-200 bg-black text-white p-2 text-xs">
      <code dangerouslySetInnerHTML={{ __html: ansiToHtml(text) }} />
    </pre>
  );
}
