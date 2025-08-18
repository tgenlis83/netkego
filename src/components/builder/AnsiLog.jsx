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
  // Strip tqdm progress lines from display (we render a separate progress bar UI)
  const cleaned = React.useMemo(()=>{
    const s = String(text||'');
    // remove carriage-return based updates and typical tqdm percentage lines
    return s
      .replaceAll(/\r[^\n]*?/g, '')
      .split(/\n/)
      .filter(line => !/(\d+)%\|.*it\/.*/.test(line))
      .join('\n');
  }, [text]);
  return (
    <pre className="h-[40vh] overflow-auto rounded-xl border border-neutral-200 bg-black text-white p-2 text-xs">
      <code dangerouslySetInnerHTML={{ __html: ansiToHtml(cleaned) }} />
    </pre>
  );
}
