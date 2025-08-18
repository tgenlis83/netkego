// Utility helpers used across components

export function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

export function copyText(text){
  try { navigator.clipboard?.writeText(text); } catch(_) {}
}

export function downloadText(filename, text){
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

export function ansiToHtml(s){
  const esc = (x)=> String(x ?? '')
    .replaceAll(/&/g,'&amp;')
    .replaceAll(/</g,'&lt;')
    .replaceAll(/>/g,'&gt;');
  let t = esc(s);
  t = t
    .replaceAll(/\x1b\[31m/g, '<span style="color:#f87171">')
    .replaceAll(/\x1b\[32m/g, '<span style="color:#34d399">')
    .replaceAll(/\x1b\[33m/g, '<span style="color:#fbbf24">')
    .replaceAll(/\x1b\[0m/g, '</span>');
  return t;
}

export const fmt = (x)=> (x===undefined || Number.isNaN(x))? '-' : Number(x).toFixed(4);
