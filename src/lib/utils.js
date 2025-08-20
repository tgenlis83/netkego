// Utility helpers used across components

export function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

export function copyText(text){
  try { navigator.clipboard?.writeText(text); } catch { /* noop */ }
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
  // Use hex escape for ESC (\x1b) in regex source to satisfy no-control-regex
  const ESC = '\x1b';
  // Escape '[' to avoid creating a character class in RegExp patterns
  t = t
    .replaceAll(new RegExp(ESC + "\\[31m", 'g'), '<span style="color:#f87171">')
    .replaceAll(new RegExp(ESC + "\\[32m", 'g'), '<span style="color:#34d399">')
    .replaceAll(new RegExp(ESC + "\\[33m", 'g'), '<span style="color:#fbbf24">')
    .replaceAll(new RegExp(ESC + "\\[0m", 'g'), '</span>');
  return t;
}

export const fmt = (x)=> (x===undefined || Number.isNaN(x))? '-' : Number(x).toFixed(4);
