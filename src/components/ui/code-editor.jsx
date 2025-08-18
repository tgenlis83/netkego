import React, { useMemo, useRef, useEffect } from 'react'
import Prism from 'prismjs'
import 'prismjs/components/prism-python'
import 'prismjs/themes/prism.css'

export function CodeEditor({ value, onChange, language = 'python', className = '' }) {
  const preRef = useRef(null)

  const highlighted = useMemo(() => {
    try {
      const gram = Prism.languages[language] || Prism.languages.markup
      return Prism.highlight(value ?? '', gram, language)
    } catch {
      return (value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
    }
  }, [value, language])

  const taRef = useRef(null)

  // keep highlight layer aligned with textarea scroll
  const syncScroll = () => {
    const ta = taRef.current
    const pre = preRef.current
    if (!ta || !pre) return
    const x = -ta.scrollLeft
    const y = -ta.scrollTop
    pre.style.transform = `translate(${x}px, ${y}px)`
  }

  useEffect(() => { syncScroll() }, [value])

  return (
    <div className={`relative font-mono text-xs rounded-xl border border-neutral-200 bg-white/90 shadow-inner overflow-hidden ${className}`}>
      {/* Highlight layer (moves via transform to mirror textarea scroll) */}
      <pre
        ref={preRef}
        className="absolute top-0 left-0 right-0 m-0 p-2 whitespace-pre-wrap break-words pointer-events-none select-none"
        aria-hidden
      >
        <code dangerouslySetInnerHTML={{ __html: highlighted }} />
      </pre>
      {/* Input layer (single scrollbar) */}
      <textarea
        ref={taRef}
        className="absolute inset-0 w-full h-full resize-none bg-transparent text-transparent caret-black p-2 outline-none overflow-auto whitespace-pre-wrap break-words"
        spellCheck={false}
        value={value}
        onChange={(e) => onChange && onChange(e.target.value)}
        onScroll={syncScroll}
      />
    </div>
  )
}
