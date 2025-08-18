import React, { useState, useRef, useEffect, useContext, createContext } from 'react'

const SelectContext = createContext(null)

export function Select({ value, onValueChange, children, className = '' }) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef(null)

  useEffect(() => {
    function onDocClick(e) {
      if (!rootRef.current) return
      if (!rootRef.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [])

  const ctx = { value, onValueChange, open, setOpen }
  return (
  <div ref={rootRef} className={`relative inline-block w-full ${className}`}>
      <SelectContext.Provider value={ctx}>{children}</SelectContext.Provider>
    </div>
  )
}

export function SelectTrigger({ className = '', children, disabled = false }) {
  const { open, setOpen } = useContext(SelectContext) || {}
  return (
    <button
      type="button"
      disabled={disabled}
  className={`border border-neutral-200 rounded-md px-2 py-1 w-full flex items-center justify-between bg-white hover:bg-neutral-50 shadow-sm transition ${className}`}
      onClick={() => !disabled && setOpen && setOpen((v) => !v)}
    >
      <span className="truncate">{children}</span>
      <span className={`ml-2 text-neutral-500 transition-transform ${open ? 'rotate-180' : ''}`}>▾</span>
    </button>
  )
}

export function SelectValue({ placeholder }) {
  const { value } = useContext(SelectContext) || {}
  if (!value) return <span className="text-neutral-500">{placeholder}</span>
  return <span>{value}</span>
}

export function SelectContent({ children, className = '' }) {
  const { open } = useContext(SelectContext) || {}
  if (!open) return null
  return (
    <div className={`absolute left-0 right-0 mt-1 border border-neutral-200 rounded-md bg-white/95 backdrop-blur-sm shadow-lg p-1 max-h-48 overflow-auto z-20 ${className}`}>
      {children}
    </div>
  )
}

export function SelectItem({ value, children, className = '' }) {
  const { onValueChange, setOpen } = useContext(SelectContext) || {}
  const handleClick = () => {
    onValueChange && onValueChange(value)
    setOpen && setOpen(false)
  }
  return (
    <div
      className={`px-2 py-1 rounded hover:bg-neutral-100 cursor-pointer text-sm ${className}`}
      onClick={handleClick}
      role="option"
      aria-selected={false}
    >
      {children}
    </div>
  )
}
