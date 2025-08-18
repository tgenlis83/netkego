import React from 'react'

export function Switch({ checked = false, onCheckedChange, className = '' }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onCheckedChange && onCheckedChange(!checked)}
  className={`w-10 h-6 rounded-full transition-colors ${checked ? 'bg-blue-600' : 'bg-neutral-300'} focus:outline-none focus:ring-2 focus:ring-blue-300 ${className}`}
    >
  <span className={`block w-5 h-5 bg-white rounded-full shadow transition-transform translate-y-0.5 ${checked ? 'translate-x-5' : 'translate-x-0.5'}`}></span>
    </button>
  )
}
