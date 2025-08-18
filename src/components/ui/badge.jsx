import React from 'react'

export function Badge({ className = '', variant = 'default', children }) {
  const styles = variant === 'secondary'
  ? 'bg-white/80 text-neutral-800 border border-neutral-200 shadow-sm'
  : 'bg-blue-50 text-blue-800 border border-blue-200 shadow-sm'
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded-md ${styles} ${className}`}>{children}</span>
  )
}
