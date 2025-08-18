import React from 'react'

export function Input({ className = '', ...props }) {
  return (
    <input
  className={`w-full rounded-md border border-neutral-200 bg-white/95 px-2 py-1 text-sm outline-none focus:ring-2 focus:ring-blue-300 focus:border-blue-300 shadow-inner ${className}`}
      {...props}
    />
  )
}
