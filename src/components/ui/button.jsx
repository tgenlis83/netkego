import React from 'react'

const variants = {
  default: 'bg-blue-600 text-white hover:bg-blue-700 shadow-sm hover:shadow-md',
  outline: 'border border-neutral-200 bg-white hover:bg-neutral-50 shadow-sm hover:shadow',
  secondary: 'bg-neutral-100 hover:bg-neutral-200 border border-neutral-200 shadow-sm hover:shadow',
  ghost: 'hover:bg-neutral-100',
}

export function Button({ className = '', variant = 'default', size = 'md', children, ...props }) {
  const sizes = {
    md: 'px-3 py-1.5 text-sm rounded-md',
    sm: 'px-2 py-1 text-xs rounded',
    icon: 'p-2 rounded'
  }
  return (
    <button
      className={`${variants[variant] || variants.default} ${sizes[size] || sizes.md} ${className} transition-colors transition-shadow`}
      {...props}
    >
      {children}
    </button>
  )
}
