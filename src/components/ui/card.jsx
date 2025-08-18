import React from 'react'

export function Card({ className = '', children, ...props }) {
  return (
    <div
      className={`rounded-2xl border border-neutral-200 bg-white/90 backdrop-blur-sm shadow-sm hover:shadow-md transition-shadow ${className}`}
      {...props}
    >
      {children}
    </div>
  )
}

export function CardHeader({ className = '', children, ...props }) {
  return (
    <div className={`px-4 py-3 border-b border-neutral-100 ${className}`} {...props}>
      {children}
    </div>
  )
}

export function CardTitle({ className = '', children, ...props }) {
  return (
    <div className={`text-base font-semibold text-neutral-900 ${className}`} {...props}>
      {children}
    </div>
  )
}

export function CardContent({ className = '', children, ...props }) {
  return (
    <div className={`px-4 py-3 ${className}`} {...props}>
      {children}
    </div>
  )
}
