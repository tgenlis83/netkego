import React from 'react'

export function Slider({ value = [0], min = 0, max = 100, step = 1, onValueChange, className = '' }) {
  const v = value[0]
  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={v}
      onChange={(e) => onValueChange && onValueChange([parseFloat(e.target.value)])}
      className={`w-full accent-blue-600 h-2 rounded-lg appearance-none bg-neutral-200 ${className}`}
    />
  )
}
