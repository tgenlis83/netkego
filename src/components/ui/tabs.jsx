import React, { useState } from 'react'

export function Tabs({ value, onValueChange, children }) {
  const [internal, setInternal] = useState(value || '')
  const current = value !== undefined ? value : internal
  const set = onValueChange || setInternal
  return React.Children.map(children, (child) => {
    if (!React.isValidElement(child)) return child
    return React.cloneElement(child, { __tabsValue: current, __setTabsValue: set })
  })
}

export function TabsList({ children, __tabsValue, __setTabsValue }) {
  return (
    <div className="inline-flex gap-2 mb-2 border border-neutral-200 rounded-md p-1 bg-white shadow-sm">
      {React.Children.map(children, (child) =>
        React.isValidElement(child)
          ? React.cloneElement(child, { __tabsValue, __setTabsValue })
          : child
      )}
    </div>
  )
}

export function TabsTrigger({ value, __tabsValue, __setTabsValue, children }) {
  const active = __tabsValue === value
  return (
    <button
      className={`px-3 py-1 text-sm rounded transition ${active ? 'bg-blue-600 text-white shadow' : 'hover:bg-neutral-100'}`}
      onClick={() => __setTabsValue && __setTabsValue(value)}
      type="button"
    >
      {children}
    </button>
  )
}

export function TabsContent({ value, __tabsValue, children }) {
  if (__tabsValue !== value) return null
  return <div>{children}</div>
}
