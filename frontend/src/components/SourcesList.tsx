import { useState } from 'react'
import type { Source } from '../types'

export function SourcesList({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false)

  if (sources.length === 0) return null

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(!open)}
        className="text-xs font-medium text-gray-400 hover:text-gray-600"
      >
        {open ? 'Hide' : 'Show'} sources ({sources.length})
      </button>
      {open && (
        <ul className="mt-2 space-y-2">
          {sources.map((s, i) => (
            <li key={i} className="rounded bg-gray-50 p-2 text-xs text-gray-500">
              <span className="font-medium text-gray-600">
                {s.source}
                {s.page !== null && ` (page ${s.page})`}
              </span>
              <p className="mt-1">{s.snippet}...</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
