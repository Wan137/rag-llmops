import type { SessionDoc } from './SessionDocs'

const ALL_DOCS = ''

type Props = {
  docs: SessionDoc[]
  selected: string | null
  onChange: (filename: string | null) => void
}

export function DocumentSelector({ docs, selected, onChange }: Props) {
  if (docs.length === 0) return null

  const filenames = [...new Set(docs.map((d) => d.filename))]

  return (
    <div className="flex items-center gap-2 border-t border-gray-100 px-6 py-3">
      <label className="text-xs font-medium text-gray-400">Ask about</label>
      <select
        value={selected ?? ALL_DOCS}
        onChange={(e) => onChange(e.target.value === ALL_DOCS ? null : e.target.value)}
        className="rounded border border-gray-300 px-2 py-1 text-sm"
      >
        <option value={ALL_DOCS}>All documents</option>
        {filenames.map((name) => (
          <option key={name} value={name}>
            {name}
          </option>
        ))}
      </select>
    </div>
  )
}
