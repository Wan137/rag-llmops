export type SessionDoc = {
  filename: string
  chunksIndexed: number
}

type Props = {
  docs: SessionDoc[]
}

export function SessionDocs({ docs }: Props) {
  if (docs.length === 0) return null

  return (
    <div className="border-t border-gray-100 px-6 py-4">
      <p className="mb-2 text-xs font-medium text-gray-400 uppercase">Uploaded this session</p>
      <ul className="space-y-1">
        {docs.map((doc, i) => (
          <li key={i} className="flex justify-between text-sm text-gray-600">
            <span className="truncate">{doc.filename}</span>
            <span className="ml-2 shrink-0 text-gray-400">{doc.chunksIndexed} chunks</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
