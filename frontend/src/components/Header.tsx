import { StatusBadge } from './StatusBadge'

export function Header() {
  return (
    <header className="flex items-center justify-between border-b border-gray-200 px-6 py-4">
      <h1 className="text-lg font-semibold text-gray-900">RAG LLMOps</h1>
      <StatusBadge />
    </header>
  )
}
