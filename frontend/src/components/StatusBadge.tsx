import { useEffect, useState } from 'react'
import { getHealth } from '../api/client'
import type { HealthResponse } from '../types'

const POLL_MS = 30_000

export function StatusBadge() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [isDown, setIsDown] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function check() {
      try {
        const h = await getHealth()
        if (!cancelled) {
          setHealth(h)
          setIsDown(false)
        }
      } catch {
        if (!cancelled) setIsDown(true)
      }
    }

    check()
    const id = setInterval(check, POLL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  const ok = health?.status === 'ok' && !isDown

  return (
    <div className="flex items-center gap-2 text-sm text-gray-500">
      <span className={`h-2 w-2 rounded-full ${ok ? 'bg-green-500' : 'bg-red-500'}`} />
      {ok ? (
        <span>
          {health!.llm_model} &middot; {health!.vector_count} chunks indexed
        </span>
      ) : (
        <span>backend unreachable</span>
      )}
    </div>
  )
}
