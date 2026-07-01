import { useEffect, useRef, useState } from 'react'
import { askQuestion } from '../api/client'
import { ApiError, type ChatHistoryMessage } from '../types'
import { MessageBubble, type Message } from './MessageBubble'

// backend only knows about user/assistant turns, so error bubbles get dropped here
function toHistory(messages: Message[]): ChatHistoryMessage[] {
  return messages
    .filter((m): m is Message & { role: 'user' | 'assistant' } => m.role !== 'error')
    .map((m) => ({ role: m.role, text: m.text }))
}

export function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [isAsking, setIsAsking] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isAsking])

  async function handleSend() {
    const q = question.trim()
    if (!q || isAsking) return

    const history = toHistory(messages)
    setMessages((m) => [...m, { role: 'user', text: q }])
    setQuestion('')
    setIsAsking(true)

    try {
      const result = await askQuestion(q, history)
      setMessages((m) => [...m, { role: 'assistant', text: result.answer, sources: result.sources }])
    } catch (err) {
      const text =
        err instanceof ApiError ? (err.fieldErrors?.question?.[0] ?? err.message) : 'Something went wrong.'
      setMessages((m) => [...m, { role: 'error', text }])
    } finally {
      setIsAsking(false)
    }
  }

  return (
    <div className="flex flex-1 flex-col">
      <div className="flex-1 space-y-3 overflow-y-auto px-6 py-4">
        {messages.length === 0 && (
          <p className="mt-8 text-center text-sm text-gray-400">
            Upload a document above, then ask something about it.
          </p>
        )}
        {messages.map((m, i) => (
          <MessageBubble key={i} message={m} />
        ))}
        {isAsking && (
          <div className="flex justify-start">
            <div className="rounded-2xl rounded-bl-sm bg-gray-100 px-4 py-2 text-sm text-gray-400">
              thinking...
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="flex gap-2 border-t border-gray-200 p-4">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          disabled={isAsking}
          placeholder="Ask a question..."
          className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-400 focus:outline-none disabled:bg-gray-50"
        />
        <button
          onClick={handleSend}
          disabled={isAsking || !question.trim()}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          Send
        </button>
      </div>
    </div>
  )
}
