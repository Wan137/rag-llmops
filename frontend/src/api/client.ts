import {
  ApiError,
  type AskResponse,
  type ChatHistoryMessage,
  type HealthResponse,
  type UploadResponse,
} from '../types'

// empty string = same-origin, which is what makes the vite dev proxy work.
// set VITE_API_BASE_URL if frontend and backend ever end up on different hosts.
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

async function handleResponse<T>(res: Response): Promise<T> {
  if (res.ok) return res.json()

  // a 400 from DRF always comes with a JSON body describing what's wrong.
  // anything else (500, HTML debug page, whatever) we treat as "just broken"
  let body: unknown
  try {
    body = await res.json()
  } catch {
    throw new ApiError('Something went wrong. Please try again.', res.status)
  }

  if (res.status === 400 && body && typeof body === 'object') {
    throw new ApiError('Please check your input.', res.status, body as Record<string, string[]>)
  }
  throw new ApiError('Something went wrong. Please try again.', res.status)
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health/`)
  return handleResponse<HealthResponse>(res)
}

export async function askQuestion(
  question: string,
  history: ChatHistoryMessage[] = [],
): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/api/ask/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, history }),
  })
  return handleResponse<AskResponse>(res)
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', file)
  // no Content-Type header here on purpose - the browser needs to set its own
  // multipart boundary, setting it manually just breaks the upload
  const res = await fetch(`${API_BASE}/api/documents/`, { method: 'POST', body: form })
  return handleResponse<UploadResponse>(res)
}
