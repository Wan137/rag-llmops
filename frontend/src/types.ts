// mirrors the Django API responses 1:1 - keep this in sync if the backend contract changes

export type Source = {
  source: string
  chunk_index: number
  page: number | null
  snippet: string
}

export type AskResponse = {
  answer: string
  sources: Source[]
}

// what we send back to the backend so it has some memory of the conversation
export type ChatHistoryMessage = {
  role: 'user' | 'assistant'
  text: string
}

export type HealthResponse = {
  status: string
  vector_count: number
  llm_model: string
  embedding_model: string
}

export type UploadResponse = {
  chunks_indexed: number
  total_vectors: number
}

// thrown by the api client on any non-2xx response
export class ApiError extends Error {
  status: number
  fieldErrors?: Record<string, string[]> // set for DRF 400s, e.g. { question: ["This field is required."] }

  constructor(message: string, status: number, fieldErrors?: Record<string, string[]>) {
    super(message)
    this.status = status
    this.fieldErrors = fieldErrors
  }
}
