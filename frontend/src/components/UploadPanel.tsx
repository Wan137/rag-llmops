import { useRef, useState } from 'react'
import { uploadDocument } from '../api/client'
import { ApiError } from '../types'

const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.txt', '.md']

type Props = {
  onUploaded: (filename: string, chunksIndexed: number) => void
}

export function UploadPanel({ onUploaded }: Props) {
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  function getExtension(name: string) {
    const dot = name.lastIndexOf('.')
    return dot === -1 ? '' : name.slice(dot).toLowerCase()
  }

  async function handleFile(file: File) {
    setError(null)

    // check the extension before even hitting the network - instant feedback
    if (!ALLOWED_EXTENSIONS.includes(getExtension(file.name))) {
      setError(`Unsupported file type. Use one of: ${ALLOWED_EXTENSIONS.join(', ')}`)
      return
    }

    setIsUploading(true)
    try {
      const result = await uploadDocument(file)
      onUploaded(file.name, result.chunks_indexed)
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.fieldErrors?.file?.[0] ?? err.message)
      } else {
        setError('Something went wrong. Please try again.')
      }
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="p-6">
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setIsDragging(false)
          const file = e.dataTransfer.files[0]
          if (file) handleFile(file)
        }}
        onClick={() => inputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 text-center transition-colors ${
          isDragging ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ALLOWED_EXTENSIONS.join(',')}
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) handleFile(file)
            e.target.value = '' // so picking the same file twice still fires onChange
          }}
        />
        {isUploading ? (
          <p className="text-sm text-gray-500">Uploading...</p>
        ) : (
          <>
            <p className="text-sm text-gray-600">Drop a document here, or click to browse</p>
            <p className="mt-1 text-xs text-gray-400">{ALLOWED_EXTENSIONS.join(', ')}</p>
          </>
        )}
      </div>
      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
    </div>
  )
}
