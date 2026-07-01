import { useState } from 'react'
import { Header } from './components/Header'
import { UploadPanel } from './components/UploadPanel'
import { SessionDocs, type SessionDoc } from './components/SessionDocs'
import { ChatPanel } from './components/ChatPanel'

function App() {
  const [sessionDocs, setSessionDocs] = useState<SessionDoc[]>([])

  return (
    <div className="mx-auto flex h-screen max-w-2xl flex-col bg-white">
      <Header />
      <UploadPanel
        onUploaded={(filename, chunksIndexed) =>
          setSessionDocs((docs) => [...docs, { filename, chunksIndexed }])
        }
      />
      <SessionDocs docs={sessionDocs} />
      <ChatPanel />
    </div>
  )
}

export default App
