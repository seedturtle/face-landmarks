import { useState } from 'react'
import FaceLandmarks from './components/FaceLandmarks'
import TongueTracker from './components/TongueTracker'

export default function App() {
  const [mode, setMode] = useState<'landmarks' | 'tongue'>('landmarks')

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%)',
      color: '#e0e0e0',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '24px 16px',
      boxSizing: 'border-box',
    }}>
      <div style={{ textAlign: 'center', marginBottom: 24, maxWidth: 640 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, margin: '0 0 8px', background: 'linear-gradient(90deg, #667eea, #764ba2)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Face Landmarks + Tongue Tracking
        </h1>
        <p style={{ margin: 0, color: '#999', fontSize: 14, lineHeight: 1.6 }}>
          Real-time facial feature detection using TensorFlow.js + MediaPipe FaceMesh.
          <br />Detects 468 facial landmarks and tracks tongue tip position.
        </p>
      </div>

      <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
        <button
          onClick={() => setMode('landmarks')}
          style={{
            padding: '10px 20px',
            borderRadius: 8,
            border: 'none',
            cursor: 'pointer',
            fontSize: 14,
            fontWeight: 600,
            background: mode === 'landmarks' ? '#667eea' : 'rgba(255,255,255,0.08)',
            color: mode === 'landmarks' ? '#fff' : '#999',
          }}
        >
          🤖 人臉五官辨識
        </button>
        <button
          onClick={() => setMode('tongue')}
          style={{
            padding: '10px 20px',
            borderRadius: 8,
            border: 'none',
            cursor: 'pointer',
            fontSize: 14,
            fontWeight: 600,
            background: mode === 'tongue' ? '#667eea' : 'rgba(255,255,255,0.08)',
            color: mode === 'tongue' ? '#fff' : '#999',
          }}
        >
          👅 舌頭追蹤
        </button>
      </div>

      {mode === 'landmarks' ? <FaceLandmarks /> : <TongueTracker />}
    </div>
  )
}
