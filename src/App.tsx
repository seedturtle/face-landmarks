import FaceLandmarks from './components/FaceLandmarks'

export default function App() {
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
      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: 24, maxWidth: 640 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, margin: '0 0 8px', background: 'linear-gradient(90deg, #667eea, #764ba2)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Face Landmarks Detection
        </h1>
        <p style={{ margin: 0, color: '#999', fontSize: 14, lineHeight: 1.6 }}>
          Real-time facial feature detection using TensorFlow.js + MediaPipe FaceMesh.{' '}
          <br />Detects 468 facial landmarks — eyes, nose, lips, eyebrows, and face contour.
        </p>
      </div>

      {/* Main component */}
      <FaceLandmarks />

      {/* Instructions */}
      <div style={{ marginTop: 24, padding: '16px 20px', borderRadius: 12, background: 'rgba(255,255,255,0.04)', maxWidth: 640, width: '100%', boxSizing: 'border-box', fontSize: 13, lineHeight: 1.7, color: '#aaa' }}>
        <strong style={{ color: '#ddd' }}>Instructions:</strong>
        <ol style={{ margin: '8px 0 0', paddingLeft: 20 }}>
          <li>Allow camera access when prompted</li>
          <li>Position your face in the frame with good lighting</li>
          <li>The face mesh overlay will appear in real-time</li>
          <li>Each facial feature is color-coded per the legend below</li>
        </ol>
      </div>
    </div>
  )
}
