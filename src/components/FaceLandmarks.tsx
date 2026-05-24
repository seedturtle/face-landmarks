import { useEffect, useRef, useState, useCallback } from 'react'

// ── Facial feature landmark index groups (MediaPipe FaceMesh 468 points) ──
const FACIAL_FEATURES = {
  faceOval: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
  leftEye: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
  rightEye: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
  leftEyebrow: [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
  rightEyebrow: [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
  nose: [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 3, 45, 48, 115, 220, 237, 44, 237, 44, 240, 198, 209, 49, 131, 134, 51, 219, 78],
  lips: [0, 267, 269, 270, 409, 292, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 36, 184, 17, 95, 78, 191, 80, 81, 82, 13, 312, 308, 415, 310, 311, 297, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
  leftIris: [468, 469, 470, 471, 472],
  rightIris: [473, 474, 475, 476, 477],
}

const FEATURE_COLORS: Record<string, string> = {
  faceOval: '#8B8B8B',
  leftEye: '#00FF88',
  rightEye: '#00FF88',
  leftEyebrow: '#FFD700',
  rightEyebrow: '#FFD700',
  nose: '#FF6B6B',
  lips: '#FF69B4',
  leftIris: '#00BFFF',
  rightIris: '#00BFFF',
}

function drawFaceMesh(
  ctx: CanvasRenderingContext2D,
  keypoints: { x: number; y: number; z?: number }[],
  width: number,
  height: number,
) {
  ctx.clearRect(0, 0, width, height)

  for (const [feature, indices] of Object.entries(FACIAL_FEATURES)) {
    const color = FEATURE_COLORS[feature] || '#FFFFFF'

    // Draw connecting lines for the feature
    ctx.beginPath()
    ctx.strokeStyle = color
    ctx.lineWidth = feature === 'faceOval' ? 1.5 : 2
    ctx.globalAlpha = feature === 'faceOval' ? 0.4 : 0.9

    for (let i = 0; i < indices.length; i++) {
      const kp = keypoints[indices[i]]
      if (!kp) continue
      if (i === 0) ctx.moveTo(kp.x, kp.y)
      else ctx.lineTo(kp.x, kp.y)
    }
    ctx.stroke()

    // Draw dots for each keypoint
    const dotRadius = feature === 'leftIris' || feature === 'rightIris' ? 3 : 2.5
    for (const idx of indices) {
      const kp = keypoints[idx]
      if (!kp) continue
      ctx.beginPath()
      ctx.arc(kp.x, kp.y, dotRadius, 0, 2 * Math.PI)
      ctx.fillStyle = color
      ctx.globalAlpha = feature === 'faceOval' ? 0.3 : 0.9
      ctx.fill()
    }
  }
}

export default function FaceLandmarks() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const detectorRef = useRef<any>(null)
  const animationRef = useRef<number>(0)
  const [status, setStatus] = useState<string>('Initializing...')
  const [faceCount, setFaceCount] = useState(0)
  const [modelLoaded, setModelLoaded] = useState(false)
  const [cameraReady, setCameraReady] = useState(false)
  const streamRef = useRef<MediaStream | null>(null)

  const startDetection = useCallback(async () => {
    async function detect() {
      const video = videoRef.current
      const canvas = canvasRef.current
      const detector = detectorRef.current
      if (!video || !canvas || !detector) {
        animationRef.current = requestAnimationFrame(detect)
        return
      }

      if (video.readyState < 2) {
        animationRef.current = requestAnimationFrame(detect)
        return
      }

      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      const ctx = canvas.getContext('2d')
      if (!ctx) {
        animationRef.current = requestAnimationFrame(detect)
        return
      }

      try {
        const faces = await detector.estimateFaces(video)
        setFaceCount(faces.length)

        if (faces.length > 0) {
          for (const face of faces) {
            drawFaceMesh(ctx, face.keypoints, canvas.width, canvas.height)
          }
        } else {
          ctx.clearRect(0, 0, canvas.width, canvas.height)
        }
      } catch (err) {
        console.error('Detection error:', err)
      }

      animationRef.current = requestAnimationFrame(detect)
    }

    detect()
  }, [])

  useEffect(() => {
    let cancelled = false

    async function init() {
      setStatus('Loading TensorFlow.js...')

      try {
        const tf = await import('@tensorflow/tfjs-core')
        await import('@tensorflow/tfjs-backend-webgl')
        await import('@tensorflow/tfjs-backend-cpu')
        await tf.setBackend('webgl')
        await tf.ready()
        console.log('TF backend:', tf.getBackend())
      } catch (e) {
        console.warn('WebGL backend failed, using CPU:', e)
        await import('@tensorflow/tfjs-backend-cpu')
      }

      if (cancelled) return
      setStatus('Loading face mesh model (MediaPipeFaceMesh)...')

      try {
        const faceLandmarksDetection = await import('@tensorflow-models/face-landmarks-detection')
        const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh
        const detector = await faceLandmarksDetection.createDetector(model, {
          runtime: 'tfjs',
          refineLandmarks: true,
          maxFaces: 3,
        })
        detectorRef.current = detector
        setModelLoaded(true)
        setStatus('Model loaded. Requesting camera...')
      } catch (e) {
        console.error('Model load error:', e)
        setStatus('Error loading model: ' + (e instanceof Error ? e.message : 'Unknown error'))
        return
      }

      if (cancelled) return

      // Access webcam
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user',
          },
        })
        if (cancelled) {
          stream.getTracks().forEach(t => t.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
        setCameraReady(true)
        setStatus('Ready')
      } catch (e) {
        console.error('Camera error:', e)
        setStatus('Camera access denied. Please allow camera permission.')
      }
    }

    init()

    return () => {
      cancelled = true
      cancelAnimationFrame(animationRef.current)
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop())
      }
      // Clean up detector
      if (detectorRef.current) {
        detectorRef.current.dispose?.()
      }
    }
  }, [])

  useEffect(() => {
    if (cameraReady && modelLoaded) {
      startDetection()
    }
  }, [cameraReady, modelLoaded, startDetection])

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '16px',
      width: '100%',
      maxWidth: 720,
      margin: '0 auto',
    }}>
      {/* Status bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        padding: '8px 16px',
        borderRadius: 8,
        background: status === 'Ready' ? 'rgba(0,200,83,0.15)' : 'rgba(255,183,77,0.15)',
        fontSize: 14,
        width: '100%',
        boxSizing: 'border-box',
      }}>
        <span style={{
          width: 10, height: 10, borderRadius: '50%', display: 'inline-block', flexShrink: 0,
          background: status === 'Ready' ? '#00C853' : status.includes('Error') ? '#FF5252' : '#FFB74D',
        }} />
        <span>{status}</span>
        {faceCount > 0 && <span style={{ marginLeft: 'auto', color: '#888' }}>Faces detected: {faceCount}</span>}
      </div>

      {/* Video + Canvas container */}
      <div style={{ position: 'relative', width: '100%', borderRadius: 12, overflow: 'hidden', background: '#1a1a1a' }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ width: '100%', display: 'block', transform: 'scaleX(-1)' }}
          onLoadedData={() => {
            if (canvasRef.current && videoRef.current) {
              canvasRef.current.width = videoRef.current.videoWidth
              canvasRef.current.height = videoRef.current.videoHeight
            }
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
            transform: 'scaleX(-1)', pointerEvents: 'none',
          }}
        />
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', justifyContent: 'center', fontSize: 12 }}>
        {Object.entries(FEATURE_COLORS).map(([name, color]) => (
          <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 12, height: 12, borderRadius: '50%', background: color, display: 'inline-block' }} />
            <span>{name.replace(/([A-Z])/g, ' $1').trim()}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
