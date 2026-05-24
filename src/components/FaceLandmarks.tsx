import { useEffect, useRef, useState, useCallback } from 'react'
import * as faceapi from 'face-api.js'

// Load models from public/models/ (bundled with the app)
const MODEL_URL = '/models'

const COLORS: Record<string, string> = {
  Face: '#ff4444',
  LeftBrow: '#ffff44',
  RightBrow: '#ffff44',
  LeftEye: '#44ff44',
  RightEye: '#44ff44',
  Nose: '#44aaff',
  Mouth: '#ff44ff',
  IrisL: '#44ffff',
  IrisR: '#44ffff',
}

function drawPath(ctx: CanvasRenderingContext2D, pts: faceapi.Point[], color: string, filled = false) {
  ctx.beginPath()
  ctx.strokeStyle = color
  ctx.fillStyle = color
  ctx.lineWidth = 2
  if (pts.length === 0) return
  ctx.moveTo(pts[0].x, pts[0].y)
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y)
  if (filled) ctx.fill()
  ctx.stroke()
}

export default function FaceLandmarks() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const msgRef = useRef<HTMLDivElement>(null)
  const [status, setStatus] = useState('⏳ 初始化中...')
  const [loaded, setLoaded] = useState(false)
  const [faceCount, setFaceCount] = useState(0)
  const [detectCount, setDetectCount] = useState(0)
  const lastTime = useRef(0)
  const rafId = useRef(0)

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const drawOverlays = useCallback((canvas: HTMLCanvasElement | null, results: any[]) => {
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    setFaceCount(results.length)

    if (results.length > 0) {
      setDetectCount(c => c + 1)
      lastTime.current = Date.now()
      if (msgRef.current) { msgRef.current.textContent = '✅ 偵測到臉'; msgRef.current.style.color = '#4ade80' }
    }

    for (const result of results) {
      const lm = result.landmarks
      drawPath(ctx, lm.getJawOutline(), COLORS.Face)
      drawPath(ctx, lm.getLeftEyeBrow(), COLORS.LeftBrow)
      drawPath(ctx, lm.getRightEyeBrow(), COLORS.RightBrow)
      drawPath(ctx, lm.getLeftEye(), COLORS.LeftEye)
      drawPath(ctx, lm.getRightEye(), COLORS.RightEye)
      drawPath(ctx, lm.getNose(), COLORS.Nose)
      drawPath(ctx, lm.getMouth(), COLORS.Mouth)
      const leftIris = lm.getLeftEye().slice(0, 2)
      const rightIris = lm.getRightEye().slice(0, 2)
      drawPath(ctx, leftIris, COLORS.IrisL, true)
      drawPath(ctx, rightIris, COLORS.IrisR, true)
    }
  }, [])

  const start = useCallback(async () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    setStatus('⏳ 啟動攝影機...')
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } } })
      video.srcObject = stream
      await video.play()
    } catch {
      setStatus('❌ 無法取得攝影機權限')
      return
    }

    await new Promise<void>(res => { video.onloadedmetadata = () => { canvas.width = video.videoWidth; canvas.height = video.videoHeight; res() } })

    setStatus('⏳ 載入模型（ 約需下載 0.5MB）...')
    try {
      await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL)
      await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL)
      setLoaded(true)
      setStatus('✅ 準備就緒，請將臉對準鏡頭')
    } catch (e) {
      setStatus('❌ 模型載入失敗：' + String(e))
      return
    }

    async function tick() {
      if (!video || video.readyState < 2) { rafId.current = requestAnimationFrame(tick); return }
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const results: any[] = await (faceapi as any).detectAllFaces(video as any, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 }))
          .withFaceLandmarks()
        drawOverlays(canvas, results)
      } catch {}
      if (Date.now() - lastTime.current > 5000 && msgRef.current) {
        msgRef.current.textContent = '⚠️ 未偵測到臉，請對準鏡頭並確保光線充足'
        msgRef.current.style.color = '#fbbf24'
      }
      rafId.current = requestAnimationFrame(tick)
    }
    tick()
  }, [drawOverlays])

  useEffect(() => {
    start()
    return () => { cancelAnimationFrame(rafId.current) }
  }, [start])

  return (
    <div style={{ minHeight: '100vh', background: '#111', color: '#eee', fontFamily: 'system-ui, sans-serif', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', padding: '24px 16px', gap: 16 }}>
      <h1 style={{ fontSize: 20, fontWeight: 700, margin: 0 }}>🤖 人臉五官辨識</h1>
      <div style={{ position: 'relative', borderRadius: 12, overflow: 'hidden', border: '2px solid #333', maxWidth: '100%', width: 480, aspectRatio: '4/3' }}>
        <video ref={videoRef} style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)', background: '#000', display: 'block' }} autoPlay muted playsInline />
        <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }} />
      </div>
      <div ref={msgRef} style={{ fontSize: 14, color: '#9ca3af', fontWeight: 500, textAlign: 'center', minHeight: 20 }}>{status}</div>
      <div style={{ fontSize: 13, color: '#6b7280', display: 'flex', gap: 16, flexWrap: 'wrap', justifyContent: 'center' }}>
        <span>偵測：{detectCount} 次</span><span>臉數：{faceCount}</span><span>模型：{loaded ? 'face-api.js 68pt' : '載入中...'}</span>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, justifyContent: 'center', fontSize: 12 }}>
        {Object.entries(COLORS).map(([name, color]) => (
          <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 12, height: 12, borderRadius: '50%', background: color, display: 'inline-block' }} />
            <span>{name}</span>
          </div>
        ))}
      </div>
    </div>
  )
}