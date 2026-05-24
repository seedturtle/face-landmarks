import { useEffect, useRef, useState } from 'react'
import * as faceapi from '@vladmandic/face-api'

const MODEL_URL = '/models'

const COLORS: Record<string, string> = {
  Face: '#ff4444',
  LeftBrow: '#ffff44',
  RightBrow: '#ffff44',
  LeftEye: '#44ff44',
  RightEye: '#44ff44',
  Nose: '#44aaff',
  Mouth: '#ff44ff',
}

function drawPath(ctx: CanvasRenderingContext2D, pts: Array<{ x: number; y: number }>, color: string, close = false) {
  if (!pts.length) return
  ctx.beginPath()
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.moveTo(pts[0].x, pts[0].y)
  for (let i = 1; i < pts.length; i += 1) ctx.lineTo(pts[i].x, pts[i].y)
  if (close) ctx.closePath()
  ctx.stroke()
}

function drawDots(ctx: CanvasRenderingContext2D, pts: Array<{ x: number; y: number }>, color: string) {
  ctx.fillStyle = color
  for (const pt of pts) {
    ctx.beginPath()
    ctx.arc(pt.x, pt.y, 2.5, 0, Math.PI * 2)
    ctx.fill()
  }
}

function drawOverlays(canvas: HTMLCanvasElement, results: any[]) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  for (const result of results) {
    const lm = result.landmarks
    const jaw = lm.getJawOutline()
    const leftBrow = lm.getLeftEyeBrow()
    const rightBrow = lm.getRightEyeBrow()
    const leftEye = lm.getLeftEye()
    const rightEye = lm.getRightEye()
    const nose = lm.getNose()
    const mouth = lm.getMouth()

    drawPath(ctx, jaw, COLORS.Face)
    drawPath(ctx, leftBrow, COLORS.LeftBrow)
    drawPath(ctx, rightBrow, COLORS.RightBrow)
    drawPath(ctx, leftEye, COLORS.LeftEye, true)
    drawPath(ctx, rightEye, COLORS.RightEye, true)
    drawPath(ctx, nose, COLORS.Nose)
    drawPath(ctx, mouth, COLORS.Mouth, true)
    drawDots(ctx, [...jaw, ...leftBrow, ...rightBrow, ...leftEye, ...rightEye, ...nose, ...mouth], '#ffffff')
  }
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error(`${label} timeout after ${ms}ms`)), ms)
    promise.then(
      value => {
        window.clearTimeout(timer)
        resolve(value)
      },
      err => {
        window.clearTimeout(timer)
        reject(err)
      },
    )
  })
}

export default function FaceLandmarks() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const msgRef = useRef<HTMLDivElement>(null)
  const rafIdRef = useRef(0)
  const streamRef = useRef<MediaStream | null>(null)
  const runningRef = useRef(false)
  const lastDetectedAtRef = useRef(Date.now())
  const [status, setStatus] = useState('⏳ 初始化中...')
  const [faceCount, setFaceCount] = useState(0)
  const [detectCount, setDetectCount] = useState(0)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function init() {
      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas) return

      try {
        setStatus('⏳ 啟動攝影機...')
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        })
        streamRef.current = stream
        video.srcObject = stream
        await video.play()
      } catch (e) {
        setStatus(`❌ 無法取得攝影機權限：${String(e)}`)
        return
      }

      await new Promise<void>(resolve => {
        const applySize = () => {
          if (video.videoWidth > 0 && video.videoHeight > 0) {
            canvas.width = video.videoWidth
            canvas.height = video.videoHeight
            resolve()
          }
        }
        applySize()
        video.onloadedmetadata = applySize
      })
      if (cancelled) return

      try {
        setStatus('⏳ 載入模型：TinyFaceDetector...')
        await withTimeout(faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL), 15000, 'TinyFaceDetector')
        if (cancelled) return

        setStatus('⏳ 載入模型：FaceLandmark68Tiny...')
        await withTimeout(faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL), 15000, 'FaceLandmark68Tiny')
        if (cancelled) return
      } catch (e) {
        setStatus(`❌ 模型載入失敗：${String(e)}`)
        return
      }

      setLoaded(true)
      setStatus('✅ 準備就緒，請將臉對準鏡頭')
      runningRef.current = true
      lastDetectedAtRef.current = Date.now()

      async function tick() {
        const currentVideo = videoRef.current
        const currentCanvas = canvasRef.current
        if (!runningRef.current || !currentVideo || !currentCanvas) return

        if (currentVideo.readyState >= 2) {
          try {
            const results = await faceapi
              .detectAllFaces(currentVideo, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.35 }))
              .withFaceLandmarks(true)

            setFaceCount(results.length)
            drawOverlays(currentCanvas, results as any[])

            if (results.length > 0) {
              setDetectCount(c => c + 1)
              lastDetectedAtRef.current = Date.now()
              if (msgRef.current) {
                msgRef.current.textContent = `✅ 偵測到臉：${results.length}`
                msgRef.current.style.color = '#4ade80'
              }
            } else if (Date.now() - lastDetectedAtRef.current > 3000 && msgRef.current) {
              msgRef.current.textContent = '⚠️ 未偵測到臉，請正面看鏡頭並保持光線充足'
              msgRef.current.style.color = '#fbbf24'
            }
          } catch (e) {
            if (msgRef.current) {
              msgRef.current.textContent = `❌ 偵測錯誤：${String(e)}`
              msgRef.current.style.color = '#f87171'
            }
          }
        }

        rafIdRef.current = requestAnimationFrame(tick)
      }

      tick()
    }

    init()

    return () => {
      cancelled = true
      runningRef.current = false
      cancelAnimationFrame(rafIdRef.current)
      streamRef.current?.getTracks().forEach(track => track.stop())
    }
  }, [])

  return (
    <div style={{ minHeight: '100vh', background: '#111', color: '#eee', fontFamily: 'system-ui, sans-serif', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', padding: '24px 16px', gap: 16 }}>
      <h1 style={{ fontSize: 20, fontWeight: 700, margin: 0 }}>🤖 人臉五官辨識</h1>
      <div style={{ position: 'relative', borderRadius: 12, overflow: 'hidden', border: '2px solid #333', maxWidth: '100%', width: 480, aspectRatio: '4 / 3', background: '#000' }}>
        <video ref={videoRef} style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)', background: '#000', display: 'block' }} autoPlay muted playsInline />
        <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)', pointerEvents: 'none' }} />
      </div>
      <div ref={msgRef} style={{ fontSize: 14, color: '#9ca3af', fontWeight: 500, textAlign: 'center', minHeight: 20 }}>{status}</div>
      <div style={{ fontSize: 13, color: '#9ca3af', display: 'flex', gap: 16, flexWrap: 'wrap', justifyContent: 'center' }}>
        <span>偵測：{detectCount} 次</span>
        <span>臉數：{faceCount}</span>
        <span>模型：{loaded ? 'Tiny 68pt ✅' : '載入中...'}</span>
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
