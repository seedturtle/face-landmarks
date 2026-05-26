import { useEffect, useRef, useState } from 'react'
import * as faceapi from '@vladmandic/face-api'

const MODEL_URL = '/models'

interface TonguePoint {
  x: number
  y: number
  confidence: number
}

export default function TongueTracker() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef(0)
  const streamRef = useRef<MediaStream | null>(null)
  const runningRef = useRef(false)

  const [status, setStatus] = useState('載入中...')
  const [tonguePos, setTonguePos] = useState<TonguePoint | null>(null)
  const [mouthOpen, setMouthOpen] = useState(false)
  const [metric, setMetric] = useState(0)

  function detectTongueInMouth(
    imageData: ImageData,
    mouthPoints: Array<{ x: number; y: number }>,
  ): TonguePoint | null {
    const pixels = imageData.data
    const width = imageData.width
    const height = imageData.height

    const mouthMinY = Math.min(...mouthPoints.map(p => p.y))
    const mouthMaxY = Math.max(...mouthPoints.map(p => p.y))
    const mouthHeight = mouthMaxY - mouthMinY

    let bestY = -1
    let bestX = -1
    let darkestVal = 255

    const startRow = Math.floor(height * 0.3)

    for (let py = startRow; py < height; py++) {
      for (let px = 0; px < width; px++) {
        const idx = (py * width + px) * 4
        const r = pixels[idx]
        const g = pixels[idx + 1]
        const b = pixels[idx + 2]
        const gray = 0.299 * r + 0.587 * g + 0.114 * b

        if (gray < darkestVal) {
          darkestVal = gray
          bestX = px
          bestY = py
        }
      }
    }

    if (bestY < 0 || darkestVal > 80) return null

    const normalizedY = (bestY - startRow) / (height - startRow)
    const confidence = 1 - darkestVal / 80

    return {
      x: bestX,
      y: bestY,
      confidence: Math.min(1, Math.max(0, confidence)),
    }
  }

  function mouthAspectRatio(mouthPoints: Array<{ x: number; y: number }>): number {
    if (mouthPoints.length < 20) return 0
    const a = Math.hypot(
      mouthPoints[13].x - mouthPoints[19].x,
      mouthPoints[13].y - mouthPoints[19].y,
    )
    const b = Math.hypot(
      mouthPoints[14].x - mouthPoints[18].x,
      mouthPoints[14].y - mouthPoints[18].y,
    )
    const c = Math.hypot(
      mouthPoints[2].x - mouthPoints[10].x,
      mouthPoints[2].y - mouthPoints[10].y,
    )
    if (c === 0) return 0
    return (a + b) / (2 * c)
  }

  useEffect(() => {
    let cancelled = false

    async function init() {
      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas) return

      try {
        setStatus('啟動攝影機...')
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        })
        streamRef.current = stream
        video.srcObject = stream
        await video.play()
      } catch {
        setStatus('無法取得攝影機權限')
        return
      }

      await new Promise<void>(resolve => {
        const apply = () => {
          if (video.videoWidth > 0) {
            canvas.width = video.videoWidth
            canvas.height = video.videoHeight
            resolve()
          }
        }
        apply()
        video.onloadedmetadata = apply
      })
      if (cancelled) return

      try {
        setStatus('載入模型...')
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL)
        await faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL)
      } catch {
        setStatus('模型載入失敗')
        return
      }

      setStatus('就緒 — 請張開嘴巴')

      const MOUTH_OPEN_THRESH = 0.55
      runningRef.current = true

      async function tick() {
        if (!runningRef.current) return
        const v = videoRef.current
        const c = canvasRef.current
        const o = overlayRef.current
        if (!v || !c || !o) return

        if (v.readyState >= 2) {
          try {
            const results = await faceapi
              .detectAllFaces(v, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.4 }))
              .withFaceLandmarks(true)

            if (results.length > 0) {
              const lm = results[0].landmarks
              const mouthPts = lm.getMouth()

              const mar = mouthAspectRatio(mouthPts)
              const isOpen = mar > MOUTH_OPEN_THRESH
              setMouthOpen(isOpen)

              const ctx = c.getContext('2d')
              if (ctx) {
                ctx.clearRect(0, 0, c.width, c.height)

                if (isOpen) {
                  const xs = mouthPts.map(p => p.x)
                  const ys = mouthPts.map(p => p.y)
                  const minX = Math.max(0, Math.floor(Math.min(...xs) - 10))
                  const minY = Math.max(0, Math.floor(Math.min(...ys) - 5))
                  const maxX = Math.min(v.videoWidth, Math.ceil(Math.max(...xs) + 10))
                  const maxY = Math.min(v.videoHeight, Math.ceil(Math.max(...ys) + 5))
                  const roiW = maxX - minX
                  const roiH = maxY - minY

                  if (roiW > 10 && roiH > 10) {
                    ctx.drawImage(v, minX, minY, roiW, roiH, 0, 0, roiW, roiH)
                    const roiData = ctx.getImageData(0, 0, roiW, roiH)
                    const localMouth = mouthPts.map(p => ({ x: p.x - minX, y: p.y - minY }))

                    const tip = detectTongueInMouth(roiData, localMouth)

                    const oCtx = o.getContext('2d')
                    if (oCtx) {
                      oCtx.clearRect(0, 0, o.width, o.height)

                      oCtx.strokeStyle = '#00ff88'
                      oCtx.lineWidth = 2
                      oCtx.beginPath()
                      oCtx.moveTo(mouthPts[0].x, mouthPts[0].y)
                      for (let i = 1; i < mouthPts.length; i++) {
                        oCtx.lineTo(mouthPts[i].x, mouthPts[i].y)
                      }
                      oCtx.closePath()
                      oCtx.stroke()

                      if (tip) {
                        const globalX = tip.x + minX
                        const globalY = tip.y + minY
                        setTonguePos({ x: globalX, y: globalY, confidence: tip.confidence })
                        setMetric(Math.round(tip.confidence * 100))

                        oCtx.fillStyle = '#ff3366'
                        oCtx.beginPath()
                        oCtx.arc(globalX, globalY, 6, 0, Math.PI * 2)
                        oCtx.fill()

                        oCtx.fillStyle = '#ffffff'
                        oCtx.font = '12px monospace'
                        oCtx.fillText(`👅 ${Math.round(tip.confidence * 100)}%`, globalX + 12, globalY - 8)
                      } else {
                        setTonguePos(null)
                        setMetric(0)
                      }
                    }
                  }
                } else {
                  const oCtx = o.getContext('2d')
                  if (oCtx) oCtx.clearRect(0, 0, o.width, o.height)
                  setTonguePos(null)
                  setMetric(0)
                }
              }
            }
          } catch {
            // detection frame skip
          }
        }
        rafRef.current = requestAnimationFrame(tick)
      }

      tick()
    }

    init()
    return () => {
      cancelled = true
      runningRef.current = false
      cancelAnimationFrame(rafRef.current)
      streamRef.current?.getTracks().forEach(t => t.stop())
    }
  }, [])

  return (
    <div style={{
      minHeight: '100vh', background: '#0d1117', color: '#c9d1d9',
      fontFamily: 'system-ui, sans-serif', display: 'flex',
      flexDirection: 'column', alignItems: 'center',
      padding: '24px 16px', gap: 16,
    }}>
      <h1 style={{ fontSize: 20, fontWeight: 700, margin: 0 }}>
        👅 舌頭尖端追蹤 (Tongue Tip Tracker)
      </h1>

      <div style={{
        position: 'relative', borderRadius: 12, overflow: 'hidden',
        border: '2px solid #30363d', maxWidth: '100%', width: 480,
        aspectRatio: '4 / 3', background: '#000',
      }}>
        <video
          ref={videoRef}
          style={{
            width: '100%', height: '100%', objectFit: 'cover',
            transform: 'scaleX(-1)', background: '#000', display: 'block',
          }}
          autoPlay muted playsInline
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute', inset: 0, width: '100%', height: '100%',
            objectFit: 'cover', transform: 'scaleX(-1)',
            display: 'none',
          }}
        />
        <canvas
          ref={overlayRef}
          style={{
            position: 'absolute', inset: 0, width: '100%', height: '100%',
            objectFit: 'cover', transform: 'scaleX(-1)', pointerEvents: 'none',
          }}
        />
      </div>

      <div style={{ fontSize: 14, color: '#8b949e' }}>{status}</div>

      <div style={{
        display: 'flex', gap: 24, fontSize: 13, color: '#8b949e',
        flexWrap: 'wrap', justifyContent: 'center',
      }}>
        <span>嘴巴：{mouthOpen ? '🟢 張開' : '🔴 閉合'}</span>
        <span>舌頭偵測：{metric}%</span>
        <span>
          位置：{tonguePos ? `${tonguePos.x.toFixed(0)}, ${tonguePos.y.toFixed(0)}` : '—'}
        </span>
      </div>

      <div style={{
        marginTop: 8, padding: '12px 16px', borderRadius: 8,
        background: 'rgba(255,255,255,0.03)', fontSize: 12, color: '#6e7681',
        maxWidth: 480, width: '100%', boxSizing: 'border-box',
      }}>
        <strong>提示：</strong>請張開嘴巴，保持光線充足。紅色圓點會追蹤舌頭尖端位置。
      </div>
    </div>
  )
}
