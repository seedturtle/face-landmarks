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

  function pointInPolygon(
    px: number, py: number,
    polygon: Array<{ x: number; y: number }>,
  ): boolean {
    let inside = false
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i].x, yi = polygon[i].y
      const xj = polygon[j].x, yj = polygon[j].y
      if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
        inside = !inside
      }
    }
    return inside
  }

  function detectTongueInMouth(
    imageData: ImageData,
    mouthPoints: Array<{ x: number; y: number }>,
  ): TonguePoint | null {
    const pixels = imageData.data
    const width = imageData.width
    const height = imageData.height

    // Use outer lip contour (first 12 points) as the mask — tongue cannot be outside the lips
    const outerLip = mouthPoints.slice(0, 12)
    if (outerLip.length < 6) return null

    const mouthMinY = Math.min(...mouthPoints.map(p => p.y))
    const mouthMaxY = Math.max(...mouthPoints.map(p => p.y))
    const mouthHeight = Math.max(1, mouthMaxY - mouthMinY)
    const lowerMouthStart = Math.max(0, Math.floor(mouthMinY + mouthHeight * 0.35))

    let totalWeight = 0
    let weightedX = 0
    let weightedY = 0
    let bestScore = 0
    let candidates = 0

    for (let py = lowerMouthStart; py < height; py++) {
      for (let px = 0; px < width; px++) {
        // Skip pixels outside the lip polygon
        if (!pointInPolygon(px, py, outerLip)) continue

        const idx = (py * width + px) * 4
        const r = pixels[idx]
        const g = pixels[idx + 1]
        const b = pixels[idx + 2]
        const max = Math.max(r, g, b)
        const min = Math.min(r, g, b)
        const saturation = max - min
        const brightness = (r + g + b) / 3

        const redDominance = r - Math.max(g, b)
        const pinkScore = redDominance * 1.6 + saturation * 0.35 - Math.abs(brightness - 135) * 0.15
        const isCandidate = r > 80 && r > g + 10 && r > b + 8 && brightness > 45 && brightness < 230

        if (isCandidate && pinkScore > 10) {
          const weight = Math.max(1, pinkScore)
          totalWeight += weight
          weightedX += px * weight
          weightedY += py * weight
          bestScore = Math.max(bestScore, pinkScore)
          candidates += 1
        }
      }
    }

    if (candidates < 12 || totalWeight <= 0) return null

    const x = weightedX / totalWeight
    const y = weightedY / totalWeight
    const confidence = Math.min(1, Math.max(0, (bestScore / 80) * Math.min(1, candidates / 80)))

    if (confidence < 0.12) return null

    return { x, y, confidence }
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

      const MOUTH_OPEN_THRESH = 0.42
      runningRef.current = true

      async function tick() {
        if (!runningRef.current) return
        const v = videoRef.current
        const c = canvasRef.current
        const o = overlayRef.current
        if (!v || !c || !o) return

        if (v.readyState >= 2) {
          if (c.width !== v.videoWidth || c.height !== v.videoHeight) {
            c.width = v.videoWidth
            c.height = v.videoHeight
          }
          if (o.width !== v.videoWidth || o.height !== v.videoHeight) {
            o.width = v.videoWidth
            o.height = v.videoHeight
          }
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

                      // Draw pink tongue region overlay — only inside lip contour
                      const lipPoly = mouthPts.slice(0, 12)
                      const rPixels = roiData.data
                      for (let py = 0; py < roiH; py++) {
                        for (let px = 0; px < roiW; px++) {
                          const gx = px + minX
                          const gy = py + minY
                          // Skip pixels outside the lip polygon
                          if (!pointInPolygon(gx, gy, lipPoly)) continue

                          const idx = (py * roiW + px) * 4
                          const r = rPixels[idx]
                          const g = rPixels[idx + 1]
                          const b = rPixels[idx + 2]
                          const brightness = (r + g + b) / 3
                          if (r > 80 && r > g + 10 && r > b + 8 && brightness > 45 && brightness < 230) {
                            const redD = r - Math.max(g, b)
                            const max = Math.max(r, g, b)
                            const min = Math.min(r, g, b)
                            const sat = max - min
                            const pScore = redD * 1.6 + sat * 0.35 - Math.abs(brightness - 135) * 0.15
                            if (pScore > 10) {
                              oCtx.fillStyle = 'rgba(255, 100, 150, 0.35)'
                              oCtx.fillRect(gx, gy, 2, 2)
                            }
                          }
                        }
                      }

                      oCtx.strokeStyle = '#ffaa00'
                      oCtx.lineWidth = 2
                      oCtx.strokeRect(minX, minY, roiW, roiH)

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
        👅 舌面偵測 (Tongue Surface Tracker)
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
        <strong>提示：</strong>請張開嘴巴，保持光線充足。僅偵測嘴唇輪廓（綠色線條）內的粉色區域。粉色半透明區域 = 舌面，紅色圓點 = 舌尖位置。
      </div>
    </div>
  )
}
