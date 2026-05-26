import { useEffect, useRef, useState } from 'react'
import * as faceapi from '@vladmandic/face-api'

const MODEL_URL = '/models'

interface TonguePoint {
  x: number
  y: number
  confidence: number
}

interface TongueResult {
  point: TonguePoint | null
  mask: Uint8Array | null  // 1=confirmed tongue pixel (flood-filled), 0=not
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

  const tongueResultRef = useRef<TongueResult | null>(null)

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

  function isPinkPixel(r: number, g: number, b: number): boolean {
    const max = Math.max(r, g, b)
    const min = Math.min(r, g, b)
    const brightness = (r + g + b) / 3
    const saturation = max - min
    const redDominance = r - Math.max(g, b)
    const pinkScore = redDominance * 1.6 + saturation * 0.35 - Math.abs(brightness - 135) * 0.15
    return r > 80 && r > g + 10 && r > b + 8 && brightness > 45 && brightness < 230 && pinkScore > 10
  }

  function detectTongueInMouth(
    imageData: ImageData,
    mouthPoints: Array<{ x: number; y: number }>,
  ): TongueResult {
    const pixels = imageData.data
    const w = imageData.width
    const h = imageData.height

    const outerLip = mouthPoints.slice(0, 12)
    if (outerLip.length < 6) return { point: null, mask: null }

    // Step 1: build pink pixel candidate map
    // 0=not pink, 1=candidate pink (unvisited), 2=confirmed tongue (visited in BFS)
    const label = new Uint8Array(w * h)

    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const idx = (py * w + px) * 4
        if (isPinkPixel(pixels[idx], pixels[idx + 1], pixels[idx + 2])) {
          // Even outside the lip polygon it might be tongue extending out —
          // just mark as candidate for now
          label[py * w + px] = 1
        }
      }
    }

    // Step 2: BFS flood-fill from pink pixels inside lip polygon outward
    const queue: Array<[number, number]> = []
    let head = 0

    // Seed: pink pixels inside the lip polygon
    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        if (label[py * w + px] === 1 && pointInPolygon(px, py, outerLip)) {
          label[py * w + px] = 2
          queue.push([px, py])
        }
      }
    }

    // No tongue seeds found inside the lip — try a fallback: use the middle of the lip as seed
    if (queue.length === 0) {
      const cx = Math.round(outerLip.reduce((s, p) => s + p.x, 0) / outerLip.length)
      const cy = Math.round(outerLip.reduce((s, p) => s + p.y, 0) / outerLip.length)
      // Check a vertical strip near the lip center
      for (let px = Math.max(0, cx - 8); px <= Math.min(w - 1, cx + 8); px++) {
        for (let py = Math.max(0, cy - 5); py <= Math.min(h - 1, cy + 5); py++) {
          if (label[py * w + px] === 1) {
            label[py * w + px] = 2
            queue.push([px, py])
          }
        }
      }
    }

    // BFS: expand through 4-directionally connected pink pixels
    const dirs = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    while (head < queue.length) {
      const [cx, cy] = queue[head++]
      for (const [dx, dy] of dirs) {
        const nx = cx + dx
        const ny = cy + dy
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue
        const idx = ny * w + nx
        if (label[idx] === 1) {
          label[idx] = 2
          queue.push([nx, ny])
        }
      }
    }

    // Step 3: Compute tongue centroid and find tip from confirmed (label===2) pixels
    let totalPixels = 0
    let sumX = 0
    let sumY = 0

    // First pass: centroid
    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        if (label[py * w + px] === 2) {
          sumX += px
          sumY += py
          totalPixels++
        }
      }
    }

    if (totalPixels < 15) {
      // No significant tongue region found — return empty mask too
      return { point: null, mask: null }
    }

    const cx = sumX / totalPixels
    const cy = sumY / totalPixels

    // Find tip: the confirmed tongue pixel furthest from centroid,
    // biased toward the lower direction (where the tongue tip typically points)
    let maxDist2 = 0
    let tipX = cx
    let tipY = cy

    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        if (label[py * w + px] === 2) {
          const dx = px - cx
          const dy = py - cy
          // Bias downward (positive Y): dy gets 1.5x weight
          const dist2 = dx * dx + dy * dy * 1.5
          if (dist2 > maxDist2) {
            maxDist2 = dist2
            tipX = px
            tipY = py
          }
        }
      }
    }

    const confidence = Math.min(1, totalPixels / 200)

    // Build output mask (just label === 2 → 1)
    const mask = new Uint8Array(w * h)
    for (let i = 0; i < w * h; i++) {
      mask[i] = label[i] === 2 ? 1 : 0
    }

    tongueResultRef.current = {
      point: { x: tipX, y: tipY, confidence },
      mask,
    }

    return tongueResultRef.current
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

                    const result = detectTongueInMouth(roiData, localMouth)
                    const tip = result.point
                    const mask = result.mask

                    const oCtx = o.getContext('2d')
                    if (oCtx) {
                      oCtx.clearRect(0, 0, o.width, o.height)

                      // Draw pink tongue region overlay using the flood-filled mask
                      if (mask && mask.length === roiW * roiH) {
                        for (let py = 0; py < roiH; py++) {
                          for (let px = 0; px < roiW; px++) {
                            if (mask[py * roiW + px] === 1) {
                              const gx = px + minX
                              const gy = py + minY
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
                        oCtx.arc(globalX, globalY, 5, 0, Math.PI * 2)
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
        <strong>提示：</strong>請張開嘴巴，保持光線充足。以嘴唇輪廓內為種子點，經 flood-fill 連通區域分析保留完整舌面（含伸出嘴唇的部份）。粉色半透明區域 = 舌面，紅色圓點 = 舌尖位置。
      </div>
    </div>
  )
}
