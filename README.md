# Face Landmarks Detection

A real-time facial feature detection web app using TensorFlow.js and MediaPipe FaceMesh.

Built with **Vite + React + TypeScript**, deployable to **Zeabur** as a static site.

## Features

- Real-time face detection via webcam
- 468 facial landmark points
- Color-coded facial features: eyes, eyebrows, nose, lips, face contour, and iris
- Canvas overlay with mesh visualization
- Runs entirely in the browser — no server-side processing

## Quick Start

```bash
npm install
npm run dev
```

Open http://localhost:3000 in your browser. Allow camera access when prompted.

## Deploy to Zeabur

1. Push this repository to GitHub
2. Go to [Zeabur.com](https://zeabur.com) and create a new project
3. Select your GitHub repo
4. Zeabur will auto-detect the Vite project and deploy it

The `zeabur.json` configuration is already included.

## Tech Stack

- **Vite** — Build tool and dev server
- **React 18** — UI framework
- **TypeScript** — Type safety
- **TensorFlow.js** — Machine learning runtime
- **@tensorflow-models/face-landmarks-detection** — MediaPipe FaceMesh model

## Facial Feature Colors

| Feature | Color |
|---------|-------|
| Face Oval | Gray |
| Left / Right Eye | Green |
| Left / Right Eyebrow | Gold |
| Nose | Red |
| Lips | Pink |
| Iris | Blue |

## Browser Support

Requires a browser with WebGL support and camera access (HTTPS or localhost).

- Chrome / Edge (recommended)
- Firefox
- Safari 16.4+
