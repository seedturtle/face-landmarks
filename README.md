# Face Landmarks Detection

Real-time facial feature detection and tongue tip tracking.

## Two Approaches

This repository contains two implementations:

### 1. Web App (Browser-based)

A real-time facial feature detection web app using TensorFlow.js and MediaPipe FaceMesh.

- **Location:** Root directory (Vite + React + TypeScript)
- **Features:** 468 facial landmarks, color-coded features, runs in browser
- **Deploy:** Zeabur-ready

### 2. Python Module (Desktop)

Python-based face landmark detection and tongue tip tracking using OpenCV and dlib.

- **Location:** `python/` directory
- **Features:** 68-point landmarks, tongue tip tracking, mouth aspect ratio
- **Use case:** Research, clinical applications, webcam/video processing

## Quick Start

### Web App

```bash
npm install
npm run dev
```

Open http://localhost:3000 and allow camera access.

### Python Module

```bash
cd python
pip install -r requirements.txt
# Download shape_predictor_68_face_landmarks.dat (see python/README.md)
python detect-tongue-tip-real-time.py
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Web App | Vite, React 18, TypeScript, TensorFlow.js, MediaPipe FaceMesh |
| Python | OpenCV, dlib, imutils, numpy, scipy |

## Facial Feature Colors (Web App)

| Feature | Color |
|---------|-------|
| Face Oval | Gray |
| Left / Right Eye | Green |
| Left / Right Eyebrow | Gold |
| Nose | Red |
| Lips | Pink |
| Iris | Blue |

## Deploy to Zeabur

1. Push this repository to GitHub
2. Go to [Zeabur.com](https://zeabur.com) and create a new project
3. Select your GitHub repo
4. Zeabur will auto-detect the Vite project and deploy it

The `zeabur.json` configuration is already included.

## Browser Support (Web App)

Requires WebGL and camera access (HTTPS or localhost):

- Chrome / Edge (recommended)
- Firefox
- Safari 16.4+

## License

See repository for license details.
