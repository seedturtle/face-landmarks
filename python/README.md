# Python Face & Tongue Landmark Detection

Real-time facial feature detection and tongue tip tracking using Python, OpenCV, and dlib.

## Overview

This module provides Python-based face landmark detection and tongue tip tracking for research and clinical applications. It uses dlib's 68-point face landmarks model combined with OpenCV for real-time webcam and video processing.

## Features

- **Facial Landmark Detection** — 68-point face landmark detection using dlib
- **Mouth Region Extraction** — Automatic mouth/inner mouth ROI detection
- **Tongue Tip Tracking** — Real-time tongue tip position detection using ORB keypoints and blob detection
- **Mouth Aspect Ratio** — Mouth open/closed state detection via MAR calculation
- **Optical Flow** — Dense optical flow for tongue movement analysis (experimental)

## Scripts

| Script | Description |
|--------|-------------|
| `facial_landmarks.py` | Basic face landmark detection on static images |
| `face_utils.py` | Core utility functions (mouth detection, MAR, drawing) |
| `detect-tongue-tip-real-time.py` | Real-time tongue tip detection via webcam (recommended) |
| `detect-tongue-real-time-v2.py` | v2 — Improved mouth ROI with inner mouth height threshold |
| `detect-tongue-real-time-v4.py` | v4 — Rotated mouth detection with SimpleBlobDetector |
| `detedt-tongue-real-time-v1.py` | v1 — Early version with blob tracking |
| `detect-tongue.py` | Optical flow based tongue detection on video files |
| `detect-tongue-with-blob-tracking.py` | Blob tracking on video files |
| `detect-tongue-with-blob-tracking-revised.py` | Revised blob tracking with improved thresholds |
| `detect_face_parts.py` | Face parts detection and visualization |

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Face Landmark Model

Download the dlib shape predictor file and place it in the `python/` directory:

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

Or download manually from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

## Usage

### Real-time Tongue Tip Detection (Webcam)

```bash
python detect-tongue-tip-real-time.py
```

Press `q` to quit. The script will:
1. Open your webcam
2. Detect face landmarks using dlib
3. Extract mouth region
4. Track tongue tip position using ORB keypoints
5. Display results with mouth aspect ratio and tongue position

### Process Video File

```bash
python detect-tongue-with-blob-tracking-revised.py
```

Edit line 12 to change the input video path.

### Face Landmark Detection (Static Image)

```bash
python facial_landmarks.py --image path/to/image.jpg
```

## How It Works

1. **Face Detection** — dlib's HOG-based frontal face detector
2. **Landmark Detection** — 68 facial landmarks via dlib's shape predictor
3. **Mouth Extraction** — ROI extraction using mouth and inner mouth landmark indices
4. **Tongue Tip Detection** — ORB keypoint detection within the mouth region, selecting the lowest keypoint as the tongue tip
5. **MAR Calculation** — Mouth Aspect Ratio for open/closed state detection

## Requirements

- Python 3.7+
- dlib (requires CMake and C++ compiler)
- OpenCV
- imutils
- numpy
- scipy

See `requirements.txt` for specific versions.

## Notes

- The `shape_predictor_68_face_landmarks.dat` file (~100MB) is not included in the repository. Download it separately from the link above.
- Real-time scripts require a webcam. For USB webcams, adjust the camera index (default: 0).
- The `result/` and `test/` directories contain sample outputs and test data.
