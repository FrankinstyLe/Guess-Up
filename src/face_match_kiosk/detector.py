"""Face detection + ArcFace-style alignment.

Primary path is YuNet (cv2.FaceDetectorYN), which gives us five facial landmarks
per face. Those landmarks matter: warping the face onto a canonical template
before embedding is worth a lot of accuracy, and it is the difference between
"this match feels plausible" and "this match feels random".

The Haar cascade from the original age-predictor app is kept as a fallback so a
missing YuNet file degrades the demo instead of killing it. Haar gives no
landmarks, so that path can only do a plain padded crop.
"""

# System imports
import os

# Library imports
import cv2
import numpy as np

# Our imports
from face_match_kiosk.configs import (
    DETECTOR_ONNX_PATH,
    HAAR_CASCADE_PATH,
    DETECTOR_INPUT_SIZE,
    DETECTOR_SCORE_THRESHOLD,
    DETECTOR_NMS_THRESHOLD,
    DETECTOR_TOP_K,
    DETECT_MAX_WIDTH,
    EMBED_INPUT_SIZE,
    AGE_INPUT_SIZE,
)


# Canonical 5-point template ArcFace was trained against, for a 112x112 crop.
# Order: right eye, left eye, nose, right mouth corner, left mouth corner --
# as they appear left-to-right in the image, which is also YuNet's output order.
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


class Face:
    """One detected face: pixel box, optional landmarks, detector confidence."""

    def __init__(self, box, landmarks=None, score=1.0):
        self.x, self.y, self.w, self.h = [int(round(v)) for v in box]
        self.landmarks = landmarks
        self.score = float(score)

    @property
    def area(self):
        return self.w * self.h

    @property
    def box(self):
        return (self.x, self.y, self.w, self.h)

    @property
    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)


class FaceDetector:

    def __init__(self):
        self.backend = None
        self.yunet = None
        self.haar = None
        self._input_size = None

        if os.path.exists(DETECTOR_ONNX_PATH) and hasattr(cv2, 'FaceDetectorYN'):
            self.yunet = cv2.FaceDetectorYN.create(
                model=DETECTOR_ONNX_PATH,
                config='',
                input_size=DETECTOR_INPUT_SIZE,
                score_threshold=DETECTOR_SCORE_THRESHOLD,
                nms_threshold=DETECTOR_NMS_THRESHOLD,
                top_k=DETECTOR_TOP_K,
            )
            self._input_size = DETECTOR_INPUT_SIZE
            self.backend = 'yunet'
        elif os.path.exists(HAAR_CASCADE_PATH):
            self.haar = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
            self.backend = 'haar'
        else:
            raise RuntimeError(
                'No face detector available. Run scripts/fetch_models.py to get '
                'YuNet, or restore %s' % HAAR_CASCADE_PATH)

    # ------------------------------------------------------------ detection

    def detect(self, frame):
        """Return a list of Face, largest first."""
        if self.backend == 'yunet':
            faces = self._detect_yunet(frame)
        else:
            faces = self._detect_haar(frame)

        faces.sort(key=lambda face: face.area, reverse=True)
        return faces

    def detect_largest(self, frame):
        faces = self.detect(frame)
        return faces[0] if faces else None

    def _detect_yunet(self, frame):
        # Detect on a downscaled copy; see DETECT_MAX_WIDTH for why.
        scale = 1.0
        if frame.shape[1] > DETECT_MAX_WIDTH:
            scale = DETECT_MAX_WIDTH / float(frame.shape[1])
            small = cv2.resize(
                frame, (DETECT_MAX_WIDTH, int(round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_LINEAR)
        else:
            small = frame

        height, width = small.shape[:2]

        # YuNet needs to be told its input size, and rebuilding that on every
        # frame is wasteful, so only push it when the size actually changes.
        if self._input_size != (width, height):
            self.yunet.setInputSize((width, height))
            self._input_size = (width, height)

        _, detections = self.yunet.detect(small)
        if detections is None:
            return []

        inverse = 1.0 / scale

        faces = []
        for detection in detections:
            box = detection[0:4] * inverse
            landmarks = detection[4:14].reshape(5, 2).astype(np.float32) * inverse
            faces.append(Face(box, landmarks, detection[14]))
        return faces

    def _detect_haar(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.haar.detectMultiScale(gray_frame, 1.1, 5, minSize=(80, 80))
        return [Face(box) for box in boxes]

    # ------------------------------------------------------------ alignment

    def align(self, frame, face):
        """Warp a detected face to the canonical EMBED_INPUT_SIZE square crop."""
        if face.landmarks is not None:
            return self._align_by_landmarks(frame, face)
        return self._align_by_box(frame, face)

    def _align_by_landmarks(self, frame, face):
        scale = EMBED_INPUT_SIZE / 112.0
        transform, _ = cv2.estimateAffinePartial2D(
            face.landmarks, ARCFACE_TEMPLATE * scale, method=cv2.LMEDS)

        if transform is None:
            return self._align_by_box(frame, face)

        return cv2.warpAffine(
            frame, transform, (EMBED_INPUT_SIZE, EMBED_INPUT_SIZE),
            flags=cv2.INTER_LINEAR, borderValue=0)

    def _align_by_box(self, frame, face):
        """Landmark-free fallback: pad the box a little and resize."""
        height, width = frame.shape[:2]
        pad = int(0.15 * max(face.w, face.h))

        x0 = max(0, face.x - pad)
        y0 = max(0, face.y - pad)
        x1 = min(width, face.x + face.w + pad)
        y1 = min(height, face.y + face.h + pad)

        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return np.zeros((EMBED_INPUT_SIZE, EMBED_INPUT_SIZE, 3), dtype=np.uint8)

        return cv2.resize(crop, (EMBED_INPUT_SIZE, EMBED_INPUT_SIZE),
                          interpolation=cv2.INTER_AREA)

    def crop_for_age(self, frame, face):
        """Crop a little context around the face for the age model."""
        height, width = frame.shape[:2]
        pad = int(0.15 * max(face.w, face.h))
        x0 = max(0, face.x - pad)
        y0 = max(0, face.y - pad)
        x1 = min(width, face.x + face.w + pad)
        y1 = min(height, face.y + face.h + pad)
        crop = frame[y0:y1, x0:x1]
        return crop if crop.size else np.zeros(
            (AGE_INPUT_SIZE[1], AGE_INPUT_SIZE[0], 3), dtype=np.uint8)
