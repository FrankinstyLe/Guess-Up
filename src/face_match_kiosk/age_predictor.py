"""Stable age inference for the kiosk's optional age lens."""

# System imports
import os

# Library imports
import cv2
import numpy as np
import onnxruntime as ort

# Our imports
from face_match_kiosk.configs import AGE_PREDICTOR_ONNX_PATH, AGE_INPUT_SIZE


class AgePredictor:
    """Run the existing trained age model without importing torch."""

    def __init__(self):
        if not os.path.exists(AGE_PREDICTOR_ONNX_PATH):
            raise RuntimeError(
                'Age model not found at %s' % AGE_PREDICTOR_ONNX_PATH)

        self.session = ort.InferenceSession(
            AGE_PREDICTOR_ONNX_PATH, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, face_crop):
        image = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, AGE_INPUT_SIZE, interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32))
        image /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return np.transpose(image, (2, 0, 1))[None, ...]

    def _predict_one(self, face_crop):
        output = self.session.run(None, {self.input_name: self._preprocess(face_crop)})[0]
        return float(np.clip(output[0][0] * 80.0, 0.0, 80.0))

    def predict(self, face_crop):
        """Return a median prediction from original, flipped, and brightened views."""
        views = [
            face_crop,
            cv2.flip(face_crop, 1),
            cv2.convertScaleAbs(face_crop, alpha=1.0, beta=12),
        ]
        return float(np.median([self._predict_one(view) for view in views]))
