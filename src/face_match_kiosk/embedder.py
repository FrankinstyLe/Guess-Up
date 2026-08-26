"""Face embedding via ArcFace/MobileFaceNet on onnxruntime.

Two design notes worth keeping in mind if you edit this:

1. The InferenceSession is built ONCE, in __init__. The original age-predictor
   app rebuilt a 37 MB session on every single prediction, which is why it felt
   sluggish; at 30 fps that mistake is fatal.

2. embed_batch exists specifically so the occlusion heatmap can push all 36
   masked variants through in one call (~0.19s) instead of 36 calls.
"""

# Library imports
import os

import cv2
import numpy as np
import onnxruntime

# Our imports
from face_match_kiosk.configs import (
    EMBEDDER_CANDIDATES,
    EMBED_INPUT_SIZE,
    EMBED_MEAN,
    EMBED_SCALE,
    EMBED_DIM,
)


class Embedder:

    def __init__(self, onnx_path=None):
        # Take the best model that is actually present, so a laptop that only
        # got the small one still runs instead of refusing to start.
        if onnx_path is None:
            onnx_path = next(
                (path for path in EMBEDDER_CANDIDATES if os.path.exists(path)), None)

        if onnx_path is None or not os.path.exists(onnx_path):
            raise RuntimeError(
                'No embedder model in %s.\nRun: python scripts/fetch_models.py'
                % os.path.dirname(EMBEDDER_CANDIDATES[0]))

        self.name = os.path.basename(onnx_path)

        # The model declares a static [1, 512] output but computes variable
        # batches correctly, so onnxruntime logs a shape warning per call.
        # Severity 3 = errors only, which keeps the kiosk console clean.
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = \
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = onnxruntime.InferenceSession(
            onnx_path, sess_options=options, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.dim = EMBED_DIM

    # ------------------------------------------------------------ preprocess

    def _preprocess(self, crops):
        """uint8 BGR HWC crops -> float32 NCHW scaled to roughly [-1, 1]."""
        batch = np.empty(
            (len(crops), 3, EMBED_INPUT_SIZE, EMBED_INPUT_SIZE), dtype=np.float32)

        for index, crop in enumerate(crops):
            if crop.shape[0] != EMBED_INPUT_SIZE or crop.shape[1] != EMBED_INPUT_SIZE:
                crop = cv2.resize(crop, (EMBED_INPUT_SIZE, EMBED_INPUT_SIZE),
                                  interpolation=cv2.INTER_AREA)
            batch[index] = np.transpose(
                (crop.astype(np.float32) - EMBED_MEAN) / EMBED_SCALE, (2, 0, 1))

        return batch

    # ------------------------------------------------------------ inference

    def embed_batch(self, crops):
        """Embed a list of aligned crops. Returns (N, 512) L2-normalized."""
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)

        raw = self.session.run(
            None, {self.input_name: self._preprocess(crops)})[0]
        return l2_normalize(raw.astype(np.float32))

    def embed(self, crop):
        """Embed one aligned crop. Returns a (512,) L2-normalized vector."""
        return self.embed_batch([crop])[0]


def l2_normalize(matrix, epsilon=1e-10):
    """Row-wise L2 normalize, so a dot product IS the cosine similarity."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, epsilon)
