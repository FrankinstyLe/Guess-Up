"""Occlusion sensitivity: which pixels actually moved the answer.

This is the Act 2 centerpiece, and it is deliberately NOT Grad-CAM. Grad-CAM
needs gradients, which needs torch, which would triple the kiosk's install size
for a laptop that has to boot reliably at a fair table. Occlusion sensitivity
needs only forward passes, so it runs on the same onnxruntime session the
matcher already uses.

It also explains itself in one sentence to a student with no ML background:
"we cover up part of your face and measure how much the answer moves."

Method: slide a gray patch over a GRID x GRID arrangement of positions, embed
all of the masked variants in ONE batched call, and score each cell by how far
the embedding moved (1 - cosine to the unmasked embedding). Bright cell = the
model was leaning on that region.
"""

# Library imports
import cv2
import numpy as np

# Our imports
from face_match_kiosk.configs import (
    OCCLUSION_GRID,
    OCCLUSION_PATCH_SCALE,
    OCCLUSION_FILL,
    OCCLUSION_ALPHA,
)


def sensitivity_map(crop, embedder, grid=OCCLUSION_GRID):
    """Return a (grid, grid) float32 map of how much each cell mattered.

    Values are raw cosine distances, not yet normalized, so callers can decide
    how to scale them for display.
    """
    baseline = embedder.embed(crop)

    size = crop.shape[0]
    cell = size / float(grid)
    patch = int(round(cell * OCCLUSION_PATCH_SCALE))
    half = patch // 2

    variants = []
    for row in range(grid):
        for column in range(grid):
            masked = crop.copy()

            center_y = int(round((row + 0.5) * cell))
            center_x = int(round((column + 0.5) * cell))

            y0, y1 = max(0, center_y - half), min(size, center_y + half)
            x0, x1 = max(0, center_x - half), min(size, center_x + half)
            masked[y0:y1, x0:x1] = OCCLUSION_FILL

            variants.append(masked)

    # One batched call for all grid*grid variants -- the whole reason this is
    # fast enough to feel interactive.
    embeddings = embedder.embed_batch(variants)

    distances = 1.0 - (embeddings @ baseline)
    return distances.reshape(grid, grid).astype(np.float32)


def heatmap_overlay(crop, sensitivity, alpha=OCCLUSION_ALPHA):
    """Blend a color heatmap of `sensitivity` onto `crop`. Returns BGR uint8."""
    lowest, highest = float(sensitivity.min()), float(sensitivity.max())

    if highest - lowest < 1e-8:
        normalized = np.zeros_like(sensitivity)
    else:
        normalized = (sensitivity - lowest) / (highest - lowest)

    # Upscale the coarse grid smoothly so it reads as a heat cloud, not blocks.
    smooth = cv2.resize(normalized, (crop.shape[1], crop.shape[0]),
                        interpolation=cv2.INTER_CUBIC)
    smooth = np.clip(smooth, 0.0, 1.0)

    colored = cv2.applyColorMap((smooth * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Weight the blend by intensity so cool regions stay recognizable as a face
    # and only the hot regions get strongly tinted.
    weight = (smooth * alpha)[:, :, None]
    blended = crop.astype(np.float32) * (1.0 - weight) + \
        colored.astype(np.float32) * weight

    return np.clip(blended, 0, 255).astype(np.uint8)


def peak_region(sensitivity):
    """Name the region the model leaned on hardest, for the on-screen caption.

    Rough thirds of an aligned ArcFace crop. The alignment makes this reliable
    enough for a one-line label: eyes land around y~0.46, mouth around y~0.82.
    """
    grid = sensitivity.shape[0]
    row, column = np.unravel_index(int(np.argmax(sensitivity)), sensitivity.shape)

    vertical = (row + 0.5) / grid
    horizontal = (column + 0.5) / grid

    if vertical < 0.34:
        band = 'forehead / hairline'
    elif vertical < 0.62:
        band = 'eyes'
    elif vertical < 0.80:
        band = 'nose'
    else:
        band = 'mouth / jaw'

    if horizontal < 0.33:
        side = 'left '
    elif horizontal > 0.67:
        side = 'right '
    else:
        side = ''

    return side + band
