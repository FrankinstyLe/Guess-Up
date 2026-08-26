"""Drawing helpers for the kiosk.

cv2.putText only has bitmap fonts and looks like a debugging overlay, which is
the wrong impression to give at a recruiting table. So all text goes through PIL
with a real TrueType face, cached per size.

Everything here works on BGR uint8 numpy arrays (OpenCV's native layout) and
returns the same, so the kiosk can mix these calls with plain cv2 drawing.

All text goes through Painter, which batches a whole screen's labels into one
buffer conversion. Do the cv2 drawing first, then open a Painter for the text.
"""

# System imports
import os
import functools

# Library imports
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Our imports
from face_match_kiosk.configs import (
    FONT_CANDIDATES,
    FONT_MONO_CANDIDATES,
    COLOR_TEXT,
    COLOR_DIM,
    COLOR_BG,
)


# ---------------------------------------------------------------- fonts

@functools.lru_cache(maxsize=64)
def font(size, mono=False):
    """Load a TrueType font at `size`, falling back through the candidate list."""
    for path in (FONT_MONO_CANDIDATES if mono else FONT_CANDIDATES):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    # Last resort: PIL's builtin bitmap font. Ugly but never crashes.
    return ImageFont.load_default()


def text_size(message, size, mono=False):
    """(width, height) of `message` when drawn at `size`."""
    face = font(size, mono)
    left, top, right, bottom = face.getbbox(message)
    return right - left, bottom - top


# ---------------------------------------------------------------- text

class Painter:
    """Accumulates many text draws into a single PIL round-trip.

    Each standalone draw_text call costs a full BGR->RGB->PIL->numpy->BGR
    conversion of the whole frame (~5 ms at 1280x720). A screen with eighteen
    labels on it therefore spent ~95 ms per frame just converting buffers.
    Painter converts once on entry, draws everything, and converts once on exit.

    Do all cv2 drawing (scrims, panels, pastes, bars) BEFORE opening a Painter,
    since those write to the numpy array the Painter has already copied from.
    """

    def __init__(self, frame):
        self._image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._draw = ImageDraw.Draw(self._image)

    def text(self, message, origin, size=32, color=COLOR_TEXT, mono=False,
             anchor='lt', shadow=True):
        face = font(size, mono)
        rgb = (color[2], color[1], color[0])

        if shadow:
            offset = max(2, size // 18)
            self._draw.text((origin[0] + offset, origin[1] + offset), message,
                            font=face, fill=(0, 0, 0), anchor=anchor)

        self._draw.text(origin, message, font=face, fill=rgb, anchor=anchor)
        return self

    def wrapped(self, message, origin, size, max_width, color=COLOR_TEXT,
                anchor='mm', line_spacing=1.32, mono=False):
        """Greedy word wrap; PIL has no width-fitting multiline draw."""
        words = message.split()
        lines, current = [], ''

        for word in words:
            candidate = (current + ' ' + word).strip()
            if text_size(candidate, size, mono)[0] <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)

        x, y = origin
        step = int(size * line_spacing)
        for index, line in enumerate(lines):
            self.text(line, (x, y + index * step), size=size, color=color,
                      anchor=anchor, mono=mono)

        return self

    def done(self):
        return cv2.cvtColor(np.array(self._image), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- panels

def scrim(frame, alpha=0.55, color=COLOR_BG):
    """Darken the whole frame so overlaid text reads cleanly."""
    layer = np.full_like(frame, color, dtype=np.uint8)
    return cv2.addWeighted(frame, 1.0 - alpha, layer, alpha, 0)


def panel(frame, top_left, bottom_right, alpha=0.72, color=COLOR_BG, radius=18):
    """Rounded translucent panel, drawn in place on a copy of the region."""
    x0, y0 = top_left
    x1, y1 = bottom_right

    height, width = frame.shape[:2]
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return frame

    output = frame.copy()

    # Build a rounded-rect mask, then blend only inside it.
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (x1 - x0 - radius, y1 - y0), 255, -1)
    cv2.rectangle(mask, (0, radius), (x1 - x0, y1 - y0 - radius), 255, -1)
    for corner in ((radius, radius), (x1 - x0 - radius, radius),
                   (radius, y1 - y0 - radius), (x1 - x0 - radius, y1 - y0 - radius)):
        cv2.circle(mask, corner, radius, 255, -1)

    region = output[y0:y1, x0:x1]
    tinted = cv2.addWeighted(
        region, 1.0 - alpha, np.full_like(region, color, dtype=np.uint8), alpha, 0)

    selected = mask.astype(bool)
    region[selected] = tinted[selected]

    return output


# ---------------------------------------------------------------- images

def fit_thumbnail(image, size):
    """Center-crop to square then resize to size x size."""
    height, width = image.shape[:2]
    side = min(height, width)

    y0 = (height - side) // 2
    x0 = (width - side) // 2
    square = image[y0:y0 + side, x0:x0 + side]

    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


def paste(frame, patch, top_left, border=0, border_color=COLOR_DIM):
    """Paste `patch` onto `frame`, clipped at the frame edges."""
    x, y = top_left
    height, width = patch.shape[:2]

    frame_height, frame_width = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame_width, x + width), min(frame_height, y + height)
    if x1 <= x0 or y1 <= y0:
        return frame

    frame[y0:y1, x0:x1] = patch[y0 - y:y1 - y, x0 - x:x1 - x]

    if border:
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), border_color, border)

    return frame


def face_crop(image, face_box, margin=0.55):
    """Square crop centered on a detected face box, with headroom around it."""
    x, y, w, h = face_box
    center_x, center_y = x + w / 2.0, y + h / 2.0

    side = max(w, h) * (1.0 + 2 * margin)
    height, width = image.shape[:2]
    side = min(side, min(height, width))
    half = side / 2.0

    # Keep the crop inside the image without letting it drift off the face.
    center_x = min(max(center_x, half), width - half)
    center_y = min(max(center_y, half), height - half)

    x0, y0 = int(round(center_x - half)), int(round(center_y - half))
    x1, y1 = int(round(center_x + half)), int(round(center_y + half))

    crop = image[max(0, y0):y1, max(0, x0):x1]
    return crop if crop.size else image


def load_thumbnail(entry, size):
    """Read and cache a square thumbnail on the gallery entry itself."""
    if entry.thumbnail is not None and entry.thumbnail.shape[0] == size:
        return entry.thumbnail

    image = cv2.imread(entry.path, cv2.IMREAD_COLOR)
    if image is None:
        image = np.full((size, size, 3), 60, dtype=np.uint8)
    elif getattr(entry, 'face_box', None):
        image = face_crop(image, entry.face_box)

    entry.thumbnail = fit_thumbnail(image, size)
    return entry.thumbnail


# ---------------------------------------------------------------- bars

def score_bar(frame, top_left, width, height, fraction, color,
              track_color=(58, 54, 78)):
    """Horizontal bar. `fraction` is clamped to [0, 1]."""
    x, y = top_left
    fraction = float(np.clip(fraction, 0.0, 1.0))

    cv2.rectangle(frame, (x, y), (x + width, y + height), track_color, -1)
    filled = int(round(width * fraction))
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + height), color, -1)

    return frame
