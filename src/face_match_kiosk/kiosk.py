"""The kiosk: state machine plus the four screens.

    ATTRACT  -> a stranger walks past and something big is moving on screen
    LOCKING  -> a face is held for a beat, so we do not match a blurry frame
    MATCH    -> Act 1. "You look like Grace Hopper." Fun, shareable, confident.
    REVEAL   -> Act 2. The same three scores, nearly tied, below the threshold a
                real face-ID system would need, with a heatmap of which pixels
                mattered and a live panel showing the answer flip as you move.

Design constraints that drove this, all of them from "it is a loud club fair":

  * No operator. Every state times out back to ATTRACT on its own.
  * Legible from two meters, so type is large and there is never more than one
    idea on screen at a time.
  * Never writes a frame to disk. That is a promise we make on the table sign,
    and it is also just the right default for pointing a camera at strangers.

Each screen is rendered in two passes: a `_base_*` method does all the OpenCV
drawing (scrim, panels, thumbnails, bars) and returns the canvas plus a layout
dict, then a `_text_*` method puts every label on through one shared
render.Painter. Splitting it this way is not stylistic -- one PIL round-trip per
text call cost ~5 ms each, and the REVEAL screen has eighteen labels.
"""

# System imports
import time
import threading
from collections import deque

# Library imports
import cv2
import numpy as np

# Our imports
from face_match_kiosk import gallery as gallery_module
from face_match_kiosk import matcher, occlusion, render
from face_match_kiosk.embedder import l2_normalize
from face_match_kiosk.configs import (
    WINDOW_NAME,
    CAPTURE_WIDTH,
    CAPTURE_HEIGHT,
    LENSES,
    CAMERA_BACKENDS,
    MATCH_IDLE_TIMEOUT_SECONDS,
    REVEAL_IDLE_TIMEOUT_SECONDS,
    FACE_LOST_GRACE_SECONDS,
    REVEAL_REEMBED_INTERVAL_SECONDS,
    SAME_PERSON_THRESHOLD,
    TOP_K,
    PROBE_FRAMES,
    COLOR_TEXT,
    COLOR_DIM,
    COLOR_ACCENT,
    COLOR_AMBER,
    COLOR_UH_RED,
    COLOR_BOX,
    COLOR_BG,
)


ATTRACT, LOCKING, MATCH, REVEAL = 'attract', 'locking', 'match', 'reveal'


class Kiosk:

    def __init__(self, detector, embedder, galleries, args):
        self.detector = detector
        self.embedder = embedder
        self.galleries = galleries
        self.args = args

        # Combining galleries costs a vstack, so do it once per lens, not per
        # frame. Must exist before the lens filter below, which calls _pool().
        self._pool_cache = {}

        # Only offer lenses whose gallery actually has faces in it, and whose
        # required label is populated -- a half-built gallery must never put a
        # broken screen in front of a student.
        self.lenses = [lens for lens in LENSES if self._lens_usable(lens)]
        if not self.lenses:
            raise RuntimeError(
                'No usable lens. Add photos under galleries/<name>/ and run: '
                'python scripts/build_gallery.py')

        requested = getattr(args, 'lens', None) or getattr(args, 'gallery', None)
        keys = [lens['key'] for lens in self.lenses]
        if requested in keys:
            self.lens_index = keys.index(requested)
        else:
            matched = [index for index, lens in enumerate(self.lenses)
                       if requested in self._lens_names(lens)]
            self.lens_index = matched[0] if matched else 0

        self.state = ATTRACT
        self.state_entered = time.monotonic()
        self.last_face_seen = 0.0

        self.frozen_frame = None
        self.frozen_crop = None
        self.matches = []
        self.sensitivity = None
        self.heat_overlay = None
        self.peak = ''

        self.live_matches = []
        self.live_last_embed = 0.0
        self.flip_count = 0

        self.teaser_index = 0
        self.capture = None

        # Rolling window of recent embeddings, averaged into the probe on SPACE.
        self.probe_samples = deque(maxlen=PROBE_FRAMES)

        # The occlusion heatmap costs ~1.5s with the r50 model, which would
        # freeze the window if computed when SPACE is pressed. Instead it is
        # built on a worker thread the moment MATCH is entered, so it is almost
        # always ready by the time anyone asks for it.
        self._heat_thread = None
        self._heat_lock = threading.Lock()

    # ------------------------------------------------------------ properties

    def _lens_names(self, lens):
        """Gallery folder(s) a lens draws on."""
        return lens.get('galleries') or [lens['gallery']]

    def _pool(self, lens):
        """The combined, rankable gallery for a lens."""
        key = lens['key']
        if key not in self._pool_cache:
            self._pool_cache[key] = gallery_module.combine(
                key, [self.galleries.get(name) for name in self._lens_names(lens)])
        return self._pool_cache[key]

    def _lens_usable(self, lens):
        """A lens needs a non-empty pool whose entries carry its label."""
        pool = self._pool(lens)
        if pool is None or len(pool) == 0:
            return False

        # A lens keyed on 'major' is useless if nobody has one recorded.
        field = lens['primary']
        return any(getattr(entry, field, '') for entry in pool.entries)

    @property
    def lens(self):
        return self.lenses[self.lens_index]

    @property
    def gallery(self):
        return self._pool(self.lens)

    def _label(self, entry, which):
        """Read a lens-configured field off an entry, falling back to the name."""
        return getattr(entry, self.lens[which], '') or entry.name

    @property
    def elapsed(self):
        return time.monotonic() - self.state_entered

    # ------------------------------------------------------------ lifecycle

    BACKEND_IDS = {
        'msmf': cv2.CAP_MSMF,
        'dshow': cv2.CAP_DSHOW,
        'any': cv2.CAP_ANY,
    }

    def _open_camera(self):
        """Open the camera on the fastest available backend. See CAMERA_BACKENDS."""
        requested = getattr(self.args, 'camera_backend', None)
        order = [requested] if requested else list(CAMERA_BACKENDS)

        for name in order:
            backend = self.BACKEND_IDS.get(name, cv2.CAP_ANY)
            capture = cv2.VideoCapture(self.args.camera, backend)

            if not capture.isOpened():
                capture.release()
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

            ok, _ = capture.read()
            if not ok:
                capture.release()
                continue

            self.camera_backend = name
            return capture

        raise RuntimeError(
            'Could not open camera %d on any of: %s. Close anything else using '
            'the webcam, or try --camera 1.'
            % (self.args.camera, ', '.join(order)))

    def _make_window(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if not self.args.no_fullscreen:
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def _read_frame(self, still):
        if still is not None:
            return still.copy()

        ok, frame = self.capture.read()
        if not ok:
            return None

        # Mirror, so moving left on screen matches moving left in real life.
        frame = cv2.flip(frame, 1)

        # Normalize to the render canvas size. Webcams frequently ignore the
        # resolution we ask for (a 720p request commonly comes back 640x480),
        # and since every layout value here is a fraction of the frame, drawing
        # into a small frame and letting fullscreen scale it up gives soft,
        # blurry type. Upscaling first keeps the text crisp.
        if frame.shape[1] != CAPTURE_WIDTH or frame.shape[0] != CAPTURE_HEIGHT:
            frame = cv2.resize(frame, (CAPTURE_WIDTH, CAPTURE_HEIGHT),
                               interpolation=cv2.INTER_LINEAR)

        return frame

    def run(self):
        still = None
        if self.args.image:
            still = cv2.imread(self.args.image, cv2.IMREAD_COLOR)
            if still is None:
                raise RuntimeError('Could not read --image %s' % self.args.image)
            still = cv2.resize(still, (CAPTURE_WIDTH, CAPTURE_HEIGHT))
        else:
            self.capture = self._open_camera()

        self._make_window()

        try:
            while True:
                frame = self._read_frame(still)
                if frame is None:
                    continue

                face = self.detector.detect_largest(frame)
                if face is not None:
                    self.last_face_seen = time.monotonic()

                self._advance(frame, face)
                canvas = self._render(frame, face)

                cv2.imshow(WINDOW_NAME, canvas)

                if not self._handle_key(cv2.waitKey(1) & 0xFF, frame, face):
                    break
        finally:
            if self.capture is not None:
                self.capture.release()
            cv2.destroyAllWindows()

    # ------------------------------------------------------------ state

    def _go(self, state):
        self.state = state
        self.state_entered = time.monotonic()

    @property
    def _face_recently_seen(self):
        return (time.monotonic() - self.last_face_seen) < FACE_LOST_GRACE_SECONDS

    def _advance(self, frame, face):
        if self.state == ATTRACT:
            if face is not None:
                self.probe_samples.clear()
                self._go(LOCKING)

        elif self.state == LOCKING:
            if not self._face_recently_seen:
                self._go(ATTRACT)
            else:
                # Keep the rolling window warm, but do not advance on our own --
                # the student decides when the photo is taken, by pressing SPACE.
                self._sample_probe(frame, face)

        elif self.state == MATCH:
            if self.elapsed >= MATCH_IDLE_TIMEOUT_SECONDS:
                self._go(ATTRACT)

        elif self.state == REVEAL:
            self._update_live(frame, face)
            if self.elapsed >= REVEAL_IDLE_TIMEOUT_SECONDS:
                self._go(ATTRACT)

    def _sample_probe(self, frame, face):
        """Bank one embedding into the rolling window."""
        if face is None:
            return
        self.probe_samples.append(
            self.embedder.embed(self.detector.align(frame, face)))

    def _probe_embedding(self, crop):
        """Mean of the countdown samples, re-normalized. Falls back to one frame."""
        if not self.probe_samples:
            return self.embedder.embed(crop)
        return l2_normalize(np.mean(list(self.probe_samples), axis=0))

    def _capture_match(self, frame, face):
        """Freeze the frame and rank the averaged probe against the pool."""
        self.frozen_frame = frame.copy()
        self.frozen_crop = self.detector.align(frame, face)

        self._sample_probe(frame, face)
        self.probe = self._probe_embedding(self.frozen_crop)
        self.matches = matcher.top_k(self.probe, self.gallery, TOP_K)

        self._start_heatmap(self.frozen_crop)

        # Cleared here so entering REVEAL recomputes them for this face.
        self.sensitivity = None
        self.heat_overlay = None
        self.peak = ''
        self.live_matches = []
        self.flip_count = 0

    def _start_heatmap(self, crop):
        """Kick off the occlusion heatmap on a worker thread."""
        def work(target):
            sensitivity = occlusion.sensitivity_map(target, self.embedder)
            overlay = occlusion.heatmap_overlay(target, sensitivity)
            peak = occlusion.peak_region(sensitivity)

            with self._heat_lock:
                # Drop the result if the student already walked off and a new
                # face was captured in the meantime.
                if self.frozen_crop is target:
                    self.sensitivity = sensitivity
                    self.heat_overlay = overlay
                    self.peak = peak

        self._heat_thread = threading.Thread(target=work, args=(crop,), daemon=True)
        self._heat_thread.start()

    def _prepare_reveal(self):
        """Make sure the heatmap exists, computing it inline only if we must."""
        if self.frozen_crop is None or self.sensitivity is not None:
            return

        if self._heat_thread is not None and self._heat_thread.is_alive():
            # Give the worker a moment; REVEAL renders without it if not ready.
            self._heat_thread.join(timeout=0.4)

    def _update_live(self, frame, face):
        """Re-match the live face on a throttle, to show the answer flipping."""
        if face is None:
            return

        now = time.monotonic()
        if now - self.live_last_embed < REVEAL_REEMBED_INTERVAL_SECONDS:
            return
        self.live_last_embed = now

        crop = self.detector.align(frame, face)
        embedding = self.embedder.embed(crop)
        fresh = matcher.top_k(embedding, self.gallery, TOP_K)

        if fresh and self.live_matches and fresh[0].name != self.live_matches[0].name:
            self.flip_count += 1

        self.live_matches = fresh

    # ------------------------------------------------------------ input

    def _handle_key(self, key, frame, face):
        """Returns False to quit."""
        if key in (ord('q'), 27):
            return False

        if key == ord('r'):
            self._go(ATTRACT)

        elif key == ord(' '):
            if self.state == MATCH:
                self._prepare_reveal()
                self._go(REVEAL)
            elif self.state == REVEAL:
                self._go(MATCH)
            elif self.state in (ATTRACT, LOCKING) and face is not None:
                # Impatient student: skip the countdown.
                self._capture_match(frame, face)
                self._go(MATCH)

        elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
            index = key - ord('1')
            if index < len(self.lenses):
                self.lens_index = index
                # Re-rank the frozen face against the newly selected lens. Two
                # lenses over the same gallery give identical scores, so this is
                # only real work when the gallery differs.
                if self.frozen_crop is not None and self.state in (MATCH, REVEAL):
                    probe = getattr(self, 'probe', None)
                    if probe is None:
                        probe = self._probe_embedding(self.frozen_crop)
                    self.matches = matcher.top_k(probe, self.gallery, TOP_K)
                    self.live_matches = []
                    self._go(MATCH)

        elif key == ord('f'):
            if self.state in (ATTRACT, LOCKING) and face is not None:
                self._capture_match(frame, face)
                self._go(MATCH)

        return True

    # ------------------------------------------------------------ render

    def _render(self, frame, face):
        """OpenCV pass, then one batched text pass. See the module docstring."""
        if self.state == ATTRACT:
            canvas, layout = self._base_attract(frame)
        elif self.state == LOCKING:
            canvas, layout = self._base_locking(frame, face)
        elif self.state == MATCH:
            canvas, layout = self._base_match()
        else:
            canvas, layout = self._base_reveal(frame)

        canvas = self._base_footer(canvas)

        painter = render.Painter(canvas)

        if self.state == ATTRACT:
            self._text_attract(painter, canvas.shape, layout)
        elif self.state == LOCKING:
            self._text_locking(painter, canvas.shape, layout)
        elif self.state == MATCH:
            self._text_match(painter, canvas.shape, layout)
        else:
            self._text_reveal(painter, canvas.shape, layout)

        self._text_footer(painter, canvas.shape)

        return painter.done()

    def _title_for_lens(self):
        return self.lens.get('title', 'Who do you look like?')

    # -- shared chrome

    def _base_footer(self, canvas):
        height, width = canvas.shape[:2]
        bar = max(46, height // 16)

        canvas[height - bar:, :] = cv2.addWeighted(
            canvas[height - bar:, :], 0.25,
            np.full((bar, width, 3), COLOR_BG, dtype=np.uint8), 0.75, 0)

        return canvas

    def _text_footer(self, painter, shape):
        height, width = shape[:2]
        bar = max(46, height // 16)
        size = max(15, bar // 3)
        middle = height - bar // 2

        painter.text('COUGAR AI', (24, middle), size=size,
                     color=COLOR_UH_RED, anchor='lm')

        labels = ' / '.join(
            ('[%d] %s' % (index + 1, lens['menu'])).upper()
            if index == self.lens_index
            else '[%d] %s' % (index + 1, lens['menu'])
            for index, lens in enumerate(self.lenses))
        painter.text(labels, (width // 2, middle), size=size,
                     color=COLOR_DIM, anchor='mm')

        painter.text('no photos are saved', (width - 24, middle), size=size,
                     color=COLOR_DIM, anchor='rm')

    # -- ATTRACT

    def _base_attract(self, frame):
        return render.scrim(frame, alpha=0.68), {}

    def _text_attract(self, painter, shape, layout):
        height, width = shape[:2]

        # Slow pulse so the screen is always moving; movement is what catches an
        # eye from across a crowded room.
        pulse = 0.5 + 0.5 * np.sin(time.monotonic() * 2.0)

        painter.text('WHO DO YOU', (width // 2, int(height * 0.26)),
                     size=int(height * 0.13), color=COLOR_TEXT, anchor='mm')
        painter.text('LOOK LIKE?', (width // 2, int(height * 0.40)),
                     size=int(height * 0.13), color=COLOR_UH_RED, anchor='mm')

        shade = int(120 + 135 * pulse)
        painter.text('step in front of the camera',
                     (width // 2, int(height * 0.56)),
                     size=int(height * 0.055), color=(shade, shade, shade),
                     anchor='mm')

        # Cycle gallery names as a teaser. Deliberately NOT past visitors'
        # results -- nothing about a visitor outlives their turn.
        entries = self.gallery.entries
        if entries:
            self.teaser_index = int(time.monotonic() / 1.6) % len(entries)
            painter.text('today: %s' % self._label(entries[self.teaser_index],
                                                   'primary'),
                         (width // 2, int(height * 0.68)),
                         size=int(height * 0.042), color=COLOR_ACCENT, anchor='mm')

        painter.text(self._title_for_lens(),
                     (width // 2, int(height * 0.79)),
                     size=int(height * 0.036), color=COLOR_DIM, anchor='mm')

    # -- LOCKING

    def _base_locking(self, frame, face):
        canvas = render.scrim(frame, alpha=0.22)

        if face is not None:
            cv2.rectangle(canvas, (face.x, face.y),
                          (face.x + face.w, face.y + face.h), COLOR_BOX, 3)

        return canvas, {}

    def _text_locking(self, painter, shape, layout):
        height, width = shape[:2]

        # Pulse the prompt so it reads as "waiting for you" rather than frozen.
        pulse = 0.5 + 0.5 * np.sin(time.monotonic() * 3.2)
        shade = int(150 + 105 * pulse)

        painter.text('press SPACE', (width // 2, int(height * 0.44)),
                     size=int(height * 0.15), color=(shade, shade, shade),
                     anchor='mm')
        painter.text('to take the photo', (width // 2, int(height * 0.60)),
                     size=int(height * 0.050), color=COLOR_ACCENT, anchor='mm')

    # -- MATCH

    def _base_match(self):
        if not self.matches:
            return render.scrim(self.frozen_frame, alpha=0.5), {}

        canvas = render.scrim(self.frozen_frame, alpha=0.55)
        height, width = canvas.shape[:2]

        split = int(width * 0.46)

        thumb_size = int(min(height * 0.40, split * 0.62))
        thumb_x = split // 2 - thumb_size // 2
        thumb_y = int(height * 0.16)

        canvas = render.paste(
            canvas, render.load_thumbnail(self.matches[0].entry, thumb_size),
            (thumb_x, thumb_y), border=4, border_color=COLOR_UH_RED)

        small = int(min(height * 0.15, width * 0.10))
        runner_x = split + int(width * 0.04)
        runner_rows = []

        for index, match in enumerate(self.matches[1:3]):
            row_y = int(height * 0.19) + index * int(small * 1.28)
            canvas = render.paste(
                canvas, render.load_thumbnail(match.entry, small),
                (runner_x, row_y), border=2, border_color=COLOR_DIM)
            runner_rows.append((match, row_y))

        return canvas, {
            'split': split,
            'thumb_size': thumb_size,
            'thumb_y': thumb_y,
            'small': small,
            'runner_x': runner_x,
            'runner_rows': runner_rows,
        }

    def _text_match(self, painter, shape, layout):
        if not self.matches or not layout:
            return

        height, width = shape[:2]
        split = layout['split']
        best = self.matches[0]

        # The headline is the whole difference between the two lenses:
        # "you look like <Grace Hopper>" vs
        # "you look like someone who majors in <Mathematics & Physics>".
        painter.wrapped(self.lens['headline'], (split // 2, int(height * 0.085)),
                        int(height * 0.042), int(split * 0.90),
                        color=COLOR_DIM, anchor='mm')

        primary = self._label(best.entry, 'primary')
        secondary = self._label(best.entry, 'secondary')

        primary_size = int(height * 0.075)
        while (primary_size > int(height * 0.034)
               and render.text_size(primary, primary_size)[0] > split * 0.82):
            primary_size = int(primary_size * 0.90)

        primary_y = layout['thumb_y'] + layout['thumb_size'] + int(height * 0.07)
        painter.text(primary, (split // 2, primary_y), size=primary_size,
                     color=COLOR_TEXT, anchor='mm')

        if secondary and secondary != primary:
            painter.wrapped(secondary,
                            (split // 2, primary_y + int(height * 0.07)),
                            int(height * 0.038), int(split * 0.86),
                            color=COLOR_ACCENT, anchor='mm')

        painter.text('also close', (layout['runner_x'], int(height * 0.13)),
                     size=int(height * 0.040), color=COLOR_DIM, anchor='lm')

        # Dedupe against every label already on screen, not just the winner's:
        # with 58 faces and 30 majors, #2 and #3 landing on the same field is
        # common, and three identical lines read as a bug rather than as three
        # candidates.
        shown = {self._label(best.entry, 'primary')}
        for match, row_y in layout['runner_rows']:
            label = self._label(match.entry, 'primary')
            if label in shown:
                label = self._label(match.entry, 'secondary')
            shown.add(label)
            painter.text(
                label,
                (layout['runner_x'] + layout['small'] + 18,
                 row_y + layout['small'] // 2),
                size=int(height * 0.042), color=COLOR_TEXT, anchor='lm')

        hint_x = split + int(width * 0.27)
        hint_y = int(height * 0.80)
        painter.text('press SPACE', (hint_x, hint_y),
                     size=int(height * 0.062), color=COLOR_AMBER, anchor='mm')
        painter.text('how did it decide?', (hint_x, hint_y + int(height * 0.06)),
                     size=int(height * 0.040), color=COLOR_DIM, anchor='mm')

    # -- REVEAL

    def _base_reveal(self, frame):
        canvas = render.scrim(self.frozen_frame, alpha=0.88)
        height, width = canvas.shape[:2]

        layout = {}

        # (a) the heatmap: which pixels moved the answer
        face_size = int(min(height * 0.34, width * 0.24))
        heat_x, heat_y = int(width * 0.04), int(height * 0.14)

        if self.heat_overlay is not None:
            canvas = render.paste(
                canvas,
                cv2.resize(self.heat_overlay, (face_size, face_size),
                           interpolation=cv2.INTER_LINEAR),
                (heat_x, heat_y), border=2, border_color=COLOR_DIM)

        layout.update(face_size=face_size, heat_x=heat_x, heat_y=heat_y)

        # (b) the score bars, on an axis zoomed to just above the top score
        left = int(width * 0.33)
        bar_width = int(width * 0.21)
        top = int(height * 0.16)
        bar_height = int(height * 0.055)
        step = int(height * 0.105)

        top_score = max((match.score for match in self.matches), default=0.1)
        axis_max = max(top_score * 1.18, 0.02)

        for index, match in enumerate(self.matches):
            row_y = top + index * step
            canvas = render.score_bar(
                canvas, (left, row_y + int(height * 0.008)), bar_width, bar_height,
                match.score / axis_max,
                COLOR_UH_RED if index == 0 else COLOR_DIM)

        layout.update(left=left, bar_width=bar_width, top=top,
                      bar_height=bar_height, step=step)

        # (c) the live panel, where the answer visibly flips
        panel_x = int(width * 0.68)
        panel_w = int(width * 0.28)
        canvas = render.panel(
            canvas, (panel_x, int(height * 0.13)),
            (panel_x + panel_w, int(height * 0.80)), alpha=0.55)

        live_size = int(min(panel_w * 0.62, height * 0.26))
        live_y = int(height * 0.28)
        canvas = render.paste(
            canvas, render.fit_thumbnail(frame, live_size),
            (panel_x + panel_w // 2 - live_size // 2, live_y),
            border=2, border_color=COLOR_ACCENT)

        layout.update(panel_x=panel_x, panel_w=panel_w,
                      live_size=live_size, live_y=live_y)

        return canvas, layout

    def _text_reveal(self, painter, shape, layout):
        height, width = shape[:2]

        painter.text('it was never sure',
                     (int(width * 0.04), int(height * 0.07)),
                     size=int(height * 0.070), color=COLOR_AMBER, anchor='lm')

        # (a) heatmap captions
        if self.heat_overlay is not None:
            below = layout['heat_y'] + layout['face_size']
            painter.text('it leaned on your %s' % self.peak,
                         (layout['heat_x'], below + int(height * 0.04)),
                         size=int(height * 0.034), color=COLOR_ACCENT, anchor='lm')
            painter.text('red = what it used most',
                         (layout['heat_x'], below + int(height * 0.085)),
                         size=int(height * 0.028), color=COLOR_DIM, anchor='lm')

        # (b) score labels
        left, top, step = layout['left'], layout['top'], layout['step']
        bar_width, bar_height = layout['bar_width'], layout['bar_height']

        for index, match in enumerate(self.matches):
            row_y = top + index * step
            painter.text('%d. %s' % (index + 1, match.name), (left, row_y),
                         size=int(height * 0.036), color=COLOR_TEXT, anchor='ls')
            painter.text(
                '%.3f' % match.score,
                (left + bar_width + 14,
                 row_y + int(height * 0.008) + bar_height // 2),
                size=int(height * 0.034), color=COLOR_TEXT, mono=True, anchor='lm')

        summary_y = top + len(self.matches) * step + int(height * 0.02)

        painter.text('#1 beat #2 by %.3f' % matcher.spread(self.matches),
                     (left, summary_y), size=int(height * 0.040),
                     color=COLOR_AMBER, anchor='ls')

        # The honest headline: none of these clear the bar a real system uses.
        painter.wrapped(
            'A real face-ID system needs %.2f to call two photos the same '
            'person. Every score here is far below that.' % SAME_PERSON_THRESHOLD,
            (left, summary_y + int(height * 0.055)),
            int(height * 0.030), int(width * 0.30),
            color=COLOR_DIM, anchor='ls')

        # (c) live panel labels
        self._text_live_panel(painter, shape, layout)

    def _text_live_panel(self, painter, shape, layout):
        height, width = shape[:2]
        panel_x, panel_w = layout['panel_x'], layout['panel_w']
        center_x = panel_x + panel_w // 2

        painter.text('move your head', (center_x, int(height * 0.18)),
                     size=int(height * 0.042), color=COLOR_TEXT, anchor='mm')
        painter.text('watch it change its mind', (center_x, int(height * 0.235)),
                     size=int(height * 0.030), color=COLOR_DIM, anchor='mm')

        label_y = layout['live_y'] + layout['live_size'] + int(height * 0.06)

        if self.live_matches:
            live_best = self.live_matches[0]
            changed = bool(self.matches) and live_best.name != self.matches[0].name

            painter.text('now: ', (panel_x + int(panel_w * 0.08), label_y),
                         size=int(height * 0.032), color=COLOR_DIM, anchor='lm')

            name_size = int(height * 0.044)
            if render.text_size(live_best.name, name_size)[0] > panel_w * 0.86:
                name_size = int(name_size * 0.70)

            painter.text(live_best.name, (center_x, label_y + int(height * 0.05)),
                         size=name_size,
                         color=COLOR_AMBER if changed else COLOR_TEXT, anchor='mm')

            if self.flip_count:
                painter.text(
                    'changed its mind %d time%s'
                    % (self.flip_count, '' if self.flip_count == 1 else 's'),
                    (center_x, label_y + int(height * 0.12)),
                    size=int(height * 0.030), color=COLOR_AMBER, anchor='mm')
        else:
            painter.text('step back into frame',
                         (center_x, label_y + int(height * 0.03)),
                         size=int(height * 0.030), color=COLOR_DIM, anchor='mm')

        painter.text('SPACE to go back', (center_x, int(height * 0.76)),
                     size=int(height * 0.028), color=COLOR_DIM, anchor='mm')
