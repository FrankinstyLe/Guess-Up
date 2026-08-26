"""Configuration for the Cougar AI fair kiosk.

Every path here is anchored to this file's own location, NOT the working
directory. The older age_predictor_cnn/configs.py uses relative paths, which is
why the README has to tell you to `cd src` first. The kiosk gets launched from
a desktop shortcut at a fair, so it must work from any cwd.
"""

# System imports
import os


# ---------------------------------------------------------------- paths

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(PACKAGE_DIR)
REPO_DIR = os.path.dirname(SRC_DIR)

WEIGHTS_DIR = os.path.join(PACKAGE_DIR, 'weights')
DETECTOR_ONNX_PATH = os.path.join(WEIGHTS_DIR, 'face_detection_yunet_2023mar.onnx')
# Recognition model, best first. w600k_r50 (166 MB) measurably beats w600k_mbf
# (13 MB) on this gallery: cross-person similarity 0.007 vs 0.013 (cleaner
# separation) and 0.965 vs 0.937 stability under a 6-degree head turn, which is
# what stops a near-tied top 3 reordering as someone shifts in their seat. It
# costs 34 ms vs 4 ms per face -- irrelevant, since embedding only happens on
# state transitions, never per frame.
EMBEDDER_CANDIDATES = [
    os.path.join(WEIGHTS_DIR, 'w600k_r50.onnx'),
    os.path.join(WEIGHTS_DIR, 'w600k_mbf.onnx'),
]
EMBEDDER_ONNX_PATH = EMBEDDER_CANDIDATES[0]

# Fallback detector, reused from the existing age-predictor app so the kiosk
# never hard-fails at the table if YuNet is missing.
HAAR_CASCADE_PATH = os.path.join(SRC_DIR, 'weights', 'haarcascade_frontalface_default.xml')

GALLERIES_DIR = os.path.join(PACKAGE_DIR, 'galleries')
GALLERY_CACHE_PATH = os.path.join(PACKAGE_DIR, 'gallery_embeddings.npz')

LOGO_PATH = os.path.join(SRC_DIR, 'misc', 'cougar_ai_logo.png')
AGE_PREDICTOR_ONNX_PATH = os.path.join(SRC_DIR, 'weights', 'age_predictor.onnx')


# ---------------------------------------------------------------- galleries

GALLERY_ORDER = ['scientists', 'people', 'uh_majors']

AGE_INPUT_SIZE = (244, 244)
AGE_MIN_FACE_SIZE = 90
AGE_MIN_SHARPNESS = 18.0

MIN_GALLERY_ENTRIES = 8

# A LENS is a way of *labelling* a gallery, not a separate set of photos. Both
# lenses below run over the same 18 embeddings, so switching with 1/2 at the
# kiosk is instant and costs no extra disk, no extra download and no rebuild.
#
# This is how "you look like someone who majors in X" works without needing
# photos of real students: the people already in the gallery each genuinely
# majored in something, so the sentence is true of the face on screen. Several
# are usefully surprising -- Hinton read experimental psychology, Fei-Fei Li read
# physics, Rosenblatt was a psychologist.
#
#   gallery   which folder's embeddings to rank against ('galleries' for a list,
#             ranked as one pool -- no image files are duplicated)
#   menu      short word for the footer selector
#   title     shown on the ATTRACT screen
#   headline  small line above the match
#   primary   entry field for the big line
#   secondary entry field for the small line beneath it
LENSES = [
    {
        'key': 'pioneer',
        'galleries': ['scientists', 'people'],
        'menu': 'who',
        'title': 'Who do you look like?',
        'headline': 'you look like',
        'primary': 'name',
        'secondary': 'blurb',
    },
    {
        'key': 'major',
        'galleries': ['scientists', 'people'],
        'menu': 'major',
        'title': 'What do you look like you study?',
        'headline': 'you look like someone who majors in',
        'primary': 'major',
        'secondary': 'name',
    },
    {
        'key': 'uh',
        'gallery': 'uh_majors',
        'menu': 'UH',
        'title': 'Which of us are you?',
        'headline': 'you look like someone who majors in',
        'primary': 'major',
        'secondary': 'name',
    },
    {
        'key': 'age',
        'menu': 'age',
        'title': 'How old do you look?',
        'mode': 'age',
    },
]


# ---------------------------------------------------------------- models

# YuNet input size gets set per-frame; this is just the starting value.
DETECTOR_INPUT_SIZE = (320, 320)

# Detection runs on a downscaled copy of the frame. YuNet's cost scales with
# input area, and at 1280x720 it takes ~46 ms (21 fps) which alone blows the
# frame budget -- while at 640 wide it is ~4x cheaper and still finds a face
# that fills a good part of the frame, which is the only case this kiosk cares
# about. Boxes and landmarks are scaled back up to full-frame coordinates.
DETECT_MAX_WIDTH = 640
DETECTOR_SCORE_THRESHOLD = 0.85
DETECTOR_NMS_THRESHOLD = 0.3
DETECTOR_TOP_K = 50

# ArcFace / MobileFaceNet expects 112x112 BGR, scaled to [-1, 1].
EMBED_INPUT_SIZE = 112
EMBED_MEAN = 127.5
EMBED_SCALE = 127.5
EMBED_DIM = 512


# ---------------------------------------------------------------- camera

CAMERA_INDEX = 0
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

# Capture backend preference, in order. This ordering is measured, not guessed:
# on the test machine at 1280x720, DSHOW delivered read() at 100 ms (10 fps)
# while MSMF delivered 33 ms (30 fps). DSHOW does open ~1.3 s faster, but a
# kiosk opens the camera once and then runs for hours, so sustained frame rate
# wins by a mile. DSHOW stays as a fallback since MSMF is not always available.
# Override at the table with --camera-backend if the fair laptop disagrees.
CAMERA_BACKENDS = ['msmf', 'dshow', 'any']

WINDOW_NAME = 'Cougar AI'


# ---------------------------------------------------------------- kiosk timing

# All in seconds unless the name says frames.
MATCH_HINT_DELAY_SECONDS = 1.2
REVEAL_IDLE_TIMEOUT_SECONDS = 25.0
MATCH_IDLE_TIMEOUT_SECONDS = 20.0

# How long a face must be absent before we fall back to ATTRACT. Generous,
# because YuNet drops a frame here and there and we don't want to reset on it.
FACE_LOST_GRACE_SECONDS = 1.5

# Re-embed at most this often in the live REVEAL state, to keep the frame rate up.
REVEAL_REEMBED_INTERVAL_SECONDS = 0.35

TOP_K = 3

# The probe embedding is the average of the most recent PROBE_FRAMES embeddings
# seen while waiting for SPACE, then re-normalized. One frame is noisy enough
# that a blink or a 4-degree head turn can reorder a near-tied top 3; averaging
# about a second's worth makes the headline answer stable without hiding
# anything. It is a ROLLING window on purpose -- somebody can stand there for
# twenty seconds before pressing SPACE, and the answer should reflect how they
# looked at the moment they pressed it, not when they first walked up.
# The REVEAL panel deliberately stays single-frame, because its whole job is to
# show how much one frame moves.
PROBE_FRAMES = 12

# Cosine score a real face-verification system needs before it will claim two
# photos are the same person (the standard operating point for this ArcFace
# model). Measured gallery matches land around 0.10-0.18, i.e. FAR below it --
# which is the honest, quantitative version of the Act 2 point: the kiosk is not
# saying you look like Turing, only that you look marginally less unlike Turing
# than the other seventeen. Drawn as a reference line in the REVEAL state.
SAME_PERSON_THRESHOLD = 0.28


# ---------------------------------------------------------------- occlusion

# A 6x6 grid = 36 forward passes in a single batched ONNX call.
OCCLUSION_GRID = 6
OCCLUSION_PATCH_SCALE = 1.6   # patch is this * (crop / grid), so patches overlap
OCCLUSION_FILL = 114          # mid-gray, matches the [-1,1] input midpoint closely
OCCLUSION_ALPHA = 0.55


# ---------------------------------------------------------------- look

# All colors are BGR, because that is what OpenCV draws with. Writing them in
# RGB order is the single easiest way to get confusing output here.
COLOR_BG = (28, 16, 18)             # near-black, faint warm tint
COLOR_UH_RED = (84, 83, 228)        # #E45354, the same red as src/.streamlit/config.toml
COLOR_TEXT = (250, 245, 245)
COLOR_DIM = (165, 148, 150)
COLOR_ACCENT = (255, 210, 120)      # sky blue
COLOR_AMBER = (90, 200, 255)        # warm amber, for the "it is guessing" callouts
COLOR_WARN = (90, 200, 255)
COLOR_BOX = (130, 230, 170)         # mint, for the face box

FONT_CANDIDATES = [
    r'C:\Windows\Fonts\segoeuib.ttf',
    r'C:\Windows\Fonts\arialbd.ttf',
    r'C:\Windows\Fonts\segoeui.ttf',
    r'C:\Windows\Fonts\arial.ttf',
]
FONT_MONO_CANDIDATES = [
    r'C:\Windows\Fonts\consolab.ttf',
    r'C:\Windows\Fonts\consola.ttf',
    r'C:\Windows\Fonts\cour.ttf',
]
