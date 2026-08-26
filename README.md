# [Former] CNN-Age-Predictor-App

Welcome to the Cougar AI Spring 2024 Project!

In this project we will build a computer vision application capable of predicting the age of the user through a webcam. We will teach everything needed to understand the application from the ground up. Including convolutional neural networks, transfer learning, resnets, and general user interfaces!

![alt text](blob/app_picture.jpg?raw=true)

The goal of the workshop is to give our members a fundamental understanding of neural networks and computer vision. Allowing them to take on projects of their own!

The project will be taught in weekly workshops, learning the necessary skills piece by piece until the whole application is put together. The detailed outline of the project can be found as workshop/project_outline.pdf in this github repo. The high level project outline of the workshop is as follows:
  - Week 1: Intro to Machine Learning and Single Layer Networks
  - Week 2: Multilayer Networks
  - Week 3 & 4: Convolutional Neural Networks
  - Week 5: Transfer Learning, Resnets, Coding Standards, and GUIs
  - Week 6 & Beyond: Age Predictor App

The project was inspired by the book, Modern Computer Vision with PyTorch: https://www.amazon.com/Modern-Computer-Vision-PyTorch-applications/dp/1839213477. 

The dataset used for Age Predictor is the FairFace Datset. And the labels used to train the network, come from the book, Modern Computer Vision with PyTorch.
  - Fairface Dataset: https://github.com/joojs/fairface
  - Fairface Dataset Train & Val Labels, The labels are stored in a google drive created by the book's author. Refer to the github block on importing the data: https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch/blob/master/Chapter05/age_gender_prediction.ipynb

## Installation and Running

1. Clone or download the repo
    ```bash
    git clone https://github.com/Cougar-AI/CNN-Age-Predictor-App.git
    ```
2. Create a Python environment (Python 3.9 recommended)
   ```bash
   conda create -n AgePredictor python=3.9
   conda activate AgePredictor
   ```

3. Install dependencies from requirements.txt
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application
    ```bash
    cd src
    streamlit run main.py
    ```

The application will open in your default browser. 
Feel free to contribute, and build on this project as a baseline!

---

# Fair Kiosk: "Who Do You Look Like?"

A second, self-contained app in this repo, built for **recruiting at a club
fair** rather than for the workshop. It lives in `src/face_match_kiosk/` and does
not touch the age predictor above, which still works exactly as documented.

The age predictor is a good teaching capstone but a weak demo at a table: a
stranger takes one photo, gets one number that is often off by six years, and
walks away. Nothing they did, and nothing learned. This runs in two acts instead.

**Act 1 — the hook.** Your face is embedded and matched by cosine similarity
against a pool of 58 notable people. You get a confident, fun answer.
**Act 2 — the reveal.** Press `SPACE`. The same three scores come back nearly
tied, every one of them far below the threshold a real face-ID system needs to
call two photos the same person. A heatmap shows which pixels actually moved the
answer, and a live panel shows a single frame's answer flipping as you move.

> The AI was confident. It was also basically guessing.
> Learning to tell those apart is what this club does.

The reveal is not a bolted-on second demo — it is Act 1's own numbers, read
honestly. The headline answer is stable because it averages a second of frames;
the live panel is deliberately single-frame, so you can watch what one frame is
actually worth. That contrast *is* the lesson.

The kiosk also includes a third lens, **Age**, which estimates apparent age from
the same webcam face crop. It is an estimate for demonstration purposes, not an
identity claim or a reliable measure of someone's actual age.

## Install and run

The kiosk needs **no torch and no torchvision** — detection, embedding and the
heatmap all run on `onnxruntime`.

```bash
pip install -r requirements-kiosk.txt

python scripts/fetch_models.py     # once, on wifi: detector + recognition model
python scripts/seed_scientists.py  # once, on wifi: 18 AI pioneers
python scripts/seed_people.py      # once, on wifi: 40 more, for major coverage
python scripts/build_gallery.py    # once: precomputes gallery embeddings
python run_kiosk.py                # the fair build, fullscreen
```

On Windows you can just double-click `run_kiosk.bat`. Useful flags:

| Flag | Why |
|---|---|
| `--no-fullscreen` | windowed, for development |
| `--image face.jpg` | run without a camera at all |
| `--selftest` | verify models, galleries and the embedding math, then exit |
| `--lens major` | start on the "you look like someone who majors in" lens |
| `--lens age` | start on the apparent-age lens |
| `--camera 1` | pick a different webcam |
| `--camera-backend dshow` | if the default backend misbehaves on your laptop |
| `--rebuild` | force a gallery re-embed |

**Run `--selftest` before the fair.** For every gallery it checks that the models
load, that embeddings are L2-normalized, that `cosine(x, x) == 1`, that each
image matches itself best, and that no two people collide above the same-person
threshold — which is how you catch the same person accidentally added twice.

At the kiosk: `SPACE` reveal / back · `R` reset · `1` `2` `3` switch lens ·
`F` match immediately · `Q` quit.

## How it works

```
webcam frame
  -> YuNet detector             5 facial landmarks
  -> affine warp                canonical 112x112 crop (ArcFace template)
  -> ArcFace w600k_r50          512-d embedding, L2-normalized
  -> average over ~12 frames    one stable probe per visitor
  -> cosine vs gallery matrix   one matrix-vector multiply
  -> top 3 matches

For the Age lens, the same detected face is padded for context and sent through
the bundled age model in three views: normal, mirrored, and slightly brightened.
The kiosk takes their median and then applies a rolling median across frames.
Small or blurry face crops are ignored so one poor camera frame cannot move the
result as much.
```

Because the probe and the gallery rows are both L2-normalized, cosine similarity
is just a dot product. That is worth saying out loud at the table: the "AI
decision" a student is reacting to is **one dot product**.

Three implementation notes that matter if you touch this:

- **Detection runs on a 640-wide copy** of the frame. YuNet costs ~46 ms at
  1280x720 and ~13 ms at 640 wide, and the kiosk only cares about faces that
  fill a good part of the frame. Boxes and landmarks are scaled back up.
- **All text for a screen goes through one `render.Painter`.** Each standalone
  PIL text call converts the whole frame (~5 ms), and the REVEAL screen has
  eighteen labels — batching took it from 196 ms to 26 ms per frame.
- **Capture is MSMF-first.** On the test machine DSHOW delivered `read()` at
  100 ms (10 fps) at 720p against MSMF's 33 ms (30 fps). DSHOW opens ~1.3 s
  faster, but a kiosk opens the camera once and then runs for hours.

### The models

| Model | Size | Job |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` | 0.2 MB | detection + 5 landmarks |
| `w600k_r50.onnx` | 166 MB | 512-d face embedding |
| `../weights/age_predictor.onnx` | 36 MB | apparent-age estimate |

`w600k_mbf.onnx` (13 MB) is a supported fallback if the big download is
impractical — `embedder.py` takes whichever is present, best first. r50 is the
default because it measures better on this gallery: cross-person similarity
0.007 vs 0.013 (cleaner separation) and 0.965 vs 0.937 stability under a
6-degree head turn, which is what stops a near-tied top 3 from reordering as
someone shifts in their seat. It costs 34 ms per face against 4 ms, which does
not matter because embedding only happens on state transitions, never per frame.

### The Act 2 heatmap

**Occlusion sensitivity**, not Grad-CAM. Grad-CAM needs gradients, hence torch.
Instead we slide a gray patch over a 6x6 grid, push all 36 masked variants
through the embedder in a single batched call, and score each cell by how far the
embedding moved. Forward passes only, and it explains itself in one sentence to
someone with no ML background: *we cover part of your face and measure how much
the answer moves.*

With r50 that costs ~1.5 s, which would freeze the window on a keypress, so it is
computed on a worker thread the moment MATCH is entered. By the time anyone
presses `SPACE` it is already done (measured wait: 0.000 s); press it instantly
and REVEAL renders without the heatmap rather than hanging.

## Galleries and lenses

A **gallery** is a folder of photos under `src/face_match_kiosk/galleries/` plus
an optional `meta.json` giving each face a display name, a blurb, a field of
study and a credit. The embedding cache invalidates itself when you add or swap
a photo.

A **lens** is either a way of *labelling* a gallery or a separate analysis of the
detected face. Press `1`, `2`, or `3` at the kiosk to switch. Gallery lenses can
draw on several galleries, which they rank as one pool without duplicating image
files:

| Lens | Headline | Big line | Small line |
|---|---|---|---|
| `pioneer` | "you look like" | Grace Hopper | her one-line blurb |
| `major` | "you look like someone who majors in" | Anthropology | Yo-Yo Ma |
| `age` | "estimated age" | 27 years old | an estimate, not an identity |

The first two lenses rank against the **same 58-face pool**, so the person behind the
answer is identical either way — one lens shows their name, the other shows their
major. Pointing the lenses at different pools makes them disagree about who you
look like, which reads as a bug.

The `age` lens does not use the gallery. It runs the existing MobileNet-based
ONNX age model locally and smooths several augmented views and camera frames for
a steadier result. No photo is saved.

The `major` lens is how you get a harmless "what do you look like you study?"
guess **without needing photos of real students**. Everyone in the pool genuinely
majored in something, so the sentence is true of the face on screen.

**58 faces across 30 majors** — Liberal Arts, Business, Architecture, Culinary
Arts, Social Work, Education, Journalism, Art, Theatre, Drama, Film, Philosophy,
Anthropology and more, so a visitor can land somewhere other than engineering.

Plenty are usefully counter-intuitive, which is exactly what you want a student
to argue with: Bruce Lee read **philosophy**, Yo-Yo Ma read **anthropology**,
Weird Al read **architecture**, Harrison Ford read **philosophy**, Geoffrey
Hinton read **experimental psychology**, Julia Child read **English**. Three are
University of Houston alumni — Jim Parsons (Theatre), Brené Brown (Social Work)
and Dennis Quaid (Drama) — worth pointing at when a prospective student is
standing right in front of you.

Add a lens, or repoint one, in `LENSES` in
[configs.py](src/face_match_kiosk/configs.py). A lens whose pool is empty, or
whose entries lack the label it needs, is skipped automatically — so a
half-finished gallery can never put a broken screen in front of a student.

### The galleries

- **`scientists`** — 18 AI and computing figures from
  `scripts/seed_scientists.py`, each with a blurb and a real undergraduate
  field. Several tie back to the workshop: LeCun invented the CNN the age
  predictor runs on, Fei-Fei Li built the ImageNet its backbone is pretrained
  on, and Rosenblatt's perceptron is literally week 1.
- **`people`** — 40 figures from `scripts/seed_people.py`, chosen to cover the
  majors UH actually offers. One of the 41 in the roster (Philip Glass) has no
  detectable face in its Wikipedia lead image and is skipped automatically;
  `build_gallery.py` reports it rather than failing quietly.
- **`uh_majors`** — ships **empty**, so its lens stays hidden. Optional: fill it
  with club members who consent, in about twenty minutes at a meeting.

  ```bash
  python scripts/capture_gallery.py --gallery uh_majors
  python scripts/build_gallery.py
  ```

  That prompts for a name and major per person, records that they consented, and
  writes `meta.json` for you.

Portraits come from Wikimedia Commons; the licence note per person lives in each
gallery's `meta.json`. Check it before putting a screenshot on social media.

The roster is deliberately diverse across gender, ethnicity and era, because a
face-recognition embedding is **not** demographically neutral and a lopsided
gallery produces lopsided matches. You will still see matches cluster
demographically — that is the bias itself, not a bug in the code, and it is
exactly what Act 2 exists to expose. Joy Buolamwini and Timnit Gebru are on the
roster on purpose: they are the researchers who documented this, which makes them
the best possible thing this demo can match a student to.

## Privacy

**The kiosk never writes a frame to disk.** No photos, no embeddings, no logs of
who visited. The ATTRACT screen teases gallery names, never previous visitors'
results — nothing about a visitor outlives their turn. Put "no photos are saved"
on the table sign; it is true, it is a good talking point, and the footer says so
on every screen.
