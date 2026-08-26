# `uh_majors` gallery — optional

**You do not need this gallery.** The `major` lens already works off
`scientists` + `people`: 58 faces across 30 majors, everyone with a documented
undergraduate field. Press `2` at the kiosk.

Fill this one when you want the same headline pointed at *real club members*
instead of notable strangers — more relatable, and it introduces your officers to
recruits. Once it has faces, a third lens appears automatically.

## Filling it

```bash
python scripts/capture_gallery.py --gallery uh_majors
python scripts/build_gallery.py
```

The capture script photographs people one at a time, asks for a name and a
major, confirms they consented, and writes `meta.json` for you. Aim for 8 or
more — a thin gallery makes every visitor match the same face.

Photograph people who say yes, in person. Someone who put a photo on LinkedIn
consented to *that publication*, not to being enrolled in a face-matching
gallery, and "whose faces are those?" is a question a visitor will actually ask.

## Keep showing the face

The kiosk shows the photo, the major, and the person's name together. Keep it
that way. "You look like Priya, a CS junior" is a claim about *resemblance* that
the student can check by looking at the screen. Hiding the photo and displaying
only "Computer Science" turns it into a claim about the student, which is a
different and unsupportable thing.

## Format

If you would rather write `meta.json` by hand:

```json
[
  {
    "file": "priya_raman.jpg",
    "name": "Priya Raman",
    "blurb": "Computer Science, junior",
    "major": "Computer Science",
    "credit": "photographed with consent at a Cougar AI meeting"
  }
]
```

Without `meta.json`, names come from filenames (`priya_raman.jpg` → "Priya
Raman"), but the `major` lens needs the `major` field, so use the capture script
or set it yourself. Anything with no detectable face is reported and skipped by
`build_gallery.py` — read its output.
