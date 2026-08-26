"""Photograph consenting people straight into a gallery folder.

Built for the club-meeting workflow: sit a laptop on a table, have members who
want to be in the demo come up one at a time, and shoot each one in about twenty
seconds. It writes the image and updates meta.json for you.

    python scripts/capture_gallery.py --gallery uh_majors
    python scripts/capture_gallery.py --gallery uh_majors --label-prompt "Major"

Why this exists rather than a scraper: a gallery entry is a biometric template
of a specific person. Someone posting a photo online consented to *that
publication*, not to being enrolled in a face-matching gallery. Asking a member
in person takes seconds and makes the whole demo defensible when a student at the
table asks "wait, whose faces are those?" -- which they will.

Each capture records an explicit consent acknowledgement in meta.json.
"""

# System imports
import os
import sys
import json
import argparse
import unicodedata

# Library imports
import cv2

# Our imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from face_match_kiosk import render
from face_match_kiosk.configs import (
    GALLERIES_DIR, CAMERA_BACKENDS, COLOR_BOX, COLOR_TEXT, COLOR_AMBER)
from face_match_kiosk.detector import FaceDetector


WINDOW = 'Cougar AI - gallery capture'
CONSENT_NOTE = ('photographed in person with spoken consent for use in the '
                'Cougar AI fair demo')


FALLBACK_SLUG = 'person'


def slugify(text):
    """Filename-safe ASCII slug. Returns FALLBACK_SLUG if nothing transliterates."""
    normalized = unicodedata.normalize('NFKD', text)
    ascii_only = normalized.encode('ascii', 'ignore').decode('ascii')
    keep = [character.lower() if character.isalnum() else '_'
            for character in ascii_only]
    slug = ''.join(keep)
    while '__' in slug:
        slug = slug.replace('__', '_')
    return slug.strip('_') or FALLBACK_SLUG


def unique_filename(directory, slug, extension='.jpg'):
    """Number the fallback slug so two non-transliterating names cannot collide.

    A name written entirely in a non-Latin script slugifies to FALLBACK_SLUG, so
    without this the second such member would silently be offered the first
    one's file to overwrite. Real slugs are left alone -- a genuine repeat there
    is a repeat name, which is worth prompting about.
    """
    if slug != FALLBACK_SLUG:
        return slug + extension

    index = 1
    while os.path.exists(os.path.join(directory, '%s_%d%s'
                                      % (slug, index, extension))):
        index += 1
    return '%s_%d%s' % (slug, index, extension)


def open_camera(index):
    backends = {'msmf': cv2.CAP_MSMF, 'dshow': cv2.CAP_DSHOW, 'any': cv2.CAP_ANY}
    for name in CAMERA_BACKENDS:
        capture = cv2.VideoCapture(index, backends[name])
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, _ = capture.read()
            if ok:
                return capture
        capture.release()
    raise RuntimeError('Could not open camera %d' % index)


def load_meta(directory):
    path = os.path.join(directory, 'meta.json')
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as meta_file:
        return json.load(meta_file)


def save_meta(directory, records):
    path = os.path.join(directory, 'meta.json')
    with open(path, 'w', encoding='utf-8') as meta_file:
        json.dump(records, meta_file, indent=2, ensure_ascii=False)


def shoot(capture, detector, name, label):
    """Live preview until SPACE captures or ESC skips. Returns a frame or None."""
    print('    SPACE = capture, ESC = skip. Look at the camera.')

    while True:
        ok, frame = capture.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        preview = frame.copy()

        face = detector.detect_largest(preview)
        if face is not None:
            cv2.rectangle(preview, (face.x, face.y),
                          (face.x + face.w, face.y + face.h), COLOR_BOX, 3)

        # Text goes through the kiosk's PIL renderer rather than cv2.putText,
        # which cannot draw non-ASCII -- and club rosters have accented names.
        caption = '%s  --  %s' % (name, label) if label else name
        painter = render.Painter(preview)
        painter.text(caption, (24, 34), size=38, color=COLOR_TEXT)
        painter.text(
            'face found - SPACE to capture' if face is not None
            else 'no face detected - move into frame',
            (24, preview.shape[0] - 46), size=28,
            color=COLOR_BOX if face is not None else COLOR_AMBER)

        cv2.imshow(WINDOW, painter.done())
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return None
        if key == ord(' '):
            if face is None:
                print('    no face detected in that frame, try again')
                continue
            return frame


def main():
    parser = argparse.ArgumentParser(
        description='Capture consenting people into a kiosk gallery.')
    parser.add_argument('--gallery', required=True,
                        help='gallery folder name, e.g. uh_majors')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--label-prompt', default='Major',
                        help='what to ask for as the blurb (default: %(default)s)')
    args = parser.parse_args()

    directory = os.path.join(GALLERIES_DIR, args.gallery)
    os.makedirs(directory, exist_ok=True)

    records = load_meta(directory)
    by_file = {record['file']: record for record in records}

    print('Capturing into %s' % directory)
    print('%d entr%s already there.'
          % (len(records), 'y' if len(records) == 1 else 'ies'))
    print()
    print('Say this to each person before you photograph them:')
    print('  "This photo goes in a face-matching demo we run at club fairs.')
    print('   Your face will appear on screen when someone matches you.')
    print('   It stays on our laptop. Is that OK?"')
    print()
    print('Press Enter on an empty name when you are done.')
    print()

    detector = FaceDetector()
    capture = open_camera(args.camera)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    added = 0
    try:
        while True:
            name = input('Name (blank to finish): ').strip()
            if not name:
                break

            label = input('%s: ' % args.label_prompt).strip()

            consent = input('Did they say yes, out loud, just now? [y/N]: ').strip().lower()
            if consent not in ('y', 'yes'):
                print('    skipped -- no consent recorded')
                print()
                continue

            frame = shoot(capture, detector, name, label)
            if frame is None:
                print('    skipped')
                print()
                continue

            filename = unique_filename(directory, slugify(name))
            destination = os.path.join(directory, filename)

            if os.path.exists(destination):
                overwrite = input('    %s exists. Overwrite? [y/N]: '
                                  % filename).strip().lower()
                if overwrite not in ('y', 'yes'):
                    print('    skipped')
                    print()
                    continue

            cv2.imwrite(destination, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

            # Write the label to BOTH fields: 'blurb' is the small line under a
            # name on the pioneer-style lens, and 'major' is what the majors
            # lens shows as its headline. Setting only one leaves the entry
            # invisible to the other lens.
            by_file[filename] = {
                'file': filename,
                'name': name,
                'blurb': label,
                'major': label,
                'credit': CONSENT_NOTE,
            }
            save_meta(directory, list(by_file.values()))

            added += 1
            print('    saved %s' % filename)
            print()
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print()
    print('Added %d entr%s. Total now %d.'
          % (added, 'y' if added == 1 else 'ies', len(by_file)))
    print('Now run: python scripts/build_gallery.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
