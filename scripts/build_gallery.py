"""Precompute gallery embeddings into gallery_embeddings.npz.

Run this after adding or swapping any gallery photo:

    python scripts/build_gallery.py

The kiosk will rebuild the cache by itself if it notices changed files, but
doing it here means the fair laptop starts instantly instead of spending its
first ten seconds embedding portraits while a student watches a black screen.
"""

# System imports
import os
import sys

# Our imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from face_match_kiosk import gallery
from face_match_kiosk.detector import FaceDetector
from face_match_kiosk.embedder import Embedder
from face_match_kiosk.configs import MIN_GALLERY_ENTRIES


def main():
    detector = FaceDetector()
    embedder = Embedder()
    print('detector backend: %s' % detector.backend)

    galleries = gallery.build_all(detector, embedder, verbose=True)

    print()
    empty, thin = [], []
    for category, built in galleries.items():
        print('%-14s %3d usable entr%s'
              % (category, len(built), 'y' if len(built) == 1 else 'ies'))
        if len(built) == 0:
            empty.append(category)
        elif len(built) < MIN_GALLERY_ENTRIES:
            thin.append(category)

    if empty:
        print()
        print('Empty, so the kiosk will skip: %s' % ', '.join(empty))
        print('uh_majors ships empty on purpose -- it needs photos of people who')
        print('consented. Run: python scripts/capture_gallery.py --gallery uh_majors')

    if thin:
        print()
        print('Thin (want >= %d): %s' % (MIN_GALLERY_ENTRIES, ', '.join(thin)))
        print('Add more photos, then re-run. Too few faces and every match lands')
        print('on the same person.')

    if not empty and not thin:
        print()
        print('All galleries look healthy.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
