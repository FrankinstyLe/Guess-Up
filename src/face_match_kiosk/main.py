"""Entry point for the Cougar AI fair kiosk.

    python -m face_match_kiosk.main                    # the fair build
    python -m face_match_kiosk.main --no-fullscreen    # windowed, for development
    python -m face_match_kiosk.main --image face.jpg   # no camera needed
    python -m face_match_kiosk.main --selftest         # check the install
"""

# System imports
import os
import sys
import argparse

# Library imports
import numpy as np

# Allow `python src/face_match_kiosk/main.py` as well as `-m`.
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Our imports
from face_match_kiosk import gallery as gallery_module
from face_match_kiosk import matcher
from face_match_kiosk.configs import (
    CAMERA_INDEX,
    MIN_GALLERY_ENTRIES,
    SAME_PERSON_THRESHOLD,
)
from face_match_kiosk.detector import FaceDetector
from face_match_kiosk.embedder import Embedder
from face_match_kiosk.kiosk import Kiosk


def build_parser():
    parser = argparse.ArgumentParser(
        prog='face_match_kiosk',
        description='Cougar AI fair kiosk: face embedding match, then the reveal.')

    parser.add_argument('--camera', type=int, default=CAMERA_INDEX,
                        help='camera index (default %(default)s)')
    parser.add_argument('--camera-backend', choices=['msmf', 'dshow', 'any'],
                        help='force a capture backend (default: try msmf, then '
                             'dshow, then any -- msmf is much faster at 720p)')
    parser.add_argument('--no-fullscreen', action='store_true',
                        help='run in a window instead of fullscreen')
    parser.add_argument('--image', metavar='PATH',
                        help='use a still image instead of the camera')
    parser.add_argument('--lens', metavar='KEY',
                        help='lens to start on: pioneer, major, uh')
    parser.add_argument('--gallery', metavar='NAME',
                        help='alias for --lens, also accepts a gallery name')
    parser.add_argument('--selftest', action='store_true',
                        help='verify models and galleries, then exit')
    parser.add_argument('--rebuild', action='store_true',
                        help='force a gallery embedding rebuild before starting')

    return parser


# ---------------------------------------------------------------- selftest

def selftest():
    """Check everything the kiosk needs, loudly, and return an exit code.

    Distinguishes problems (the kiosk will not work) from warnings (it will
    work, but something is worth knowing). An empty optional gallery is a
    warning: uh_majors and characters ship empty on purpose because they need
    photos the club has consent and rights to use.
    """
    problems, warnings = [], []

    print('detector...')
    detector = FaceDetector()
    print('  backend: %s' % detector.backend)
    if detector.backend != 'yunet':
        warnings.append('YuNet missing, running on the weaker Haar fallback. '
                        'Run: python scripts/fetch_models.py')

    print('embedder...')
    embedder = Embedder()
    print('  dim: %d' % embedder.dim)

    print('galleries...')
    galleries = gallery_module.load_all(detector, embedder, verbose=True)
    if not galleries:
        problems.append('no galleries found at all')

    usable = 0
    for category, built in sorted(galleries.items()):
        print('  %-14s %3d entries' % (category, len(built)))
        if len(built) == 0:
            warnings.append('gallery "%s" is empty, so the kiosk will skip it; '
                            'see its README.md to fill it' % category)
        elif len(built) < MIN_GALLERY_ENTRIES:
            warnings.append('gallery "%s" has only %d entries (want >= %d); '
                            'matches will get repetitive'
                            % (category, len(built), MIN_GALLERY_ENTRIES))
            usable += 1
        else:
            usable += 1

    if usable == 0:
        problems.append('no gallery has any usable faces, the kiosk cannot run')

    print('embedding sanity...')

    # Check EVERY gallery, not just the first. `people` is now the largest and
    # is seeded from Wikipedia titles, so a redirect collapsing two roster
    # entries onto one person is a real failure mode -- it would show up here as
    # two "different" people scoring above the same-person threshold.
    for category, built in sorted(galleries.items()):
        if len(built) == 0:
            continue

        matrix = built.matrix
        norms = np.linalg.norm(matrix, axis=1)

        if not np.allclose(norms, 1.0, atol=1e-3):
            problems.append('%s: embeddings are not L2-normalized' % category)

        self_similarity = float(matrix[0] @ matrix[0])
        if abs(self_similarity - 1.0) > 1e-3:
            problems.append('%s: cosine(x, x) != 1, embedding pipeline is wrong'
                            % category)

        if len(matrix) < 2:
            print('  %-12s %2d entry, nothing to compare' % (category, len(matrix)))
            continue

        ranked = matcher.top_k(matrix[0], built, 1)
        if ranked[0].entry is not built.entries[0]:
            problems.append('%s: a gallery image does not match itself best'
                            % category)

        cross = matrix @ matrix.T
        off_diagonal = cross[~np.eye(len(matrix), dtype=bool)]

        print('  %-12s %2d entries | cross-person cosine mean %.3f max %.3f'
              % (category, len(matrix), off_diagonal.mean(), off_diagonal.max()))

        if off_diagonal.max() > SAME_PERSON_THRESHOLD:
            # Name the pair, so the fix is obvious rather than a scavenger hunt.
            masked = np.where(np.eye(len(matrix), dtype=bool), -9.0, cross)
            row, column = np.unravel_index(int(np.argmax(masked)), masked.shape)
            problems.append(
                '%s: "%s" and "%s" score %.3f, above the %.2f same-person '
                'threshold -- probably the same person twice'
                % (category, built.entries[row].name, built.entries[column].name,
                   cross[row, column], SAME_PERSON_THRESHOLD))

    print()
    if warnings:
        print('%d warning(s):' % len(warnings))
        for warning in warnings:
            print('  ! %s' % warning)
        print()

    if problems:
        print('SELFTEST FOUND %d PROBLEM(S):' % len(problems))
        for problem in problems:
            print('  - %s' % problem)
        return 1

    print('SELFTEST OK -- the kiosk is ready to run.')
    return 0


# ---------------------------------------------------------------- main

def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.selftest:
        return selftest()

    detector = FaceDetector()
    embedder = Embedder()

    if args.rebuild:
        galleries = gallery_module.build_all(detector, embedder, verbose=True)
    else:
        galleries = gallery_module.load_all(detector, embedder, verbose=True)

    Kiosk(detector, embedder, galleries, args).run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
