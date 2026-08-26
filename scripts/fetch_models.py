"""Download the ONNX models the kiosk needs.

Run this ONCE at home, on wifi, before the fair. The kiosk itself never touches
the network at runtime -- if you skip this step you will be standing at a table
with a broken laptop.

    python scripts/fetch_models.py
"""

# System imports
import io
import os
import sys
import zipfile
import urllib.request

# Our imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from face_match_kiosk.configs import WEIGHTS_DIR, DETECTOR_ONNX_PATH


USER_AGENT = 'cougar-ai-kiosk/1.0'

# YuNet ships as a bare .onnx in opencv_zoo.
DETECTOR_URL = ('https://github.com/opencv/opencv_zoo/raw/main/models/'
                'face_detection_yunet/face_detection_yunet_2023mar.onnx')
DETECTOR_MIN_BYTES = 100_000

# Recognition models are only distributed inside InsightFace bundles, so we pull
# the zip and lift out the one file we need.
#
# r50 is the default because it is measurably more accurate and more stable to
# head movement; mbf is the small fallback for a slow connection. Either one
# works -- embedder.py takes whichever is present, best first.
EMBEDDERS = [
    {
        'name': 'w600k_r50 (accurate, 166 MB from a 288 MB bundle)',
        'url': ('https://github.com/deepinsight/insightface/releases/'
                'download/v0.7/buffalo_l.zip'),
        'member': 'w600k_r50.onnx',
        'min_bytes': 150_000_000,
    },
    {
        'name': 'w600k_mbf (small fallback, 13 MB from a 15 MB bundle)',
        'url': ('https://github.com/deepinsight/insightface/releases/'
                'download/v0.7/buffalo_sc.zip'),
        'member': 'w600k_mbf.onnx',
        'min_bytes': 10_000_000,
    },
]


def _get(url):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def _already_have(path, min_bytes):
    return os.path.exists(path) and os.path.getsize(path) >= min_bytes


def fetch_detector():
    if _already_have(DETECTOR_ONNX_PATH, DETECTOR_MIN_BYTES):
        print('  [skip] already present (%.1f MB)'
              % (os.path.getsize(DETECTOR_ONNX_PATH) / 1e6))
        return True

    data = _get(DETECTOR_URL)
    if len(data) < DETECTOR_MIN_BYTES:
        print('  got only %d bytes -- probably an LFS pointer or error page' % len(data))
        return False

    with open(DETECTOR_ONNX_PATH, 'wb') as output_file:
        output_file.write(data)
    print('  ok: %.1f MB' % (len(data) / 1e6))
    return True


def fetch_embedder():
    """Get the best recognition model we can; stop at the first success."""
    for spec in EMBEDDERS:
        destination = os.path.join(WEIGHTS_DIR, spec['member'])

        if _already_have(destination, spec['min_bytes']):
            print('  [skip] %s already present (%.0f MB)'
                  % (spec['member'], os.path.getsize(destination) / 1e6))
            return True

        print('  %s' % spec['name'])
        try:
            archive = zipfile.ZipFile(io.BytesIO(_get(spec['url'])))
        except Exception as error:
            print('    download failed: %s' % error)
            continue

        if spec['member'] not in archive.namelist():
            print('    %s not in bundle' % spec['member'])
            continue

        with archive.open(spec['member']) as source:
            data = source.read()
        with open(destination, 'wb') as output_file:
            output_file.write(data)

        print('    ok: %.0f MB extracted' % (len(data) / 1e6))
        return True

    return False


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    steps = [
        ('YuNet face detector', fetch_detector),
        ('Face recognition embedder', fetch_embedder),
    ]

    failures = []
    for name, step in steps:
        print(name)
        try:
            if not step():
                failures.append(name)
        except Exception as error:
            print('  failed: %s' % error)
            failures.append(name)

    print()
    if failures:
        print('FAILED: %s' % ', '.join(failures))
        print('Without the detector the kiosk falls back to the Haar cascade.')
        print('Without the embedder the matcher cannot run at all.')
        print('Drop the files manually into %s' % WEIGHTS_DIR)
        return 1

    print('All models present in %s' % WEIGHTS_DIR)
    return 0


if __name__ == '__main__':
    sys.exit(main())
