"""Gallery loading and embedding cache.

A gallery is just a folder of face images plus an optional meta.json describing
them:

    galleries/scientists/
        meta.json          [{"file": "turing.jpg", "name": "Alan Turing",
                             "blurb": "...", "credit": "Wikimedia, PD"}]
        turing.jpg
        ...

meta.json is optional. Anything not described there falls back to a name derived
from the filename, so a club member can drop photos into galleries/uh_majors/
and it just works without editing JSON.

Embeddings are cached to gallery_embeddings.npz and keyed on a fingerprint of
the (filename, size, mtime) of every image, so the cache self-invalidates when
you add or swap a photo.
"""

# System imports
import os
import json

# Library imports
import cv2
import numpy as np

# Our imports
from face_match_kiosk.configs import (
    GALLERIES_DIR,
    GALLERY_CACHE_PATH,
    GALLERY_ORDER,
)


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


class GalleryEntry:

    def __init__(self, path, name, blurb='', credit='', major='', face_box=None):
        self.path = path
        self.name = name
        self.blurb = blurb
        self.credit = credit

        # Field of study, shown by the 'major' lens. Optional -- galleries that
        # do not set it simply are not useful through that lens.
        self.major = major

        # Where the face actually is in the source image. Wikipedia lead images
        # are often group shots or conference photos, so a naive center crop can
        # show a poster with the person tiny in one corner. Cropping to the box
        # the detector found means the thumbnail shows the same face the
        # embedding was computed from.
        self.face_box = face_box

        self.thumbnail = None      # populated lazily by the renderer
        self.embedding = None

    def __repr__(self):
        return '<GalleryEntry %s>' % self.name


class Gallery:

    def __init__(self, category, entries, matrix):
        self.category = category
        self.entries = entries
        self.matrix = matrix       # (N, 512) L2-normalized, row i <-> entries[i]

    def __len__(self):
        return len(self.entries)


def combine(category, built_galleries):
    """Concatenate galleries into one rankable pool.

    Lets a lens draw on more than one folder without duplicating image files.
    The 'major' lens uses this to rank against `scientists` and `people`
    together, which is what turns a STEM-only spread into one that covers the
    whole university.
    """
    usable = [built for built in built_galleries if built is not None and len(built)]
    if not usable:
        return None
    if len(usable) == 1:
        return usable[0]

    entries = [entry for built in usable for entry in built.entries]
    matrix = np.vstack([built.matrix for built in usable])
    return Gallery(category, entries, matrix)


# ---------------------------------------------------------------- discovery

def _pretty_name(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.replace('_', ' ').replace('-', ' ').strip().title()


def list_categories():
    if not os.path.isdir(GALLERIES_DIR):
        return []

    found = [name for name in os.listdir(GALLERIES_DIR)
             if os.path.isdir(os.path.join(GALLERIES_DIR, name))]

    # Preserve the configured order, then append anything extra alphabetically.
    ordered = [name for name in GALLERY_ORDER if name in found]
    ordered += sorted(name for name in found if name not in GALLERY_ORDER)
    return ordered


def discover_entries(category):
    """Build the entry list for a category from meta.json plus loose files."""
    directory = os.path.join(GALLERIES_DIR, category)
    if not os.path.isdir(directory):
        return []

    described = {}
    meta_path = os.path.join(directory, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as meta_file:
            for record in json.load(meta_file):
                described[record['file']] = record

    filenames = sorted(name for name in os.listdir(directory)
                       if name.lower().endswith(IMAGE_EXTENSIONS))

    entries = []
    for filename in filenames:
        record = described.get(filename, {})
        entries.append(GalleryEntry(
            path=os.path.join(directory, filename),
            name=record.get('name') or _pretty_name(filename),
            blurb=record.get('blurb', ''),
            credit=record.get('credit', ''),
            major=record.get('major', ''),
        ))
    return entries


def _fingerprint(entries):
    """Cheap change-detector: filename + size + mtime for every image."""
    parts = []
    for entry in entries:
        stat = os.stat(entry.path)
        parts.append('%s:%d:%d' % (os.path.basename(entry.path),
                                   stat.st_size, int(stat.st_mtime)))
    return '|'.join(parts)


# ---------------------------------------------------------------- embedding

def embed_entries(entries, detector, embedder, verbose=False):
    """Detect, align and embed each gallery image. Returns kept entries + matrix.

    Images with no detectable face are dropped rather than embedded as-is: a
    whole-image embedding would be meaningless and would pollute every match.
    """
    kept, crops = [], []

    for entry in entries:
        image = cv2.imread(entry.path, cv2.IMREAD_COLOR)
        if image is None:
            if verbose:
                print('    [skip] unreadable: %s' % os.path.basename(entry.path))
            continue

        face = detector.detect_largest(image)
        if face is None:
            if verbose:
                print('    [skip] no face found: %s' % os.path.basename(entry.path))
            continue

        entry.face_box = face.box
        crops.append(detector.align(image, face))
        kept.append(entry)

    if not kept:
        return [], np.zeros((0, embedder.dim), dtype=np.float32)

    matrix = embedder.embed_batch(crops)
    for entry, vector in zip(kept, matrix):
        entry.embedding = vector

    return kept, matrix


def build_all(detector, embedder, verbose=True):
    """Embed every category and write the cache. Returns {category: Gallery}."""
    galleries, payload = {}, {}

    for category in list_categories():
        entries = discover_entries(category)
        if verbose:
            print('  %s: %d image(s)' % (category, len(entries)))

        kept, matrix = embed_entries(entries, detector, embedder, verbose)
        galleries[category] = Gallery(category, kept, matrix)

        payload['%s__matrix' % category] = matrix
        payload['%s__meta' % category] = np.array(json.dumps(
            [{'path': entry.path, 'name': entry.name, 'blurb': entry.blurb,
              'credit': entry.credit, 'major': entry.major,
              'face_box': list(entry.face_box or [])}
             for entry in kept]))
        payload['%s__fingerprint' % category] = np.array(_fingerprint(kept))

        if verbose and len(kept) != len(entries):
            print('    kept %d of %d' % (len(kept), len(entries)))

    np.savez(GALLERY_CACHE_PATH, **payload)
    if verbose:
        print('  cache -> %s' % GALLERY_CACHE_PATH)

    return galleries


def load_all(detector, embedder, verbose=False):
    """Load galleries from cache, rebuilding if any category's images changed."""
    if not os.path.exists(GALLERY_CACHE_PATH):
        if verbose:
            print('No gallery cache yet; building it now.')
        return build_all(detector, embedder, verbose)

    cached = np.load(GALLERY_CACHE_PATH, allow_pickle=False)
    galleries, stale = {}, False

    for category in list_categories():
        entries = discover_entries(category)
        matrix_key = '%s__matrix' % category

        if matrix_key not in cached.files:
            stale = True
            break

        # Compare against the files that still exist, since the cache only ever
        # stored entries that produced a usable face.
        present = {os.path.basename(entry.path): entry for entry in entries}
        meta = json.loads(str(cached['%s__meta' % category]))
        cached_names = [os.path.basename(record['path']) for record in meta]

        if any(name not in present for name in cached_names):
            stale = True
            break

        # A file appeared that the cache never saw.
        if len(present) != len(cached_names):
            stale = True
            break

        rebuilt = [present[name] for name in cached_names]
        if _fingerprint(rebuilt) != str(cached['%s__fingerprint' % category]):
            stale = True
            break

        matrix = cached[matrix_key]
        for entry, record, vector in zip(rebuilt, meta, matrix):
            entry.name = record['name']
            entry.blurb = record['blurb']
            entry.credit = record['credit']
            entry.major = record.get('major', '')
            entry.face_box = tuple(record.get('face_box') or ()) or None
            entry.embedding = vector

        galleries[category] = Gallery(category, rebuilt, matrix)

    if stale:
        if verbose:
            print('Gallery images changed; rebuilding cache.')
        return build_all(detector, embedder, verbose)

    return galleries
