"""Seed galleries/scientists/ with portraits from Wikipedia / Wikimedia Commons.

Run once, on wifi, before the fair:

    python scripts/seed_scientists.py

The roster is curated rather than scraped, for two reasons:

1. Every entry gets a blurb, so the kiosk can say something interesting instead
   of just a name. Several of them tie straight back to the workshop -- LeCun
   invented the CNN this project is built on, Fei-Fei Li built the ImageNet the
   backbone is pretrained on, Rosenblatt's perceptron is literally week 1.

2. The roster is deliberately diverse across gender, ethnicity and era. A face
   recognition embedding is not demographically neutral, so a lopsided gallery
   would produce lopsided matches. Note that Buolamwini and Gebru are the
   researchers who documented exactly that bias -- which makes them the single
   best possible thing this demo can match a student to.

Images are the Wikipedia lead image, which is Commons-hosted and public domain
or CC. The license note per person goes into meta.json; check it before you put
a screenshot on social media.
"""

# System imports
import os
import sys
import json
import time
import urllib.error
import urllib.parse
import urllib.request

# Our imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from face_match_kiosk.configs import GALLERIES_DIR


USER_AGENT = 'cougar-ai-kiosk/1.0 (UH student club educational demo)'
SUMMARY_API = 'https://en.wikipedia.org/api/rest_v1/page/summary/'

# Wikimedia serves only a whitelisted set of thumbnail widths, and it rate-limits
# full-size originals hard (HTTP 429) while explicitly asking clients to use
# thumbnails instead. 500px is on the whitelist and is far more resolution than a
# 112x112 face crop needs.
THUMB_WIDTH = 500

# Politeness / backoff, since a 429 mid-run is the normal failure here.
REQUEST_DELAY_SECONDS = 1.0
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 5.0

ROSTER = [
    ('Alan Turing', 'alan_turing',
     'Defined what a computer even is, in 1936.',
     'Mathematics'),
    ('Ada Lovelace', 'ada_lovelace',
     'Wrote the first algorithm, a century before the first computer.',
     'Mathematics'),
    ('Grace Hopper', 'grace_hopper',
     'Built the first compiler. Popularized the word "bug".',
     'Mathematics & Physics'),
    ('Frank Rosenblatt', 'frank_rosenblatt',
     'Built the perceptron in 1958 -- that is week 1 of our workshop.',
     'Psychology'),
    ('Claude Shannon', 'claude_shannon',
     'Invented information theory. Everything digital rests on it.',
     'Electrical Engineering'),
    ('John McCarthy (computer scientist)', 'john_mccarthy',
     'Coined the term "artificial intelligence" in 1955.',
     'Mathematics'),
    ('Marvin Minsky', 'marvin_minsky',
     'Co-founded the MIT AI Lab. Wrote the book that paused neural nets.',
     'Mathematics'),
    ('Geoffrey Hinton', 'geoffrey_hinton',
     'Made backpropagation work. Nobel Prize, 2024.',
     'Experimental Psychology'),
    ('Yann LeCun', 'yann_lecun',
     'Invented the convolutional neural network this very app runs on.',
     'Electrical Engineering'),
    ('Yoshua Bengio', 'yoshua_bengio',
     'Shared the Turing Award for deep learning.',
     'Computer Engineering'),
    ('Fei-Fei Li', 'fei_fei_li',
     'Built ImageNet -- the dataset our backbone was pretrained on.',
     'Physics'),
    ('Andrew Ng', 'andrew_ng',
     'Taught machine learning to millions of people online.',
     'Computer Science'),
    ('Timnit Gebru', 'timnit_gebru',
     'Exposed how large AI models encode bias, and paid for it.',
     'Electrical Engineering'),
    ('Joy Buolamwini', 'joy_buolamwini',
     'Proved face recognition fails on dark and female faces. Ask us about this one.',
     'Computer Science'),
    ('Demis Hassabis', 'demis_hassabis',
     'AlphaGo, then AlphaFold. Nobel Prize, 2024.',
     'Computer Science'),
    ('Judea Pearl', 'judea_pearl',
     'Gave AI a mathematics of cause and effect.',
     'Electrical Engineering'),
    ('Katie Bouman', 'katie_bouman',
     'Wrote the imaging algorithm that photographed a black hole.',
     'Electrical Engineering'),
    ('Cynthia Breazeal', 'cynthia_breazeal',
     'Pioneered robots that read human social cues.',
     'Electrical Engineering'),
]


def _get(url, timeout=45):
    """GET with backoff, because Wikimedia answers 429 under any real load."""
    last_error = None

    for attempt in range(MAX_ATTEMPTS):
        request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in (429, 503):
                raise
            time.sleep(BACKOFF_SECONDS * (attempt + 1))
        except Exception as error:
            last_error = error
            time.sleep(BACKOFF_SECONDS * (attempt + 1))

    raise last_error


def _strip_query(url):
    """The API decorates image URLs with utm_* params; they break width math."""
    return urllib.parse.urljoin(url, urllib.parse.urlparse(url).path)


def _best_image_url(summary):
    """Prefer a whitelisted-width thumbnail; originals get rate-limited."""
    thumbnail = (summary.get('thumbnail') or {}).get('source')
    if thumbnail:
        url = _strip_query(thumbnail)
        base, last = url.rsplit('/', 1)
        if 'px-' in last:
            return '%s/%dpx-%s' % (base, THUMB_WIDTH, last.split('px-', 1)[1])
        return url

    original = (summary.get('originalimage') or {}).get('source')
    return _strip_query(original) if original else None


def _display_name(title):
    """Drop the Wikipedia disambiguator, e.g. 'X (computer scientist)' -> 'X'."""
    return title.split(' (')[0].strip()


def fetch_person(title, slug, blurb, directory, extra=None):
    """Download one portrait. `extra` is merged into the meta.json record.

    seed_people.py reuses this and passes the person's major through `extra`.
    """
    summary_url = SUMMARY_API + urllib.parse.quote(title.replace(' ', '_'))
    summary = json.loads(_get(summary_url))

    image_url = _best_image_url(summary)
    if not image_url:
        return None, 'no lead image on the Wikipedia page'

    extension = os.path.splitext(urllib.parse.urlparse(image_url).path)[1].lower()
    if extension not in ('.jpg', '.jpeg', '.png'):
        extension = '.jpg'

    filename = slug + extension
    destination = os.path.join(directory, filename)

    record = {'file': filename, 'name': _display_name(title), 'blurb': blurb,
              'credit': 'Wikimedia Commons via en.wikipedia.org'}
    record.update(extra or {})

    if os.path.exists(destination) and os.path.getsize(destination) > 20_000:
        return record, 'skip'

    data = _get(image_url)
    if len(data) < 5_000:
        return None, 'image suspiciously small (%d bytes)' % len(data)

    with open(destination, 'wb') as output_file:
        output_file.write(data)

    return record, 'ok'


def main():
    directory = os.path.join(GALLERIES_DIR, 'scientists')
    os.makedirs(directory, exist_ok=True)

    records, failures = [], []

    for title, slug, blurb, major in ROSTER:
        try:
            record, status = fetch_person(title, slug, blurb, directory,
                                          extra={'major': major})
        except Exception as error:
            record, status = None, str(error)

        if record is None:
            print('  [fail] %-20s %s' % (title, status))
            failures.append(title)
        else:
            print('  [%-4s] %s' % (status, title))
            records.append(record)

        # Be a polite API citizen.
        time.sleep(REQUEST_DELAY_SECONDS)

    meta_path = os.path.join(directory, 'meta.json')
    with open(meta_path, 'w', encoding='utf-8') as meta_file:
        json.dump(records, meta_file, indent=2, ensure_ascii=False)

    print()
    print('%d portrait(s) in %s' % (len(records), directory))
    print('meta.json written with names, blurbs and credits.')
    if failures:
        print('No image found for: %s' % ', '.join(failures))
    print()
    print('Next: python scripts/build_gallery.py',
     'Mathematics')
    return 0


if __name__ == '__main__':
    sys.exit(main())
