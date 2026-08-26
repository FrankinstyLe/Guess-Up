"""Cosine top-k matching against a gallery.

Because both the probe and the gallery rows are L2-normalized, cosine similarity
is just a dot product, so the whole match is one matrix-vector multiply. That is
also the entire point of the Act 2 talk: the "AI decision" students are reacting
to is a single dot product.
"""

# Library imports
import numpy as np

# Our imports
from face_match_kiosk.configs import TOP_K


class Match:

    def __init__(self, entry, score, rank):
        self.entry = entry
        self.score = float(score)
        self.rank = rank

    @property
    def name(self):
        return self.entry.name

    def __repr__(self):
        return '<Match #%d %s %.4f>' % (self.rank, self.name, self.score)


def top_k(embedding, gallery, k=TOP_K):
    """Rank gallery entries against one probe embedding, best first."""
    if gallery is None or len(gallery) == 0:
        return []

    scores = gallery.matrix @ embedding
    k = min(k, len(scores))

    # argpartition to find the top k cheaply, then sort just those.
    candidates = np.argpartition(-scores, k - 1)[:k]
    candidates = candidates[np.argsort(-scores[candidates])]

    return [Match(gallery.entries[index], scores[index], rank + 1)
            for rank, index in enumerate(candidates)]


def spread(matches):
    """Gap between the #1 and #2 score: the Act 2 reveal in one number.

    A tiny spread means the model's confident-looking answer was very nearly a
    coin flip between two different people.
    """
    if len(matches) < 2:
        return 0.0
    return matches[0].score - matches[1].score
