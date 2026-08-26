import numpy as np

from src.retrieval.embeddings import embed_texts


class _FakeModel:
    """Deterministic stand-in for a real sentence-transformers model:
    embeds each text as its length and character-sum, so we can assert on
    exact output without downloading anything."""

    def encode(self, texts, **kwargs):
        return np.array([[len(t), sum(map(ord, t)) % 997] for t in texts], dtype="float32")


def test_embed_texts_returns_one_vector_per_text():
    model = _FakeModel()
    vectors = embed_texts(["hello", "a longer piece of text"], model)

    assert vectors.shape == (2, 2)
    assert vectors[0][0] == 5


def test_embed_texts_handles_empty_list():
    model = _FakeModel()
    vectors = embed_texts([], model)

    assert vectors.shape == (0,)
