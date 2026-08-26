"""Text embedding for ADR retrieval.

Import order matters on this machine: `import sentence_transformers` before
`import torch` (importing torch first causes a native segfault on import —
see PROGRESS.md, Session 2026-08-26). load_embedding_model is the only
place torch gets imported transitively, so the sentence_transformers import
below must stay first in this module.
"""
import sentence_transformers  # noqa: F401  (import before torch — see docstring)
import numpy as np

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def embed_texts(texts: list[str], model) -> np.ndarray:
    if not texts:
        return np.array([])
    return np.asarray(model.encode(texts))


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)
