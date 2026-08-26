import numpy as np

from src.retrieval.index import VectorIndex


def test_query_returns_closest_vectors_first():
    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1],
    ])
    index = VectorIndex.build(embeddings)

    results = index.query(np.array([1.0, 0.0]), k=2)

    assert [idx for idx, _score in results] == [0, 2]


def test_query_k_larger_than_dataset_returns_all():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    index = VectorIndex.build(embeddings)

    results = index.query(np.array([1.0, 0.0]), k=10)

    assert len(results) == 2


def test_query_k_zero_returns_empty_list():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    index = VectorIndex.build(embeddings)

    assert index.query(np.array([1.0, 0.0]), k=0) == []


def test_query_negative_k_returns_empty_list():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    index = VectorIndex.build(embeddings)

    assert index.query(np.array([1.0, 0.0]), k=-1) == []
