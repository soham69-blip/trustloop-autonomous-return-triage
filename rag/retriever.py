from pathlib import Path
import json
import pickle
import re

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_DIR = PROJECT_ROOT / "rag" / "index"


# ============================================================
# LOAD INDEXES
# ============================================================

with open(INDEX_DIR / "bm25.pkl", "rb") as f:
    BM25 = pickle.load(f)

with open(INDEX_DIR / "tfidf_vectorizer.pkl", "rb") as f:
    VECTORIZER = pickle.load(f)

with open(INDEX_DIR / "chunks.json", "r", encoding="utf-8") as f:
    CHUNKS = json.load(f)

FAISS_INDEX = faiss.read_index(
    str(INDEX_DIR / "faiss.index")
)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(query, top_k=5):

    tokens = tokenize(query)

    scores = BM25.get_scores(tokens)

    ranked = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "index": int(i),
            "score": float(scores[i]),
            "text": CHUNKS[i],
            "source": "bm25",
        }
        for i in ranked
    ]


# ============================================================
# FAISS SEARCH
# ============================================================

def faiss_search(query, top_k=5):

    query_vector = VECTORIZER.transform(
        [query]
    ).astype(
        np.float32
    ).toarray()

    norm = np.linalg.norm(
        query_vector,
        axis=1,
        keepdims=True
    )

    norm[norm == 0] = 1.0

    query_vector = query_vector / norm

    scores, indices = FAISS_INDEX.search(
        query_vector,
        top_k
    )

    results = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index < 0:
            continue

        results.append(
            {
                "index": int(index),
                "score": float(score),
                "text": CHUNKS[index],
                "source": "faiss",
            }
        )

    return results


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def rrf_fusion(
    bm25_results,
    faiss_results,
    k=60,
    top_k=5,
):

    scores = {}
    documents = {}

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        idx = result["index"]

        scores[idx] = scores.get(
            idx,
            0.0
        ) + 1.0 / (k + rank)

        documents[idx] = result["text"]


    for rank, result in enumerate(
        faiss_results,
        start=1
    ):

        idx = result["index"]

        scores[idx] = scores.get(
            idx,
            0.0
        ) + 1.0 / (k + rank)

        documents[idx] = result["text"]


    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]


    return [
        {
            "index": int(idx),
            "score": float(score),
            "text": documents[idx],
        }
        for idx, score in ranked
    ]


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def retrieve_policy(
    query,
    top_k=5
):

    bm25_results = bm25_search(
        query,
        top_k=top_k
    )

    faiss_results = faiss_search(
        query,
        top_k=top_k
    )

    return rrf_fusion(
        bm25_results,
        faiss_results,
        top_k=top_k
    )


# ============================================================
# TEST MODE
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TRUSTLOOP — HYBRID POLICY RETRIEVER")
    print("=" * 70)

    test_queries = [
        "How many days does a customer have to return a product?",
        "What happens when multiple accounts are associated with a customer?",
        "What are the rules for high value products?",
        "When should a return be escalated to a human?",
    ]

    for query in test_queries:

        print("\n" + "-" * 70)
        print("QUERY:")
        print(query)

        results = retrieve_policy(
            query,
            top_k=3
        )

        print("\nTOP RESULTS:")

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\n[{rank}] "
                f"RRF score: "
                f"{result['score']:.6f}"
            )

            print(
                result["text"]
            )

    print("\n" + "=" * 70)
    print("RETRIEVER TEST COMPLETED")
    print("=" * 70)