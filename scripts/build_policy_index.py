from pathlib import Path
import json
import pickle
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_PATH = PROJECT_ROOT / "policies" / "return_policy.md"

INDEX_DIR = PROJECT_ROOT / "rag" / "index"

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD POLICY
# ============================================================

print("=" * 70)
print("TRUSTLOOP — POLICY INDEX BUILDER")
print("=" * 70)

if not POLICY_PATH.exists():
    raise FileNotFoundError(
        f"Policy file not found: {POLICY_PATH}"
    )

policy_text = POLICY_PATH.read_text(
    encoding="utf-8"
)

print(
    f"Loaded policy: {POLICY_PATH}"
)


# ============================================================
# CHUNK POLICY
# ============================================================

def chunk_policy(text):
    """
    Split the policy by markdown headings.

    Each section becomes one retrievable policy chunk.
    """

    sections = re.split(
        r"\n(?=##\s+)",
        text.strip()
    )

    chunks = []

    for section in sections:

        section = section.strip()

        if not section:
            continue

        chunks.append(section)

    return chunks


chunks = chunk_policy(
    policy_text
)

if not chunks:
    raise ValueError(
        "No policy chunks were created."
    )


print(
    f"Policy chunks created: {len(chunks)}"
)


for i, chunk in enumerate(chunks):

    first_line = chunk.splitlines()[0]

    print(
        f"  [{i}] {first_line}"
    )


# ============================================================
# BM25 INDEX
# ============================================================

def tokenize(text):
    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


tokenized_chunks = [
    tokenize(chunk)
    for chunk in chunks
]


bm25 = BM25Okapi(
    tokenized_chunks
)


with open(
    INDEX_DIR / "bm25.pkl",
    "wb"
) as f:

    pickle.dump(
        bm25,
        f
    )


# ============================================================
# FAISS INDEX
# ============================================================

"""
For this first version we use a deterministic
TF-IDF representation so the index is completely
local and does not require an external embedding API.
"""

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)


matrix = vectorizer.fit_transform(
    chunks
)


matrix = matrix.astype(
    np.float32
).toarray()


# Normalize vectors for cosine similarity.

norms = np.linalg.norm(
    matrix,
    axis=1,
    keepdims=True
)

norms[norms == 0] = 1.0

matrix = matrix / norms


dimension = matrix.shape[1]


faiss_index = faiss.IndexFlatIP(
    dimension
)

faiss_index.add(
    matrix
)


faiss.write_index(
    faiss_index,
    str(
        INDEX_DIR
        / "faiss.index"
    )
)


# ============================================================
# SAVE VECTORIZER
# ============================================================

with open(
    INDEX_DIR / "tfidf_vectorizer.pkl",
    "wb"
) as f:

    pickle.dump(
        vectorizer,
        f
    )


# ============================================================
# SAVE CHUNKS
# ============================================================

with open(
    INDEX_DIR / "chunks.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        chunks,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "policy_file": str(
        POLICY_PATH
    ),
    "chunk_count": len(chunks),
    "embedding_type": "TF-IDF",
    "faiss_metric": "inner_product",
    "retrieval_methods": [
        "BM25",
        "FAISS"
    ],
}


with open(
    INDEX_DIR / "metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("POLICY INDEX BUILD COMPLETED")
print("=" * 70)

print("\nGenerated:")

print(
    "rag/index/bm25.pkl"
)

print(
    "rag/index/faiss.index"
)

print(
    "rag/index/tfidf_vectorizer.pkl"
)

print(
    "rag/index/chunks.json"
)

print(
    "rag/index/metadata.json"
)

print("\nSTATUS: PASS")