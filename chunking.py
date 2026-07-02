"""
chunking.py
-----------
Semantic chunking: instead of splitting article text every N characters
(which cuts sentences in half and mixes unrelated ideas into one chunk),
we split on *meaning boundaries*.

How it works:
1. Split the text into sentences.
2. Embed every sentence with the same SentenceTransformer used for retrieval.
3. Walk through the sentences in order, comparing each sentence's embedding
   to the previous one with cosine similarity.
4. As long as consecutive sentences stay "on topic" (similarity above
   CHUNK_SIMILARITY_THRESHOLD), keep adding them to the current chunk.
5. When similarity drops (topic shift) *and* the current chunk already has
   enough content, close the chunk and start a new one.
6. A hard character cap (CHUNK_MAX_CHARS) forces a split even mid-topic, so
   no chunk balloons in size; a minimum (CHUNK_MIN_CHARS) stops us from
   creating tiny fragment chunks.

This is a light dependency-free implementation (no langchain needed) that
reuses the embedding model already loaded elsewhere in the pipeline.
"""
import re
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_SIMILARITY_THRESHOLD,
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    log,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # embeddings are already L2-normalized, so dot product == cosine similarity
    return float(np.dot(a, b))


def semantic_chunk(text: str, model: SentenceTransformer) -> List[str]:
    """
    Splits `text` into a list of semantically-coherent chunk strings.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=True)

    chunks: List[str] = []
    current_sentences = [sentences[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        sim = _cosine_sim(embeddings[i - 1], embeddings[i])
        sentence = sentences[i]
        would_be_len = current_len + 1 + len(sentence)

        topic_shift = sim < CHUNK_SIMILARITY_THRESHOLD
        big_enough_to_split = current_len >= CHUNK_MIN_CHARS
        over_hard_cap = would_be_len > CHUNK_MAX_CHARS

        if (topic_shift and big_enough_to_split) or over_hard_cap:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_len = len(sentence)
        else:
            current_sentences.append(sentence)
            current_len = would_be_len

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


if __name__ == "__main__":
    # Quick manual smoke test:
    #   python chunking.py
    sample = (
        "Ayrton Senna joined McLaren in 1988. He won the championship that year "
        "after a season-long battle with teammate Alain Prost. The two drivers "
        "had very different driving styles. Senna was known for his raw speed "
        "in qualifying. Meanwhile, Formula One's technical regulations were "
        "changing rapidly in the late 1980s. Turbo engines were banned ahead of "
        "the 1989 season, forcing teams to redesign their cars around naturally "
        "aspirated engines. This shook up the competitive order significantly."
    )
    m = SentenceTransformer("all-MiniLM-L6-v2")
    for idx, c in enumerate(semantic_chunk(sample, m), 1):
        log.info(f"Chunk {idx}: {c}")
