"""
retriever.py
------------
Hybrid retriever: dense vector search (Chroma) + sparse BM25 + knowledge-graph
expansion, fused together with Reciprocal Rank Fusion (RRF).
"""
import pickle
from typing import Optional, List, Tuple

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR, BM25_PATH, COLLECTION, EMBED_MODEL,
    VECTOR_CANDIDATES, BM25_CANDIDATES, GRAPH_CANDIDATES, RRF_K, log,
)
from graph_utils import load_graph, get_related_entity_ids


class RetrievedDoc:
    __slots__ = ("chunk_id", "text", "metadata", "score")

    def __init__(self, chunk_id: str, text: str, metadata: dict, score: float = 0.0):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
        self.score = score

    def __repr__(self):
        title = self.metadata.get("article_title", "?")
        section = self.metadata.get("section_title", "")
        return f"<RetrievedDoc [{title} / {section}] score={self.score:.4f}>"


class F1Retriever:
    def __init__(self):
        log.info("Initialising F1Retriever...")
        self._model = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = client.get_collection(COLLECTION)
        log.info(f"ChromaDB collection loaded: {self._collection.count()} docs")

        with open(BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        self._bm25 = bm25_data["index"]
        self._bm25_chunks = bm25_data["chunks"]
        log.info(f"BM25 index loaded: {len(self._bm25_chunks)} docs")

        self._graph = load_graph()
        log.info("Knowledge graph loaded")

    def retrieve(self, query: str, entity_ids: Optional[List[str]] = None, top_k: int = 8,
                 year_filter: Optional[Tuple[int, int]] = None) -> List[RetrievedDoc]:
        entity_ids = entity_ids or []
        vector_results = self._vector_search(query, VECTOR_CANDIDATES, year_filter)
        bm25_results = self._bm25_search(query, BM25_CANDIDATES)
        graph_results = self._graph_search(query, entity_ids, GRAPH_CANDIDATES)

        fused = self._rrf_fuse([vector_results, bm25_results, graph_results], top_k=top_k)
        return fused

    def retrieve_multi(self, queries: List[str], entity_ids: Optional[List[str]] = None,
                        top_k: int = 8, year_filter: Optional[Tuple[int, int]] = None) -> List[RetrievedDoc]:
        all_result_lists = [
            self.retrieve(q, entity_ids=entity_ids, top_k=top_k, year_filter=year_filter)
            for q in queries
        ]
        return self._rrf_fuse(all_result_lists, top_k=top_k)

    def _vector_search(self, query: str, n: int, year_filter: Optional[Tuple[int, int]] = None) -> List[RetrievedDoc]:
        embedding = self._model.encode(query, normalize_embeddings=True).tolist()
        where = None
        if year_filter:
            start, end = year_filter
            where = {"$and": [{"season_start": {"$gte": start}}, {"season_end": {"$lte": end}}]}
        try:
            results = self._collection.query(
                query_embeddings=[embedding], n_results=n, where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            log.warning(f"Vector search failed: {e}")
            return []

        docs = []
        for chunk_id, text, meta, dist in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            docs.append(RetrievedDoc(chunk_id=chunk_id, text=text, metadata=meta, score=1.0 - dist))
        return docs

    def _bm25_search(self, query: str, n: int) -> List[RetrievedDoc]:
        tokenized = query.lower().split()
        scores = self._bm25.get_scores(tokenized)
        top_n_idx = np.argsort(scores)[::-1][:n]

        docs = []
        for idx in top_n_idx:
            if scores[idx] == 0:
                continue
            chunk = self._bm25_chunks[idx]
            meta = {
                "article_id": chunk.get("article_id", ""),
                "article_title": chunk.get("article_title", ""),
                "category": chunk.get("category", ""),
                "section_title": chunk.get("section_title", ""),
                "source_url": chunk.get("source_url", ""),
                "entity_ids": ",".join(chunk.get("entity_ids")) if chunk.get("entity_ids") else "",
                "season_start": chunk.get("season_range")[0] if chunk.get("season_range") else 0,
                "season_end": chunk.get("season_range")[1] if chunk.get("season_range") else 0,
            }
            docs.append(RetrievedDoc(chunk_id=chunk["chunk_id"], text=chunk["text"], metadata=meta, score=float(scores[idx])))
        return docs

    def _graph_search(self, query: str, entity_ids: List[str], n: int) -> List[RetrievedDoc]:
        if not entity_ids:
            return []
        expanded = set(entity_ids)
        for eid in entity_ids:
            neighbours = get_related_entity_ids(self._graph, eid, max_hops=1)
            expanded.update(neighbours)

        expanded_list = list(expanded)
        if not expanded_list:
            return []

        results = []
        seen = set()
        query_embedding = self._model.encode(query, normalize_embeddings=True).tolist()
        per_entity = max(1, n // min(len(expanded_list), 3))

        for eid in expanded_list[:3]:
            try:
                raw = self._collection.query(
                    query_embeddings=[query_embedding], n_results=per_entity,
                    where={"entity_ids": {"$contains": eid}},
                    include=["documents", "metadatas", "distances"],
                )
                for cid, text, meta, dist in zip(
                    raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
                ):
                    if cid not in seen:
                        seen.add(cid)
                        results.append(RetrievedDoc(chunk_id=cid, text=text, metadata=meta, score=1.0 - dist))
            except Exception as e:
                log.debug(f"Graph search skip for {eid}: {e}")

        return results[:n]

    def _rrf_fuse(self, result_lists: List[List[RetrievedDoc]], top_k: int) -> List[RetrievedDoc]:
        rrf_scores = {}
        doc_store = {}
        for ranked_list in result_lists:
            for rank, doc in enumerate(ranked_list, start=1):
                rrf_scores[doc.chunk_id] = rrf_scores.get(doc.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
                doc_store[doc.chunk_id] = doc

        fused = []
        for chunk_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            doc = doc_store[chunk_id]
            doc.score = score
            fused.append(doc)
        return fused[:top_k]
