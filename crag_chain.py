"""
crag_chain.py
-------------
F1CRAGChain: query decomposition -> hybrid retrieval -> rerank -> CRAG
confidence evaluation -> corrective action (refine / web search / both) ->
generation with Gemini.
"""
import os
import re
import json
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import spacy
from sentence_transformers import CrossEncoder
from google import genai

from config import (
    GEMINI_API_KEY, GEMINI_MODEL, SPACY_MODEL, CROSS_ENCODER, CACHE_PATH,
    RETRIEVAL_TOP_K, RERANK_TOP_K, CRAG_UPPER_THRESHOLD, CRAG_LOWER_THRESHOLD,
    STRIP_KEEP_THRESHOLD, MAX_STRIPS_PER_DOC, ENTITY_MAP, NER_TYPES, log,
)
from retriever import F1Retriever, RetrievedDoc
from graph_utils import load_graph, describe_entity


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class CRAGResult:
    answer: str
    sources: list = field(default_factory=list)
    sub_queries: list = field(default_factory=list)
    passages: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    retrieval_confidence: str = ""      # 'CORRECT' | 'AMBIGUOUS' | 'INCORRECT'
    used_web_search: bool = False


RAGResult = CRAGResult  # backwards-compatible alias


class F1CRAGChain:
    def __init__(self):
        log.info("Initialising F1CRAGChain...")

        if not GEMINI_API_KEY:
            log.warning("GEMINI_API_KEY environment variable is not set.")
        self._client = genai.Client(api_key=GEMINI_API_KEY)

        log.info(f"Loading spaCy model: {SPACY_MODEL}")
        try:
            self._nlp = spacy.load(SPACY_MODEL)
        except OSError:
            raise OSError(f"spaCy model '{SPACY_MODEL}' not found. Run: python -m spacy download {SPACY_MODEL}")

        log.info(f"Loading cross-encoder: {CROSS_ENCODER}")
        self._reranker = CrossEncoder(CROSS_ENCODER, max_length=512)

        self._retriever = F1Retriever()
        self._graph = load_graph()
        log.info("F1CRAGChain ready")

    # -----------------------------------------------------------------
    # Cache
    # -----------------------------------------------------------------
    def _load_cache(self) -> dict:
        if not os.path.exists(CACHE_PATH):
            return {}
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load cache: {e}")
            return {}

    def _save_to_cache(self, query: str, result: CRAGResult):
        cache = self._load_cache()
        passages_data = [
            {"chunk_id": d.chunk_id, "text": d.text, "metadata": d.metadata, "score": d.score}
            for d in result.passages
        ]
        cache[query] = {
            "answer": result.answer,
            "sources": result.sources,
            "sub_queries": result.sub_queries,
            "entities": result.entities,
            "passages": passages_data,
            "retrieval_confidence": result.retrieval_confidence,
            "used_web_search": result.used_web_search,
        }
        try:
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.warning(f"Failed to save cache: {e}")

    # -----------------------------------------------------------------
    # Main entrypoint
    # -----------------------------------------------------------------
    def run(self, question: str) -> CRAGResult:
        log.info(f"Question: {question!r}")

        normalized = " ".join(question.strip().lower().split())
        cache = self._load_cache()
        if normalized in cache:
            log.info("Cache hit! Returning cached answer.")
            cached = cache[normalized]
            passages = [
                RetrievedDoc(chunk_id=p["chunk_id"], text=p["text"], metadata=p["metadata"], score=p.get("score", 0.0))
                for p in cached.get("passages", [])
            ]
            return CRAGResult(
                answer=cached["answer"], sources=cached["sources"], sub_queries=cached["sub_queries"],
                entities=cached["entities"], passages=passages,
                retrieval_confidence=cached.get("retrieval_confidence", ""),
                used_web_search=cached.get("used_web_search", False),
            )

        log.info("Cache miss. Performing retrieval and generation...")
        sub_queries = self._decompose(question)
        log.info(f"Sub-queries: {sub_queries}")

        entity_ids = self._extract_entities(question)
        year_range = self._extract_year_range(question)
        log.info(f"Entities: {entity_ids}  |  Year range: {year_range}")

        docs = self._retriever.retrieve_multi(
            queries=sub_queries, entity_ids=entity_ids, top_k=RETRIEVAL_TOP_K, year_filter=year_range
        )

        top_docs = self._rerank(question, docs, top_k=RERANK_TOP_K)

        confidence, doc_scores = self._evaluate_confidence(question, top_docs)
        log.info(f"CRAG confidence: {confidence}  |  doc scores: {[round(s, 3) for s in doc_scores]}")

        web_context = ""
        if confidence == "CORRECT":
            refined_docs = self._refine_knowledge(question, top_docs, doc_scores)
        elif confidence == "INCORRECT":
            refined_docs = []
            log.info("CRAG: local retrieval discarded, no web search fallback configured")
        else:  # AMBIGUOUS
            refined_docs = self._refine_knowledge(question, top_docs, doc_scores)
            log.info("CRAG: local retrieval is ambiguous, using refined local knowledge only")

        graph_context = self._build_graph_context(entity_ids)
        answer = self._generate(question, refined_docs, graph_context, web_context)
        sources = self._format_sources(top_docs)

        result = CRAGResult(
            answer=answer, sources=sources, sub_queries=sub_queries, passages=top_docs,
            entities=entity_ids, retrieval_confidence=confidence, used_web_search=bool(web_context),
        )

        self._save_to_cache(normalized, result)
        return result

    # -----------------------------------------------------------------
    # Decomposition
    # -----------------------------------------------------------------
    _DECOMPOSE_SYSTEM = (
        "You are a Formula One research assistant.\n"
        "Break the user's question into 2-4 concise, atomic sub-queries that together\n"
        "cover all aspects of the original question.\n\n"
        "Rules:\n"
        "- Each sub-query must be self-contained and answerable independently.\n"
        "- Prefer specific sub-queries over vague ones.\n"
        "- If the question is already simple and specific, return it as-is in a list.\n"
        "- Respond ONLY with a JSON array of strings. No preamble, no explanation.\n\n"
        "Example:\n"
        "Question: \"How did Senna and Prost's rivalry affect McLaren in 1989?\"\n"
        "Response: [\"What happened between Senna and Prost at the 1989 Japanese GP?\", "
        "\"What was the team atmosphere at McLaren in 1989?\", "
        "\"How did the 1989 title outcome affect McLaren?\"]"
    )

    def _decompose(self, question: str) -> list:
        try:
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"{self._DECOMPOSE_SYSTEM}\n\nQuestion: \"{question}\"",
                config=genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
                ),
            )
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
                return parsed[:2]
            return [question]
        except Exception as e:
            log.warning(f"Decomposition failed ({e}), using original question")
            return [question]

    def _extract_entities(self, text: str) -> list:
        found = []
        doc = self._nlp(text)
        for ent in doc.ents:
            if ent.label_ in NER_TYPES:
                span_lower = ent.text.lower().strip()
                cid = ENTITY_MAP.get(span_lower)
                if not cid:
                    last_word = span_lower.split()[-1]
                    cid = ENTITY_MAP.get(last_word)
                if cid and cid not in found:
                    found.append(cid)

        text_lower = text.lower()
        for keyword, cid in ENTITY_MAP.items():
            if keyword in text_lower and cid not in found:
                found.append(cid)
        return found

    def _extract_year_range(self, text: str) -> Optional[Tuple[int, int]]:
        years = [int(y) for y in re.findall(r"\b(19[5-9]\d|20\d{2})\b", text)]
        if not years:
            return None
        return min(years), max(years)

    def _rerank(self, question: str, docs: list, top_k: int) -> list:
        if not docs:
            return []
        if len(docs) <= top_k:
            return docs
        pairs = [(question, doc.text) for doc in docs]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]

    # -----------------------------------------------------------------
    # CRAG: retrieval evaluator
    # -----------------------------------------------------------------
    def _score_pairs(self, question: str, texts: list) -> list:
        if not texts:
            return []
        pairs = [(question, t) for t in texts]
        raw_logits = self._reranker.predict(pairs)
        return [float(_sigmoid(x)) for x in raw_logits]

    def _evaluate_confidence(self, question: str, docs: list) -> Tuple[str, list]:
        if not docs:
            return "INCORRECT", []
        scores = self._score_pairs(question, [d.text for d in docs])
        best = max(scores)
        if best >= CRAG_UPPER_THRESHOLD:
            label = "CORRECT"
        elif best < CRAG_LOWER_THRESHOLD:
            label = "INCORRECT"
        else:
            label = "AMBIGUOUS"
        return label, scores

    # -----------------------------------------------------------------
    # CRAG: knowledge refinement (decompose -> filter -> recompose)
    # -----------------------------------------------------------------
    def _decompose_recompose(self, question: str, text: str) -> str:
        strips = re.split(r"(?<=[.!?])\s+", text.strip())
        strips = [s.strip() for s in strips if len(s.split()) > 3]
        if not strips:
            return text.strip()

        scores = self._score_pairs(question, strips)
        ranked = sorted(zip(scores, strips), key=lambda x: x[0], reverse=True)

        keep = [s for score, s in ranked if score >= STRIP_KEEP_THRESHOLD][:MAX_STRIPS_PER_DOC]
        if not keep:
            keep = [ranked[0][1]]

        kept_set = set(keep)
        ordered = [s for s in strips if s in kept_set]
        return " ".join(ordered)

    def _refine_knowledge(self, question: str, docs: list, doc_scores: list) -> list:
        refined = []
        for doc, score in zip(docs, doc_scores):
            if score < CRAG_LOWER_THRESHOLD:
                continue
            refined_text = self._decompose_recompose(question, doc.text)
            if not refined_text:
                continue
            refined.append(RetrievedDoc(chunk_id=doc.chunk_id, text=refined_text, metadata=doc.metadata, score=score))
        return refined

    # -----------------------------------------------------------------
    # CRAG: web search fallback
    # -----------------------------------------------------------------
    def _web_search(self, question: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=(
                    "You are an F1 research assistant. Use Google Search to find "
                    f"accurate, up-to-date facts that help answer this question: \"{question}\"\n"
                    "Return a concise bullet list of the most relevant facts you find. "
                    "No commentary, no preamble."
                ),
                config=genai.types.GenerateContentConfig(
                    tools=[genai.types.Tool(google_search=genai.types.GoogleSearch())],
                    thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return response.text.strip()
        except Exception as e:
            log.warning(f"CRAG web search fallback failed: {e}")
            return ""

    def _build_graph_context(self, entity_ids: list) -> str:
        parts = [describe_entity(self._graph, eid) for eid in entity_ids]
        parts = [p for p in parts if p]
        if not parts:
            return ""
        return "Knowledge graph context:\n" + "\n\n".join(parts)

    _GENERATE_SYSTEM = (
        "You are F1 Historian AI, an expert on Formula One history.\n\n"
        "Answer the user's question using ONLY the provided passages and/or knowledge graph\n"
        "context below.\n"
        "Be accurate, specific, and cite sources inline using [Source N] notation.\n\n"
        "Guidelines:\n"
        "- Lead with the direct answer.\n"
        "- Use specific facts: names, years, race results, points.\n"
        "- If the passages and knowledge graph context lack enough information to answer the question, respond with exactly: \"I am sorry, I don't have enough information in the indexed documents.\" - do not invent facts.\n"
        "- Keep the answer focused and well-structured.\n"
        "- Cite at least one source per factual claim."
    )

    def _generate(self, question: str, docs: list, graph_context: str, web_context: str = "") -> str:
        if not docs:
            return "I am Sorry, I don't have enough information in the indexed documents"

        user_content = ""
        if graph_context:
            user_content += graph_context + "\n\n"

        if docs:
            sources_text = "\n\n".join(
                f"[Source {i+1}] ({d.metadata.get('article_title', '?')} - {d.metadata.get('section_title', '')})\n{d.text}"
                for i, d in enumerate(docs)
            )
            user_content += f"Sources:\n{sources_text}\n\n"

        if web_context:
            user_content += f"Web Search Findings:\n{web_context}\n\n"

        user_content += f"Question: {question}"

        try:
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"{self._GENERATE_SYSTEM}\n\nContext and Sources:\n{user_content}",
                config=genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
                ),
            )
            return response.text.strip()
        except Exception as e:
            log.error(f"Generation failed: {e}")
            return f"Error calling Gemini API: {e}"

    def _format_sources(self, docs: list) -> list:
        seen = set()
        sources = []
        for i, doc in enumerate(docs, start=1):
            url = doc.metadata.get("source_url", "")
            if url in seen:
                continue
            seen.add(url)
            sources.append({
                "index": i,
                "title": doc.metadata.get("article_title", "Unknown"),
                "section": doc.metadata.get("section_title", ""),
                "url": url,
            })
        return sources
