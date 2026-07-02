"""
ingest.py
---------
Builds the three indexes the pipeline needs from your raw article JSON files:
  1. Chroma vector store   (data/chroma/)
  2. BM25 keyword index    (data/bm25_index.pkl)
  3. Knowledge graph       (data/graph/f1_knowledge.pkl)

Uses SEMANTIC CHUNKING (chunking.py) instead of fixed-size splitting, so each
chunk stays topically coherent.

--------------------------------------------------------------------------
EXPECTED INPUT FORMAT
--------------------------------------------------------------------------
Two input formats are supported side by side in data/raw_articles/:

  * .txt  — scraped Wikipedia-style files with an "ARTICLE:/CATEGORY:/URL:"
            header and "##"/"###" section headings. Parsed automatically by
            wiki_txt_parser.py — no manual conversion needed. Entity tags
            and season ranges are inferred automatically for these.

  * .json — structured articles you build/curate yourself, shaped like the
            example below (lets you hand-specify entity_ids precisely).

Put one file per article in data/raw_articles/. JSON example:

{
  "article_id": "ayrton_senna_bio",
  "article_title": "Ayrton Senna",
  "category": "driver_biography",
  "source_url": "https://example.com/senna",
  "season_range": [1984, 1994],
  "sections": [
    {
      "section_title": "McLaren years",
      "text": "Long-form article text goes here. Multiple sentences and
               paragraphs. This gets semantically chunked automatically.",
      "entity_ids": ["ayrton_senna", "mclaren"]
    },
    {
      "section_title": "Rivalry with Prost",
      "text": "...",
      "entity_ids": ["ayrton_senna", "alain_prost"]
    }
  ]
}

`entity_ids` should use the same canonical ids as config.ENTITY_MAP values
(e.g. "ayrton_senna", "mclaren"). If you don't know them yet, just tag what
you can — retrieval still works fine without graph tags, it just won't get
the knowledge-graph boost for that chunk.

--------------------------------------------------------------------------
OPTIONAL: knowledge graph entities/relations
--------------------------------------------------------------------------
To populate the knowledge graph (drivers, teams, championships, rivalries),
add two more files to data/raw_articles/:

data/raw_articles/entities.json:
[
  {"id": "ayrton_senna", "label": "Ayrton Senna", "type": "driver"},
  {"id": "mclaren", "label": "McLaren", "type": "team"}
]

data/raw_articles/relations.json:
[
  {"source": "ayrton_senna", "target": "mclaren", "relation": "DROVE_FOR", "year_start": 1988, "year_end": 1993},
  {"source": "ayrton_senna", "target": "alain_prost", "relation": "RIVAL_OF", "label": "1988-1989 McLaren rivalry"}
]

If these files are absent, ingest.py still builds the vector + BM25 indexes;
it just creates an empty graph (graph-based retrieval and knowledge-graph
context will simply contribute nothing until you add these files).

--------------------------------------------------------------------------
RUN
--------------------------------------------------------------------------
    python ingest.py
"""
import json
import pickle
from pathlib import Path

import chromadb
import networkx as nx
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    RAW_ARTICLES_DIR, CHROMA_DIR, BM25_PATH, GRAPH_DIR, PICKLE_PATH,
    COLLECTION, EMBED_MODEL, log,
)
from chunking import semantic_chunk
from wiki_txt_parser import parse_txt_file


def _load_articles():
    articles = []
    if not RAW_ARTICLES_DIR.exists():
        raise FileNotFoundError(
            f"'{RAW_ARTICLES_DIR}' does not exist. Create it and add your article "
            f"files (see the docstring at the top of ingest.py for the format)."
        )

    # JSON articles (entities.json / relations.json are graph config, not articles)
    for path in sorted(RAW_ARTICLES_DIR.rglob("*.json")):
        if path.name in ("entities.json", "relations.json"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            articles.append(json.load(f))

    # Scraped .txt articles (ARTICLE/CATEGORY/URL header + ## headings)
    txt_paths = sorted(RAW_ARTICLES_DIR.rglob("*.txt"))
    for path in txt_paths:
        try:
            articles.append(parse_txt_file(path))
        except Exception as e:
            log.warning(f"Failed to parse '{path.name}': {e}")

    if not articles:
        raise ValueError(f"No article .json or .txt files found in '{RAW_ARTICLES_DIR}'.")
    log.info(f"Loaded {len(articles)} article file(s) ({len(txt_paths)} from .txt)")
    return articles


def build_chunks(articles, model: SentenceTransformer):
    """Semantically chunk every section of every article into flat chunk dicts."""
    all_chunks = []
    for article in articles:
        article_id = article["article_id"]
        article_title = article.get("article_title", article_id)
        category = article.get("category", "")
        source_url = article.get("source_url", "")
        season_range = article.get("season_range")

        for sec_idx, section in enumerate(article.get("sections", [])):
            section_title = section.get("section_title", "")
            text = section.get("text", "")
            entity_ids = section.get("entity_ids", [])
            if not text.strip():
                continue

            pieces = semantic_chunk(text, model)
            for chunk_idx, piece in enumerate(pieces):
                chunk_id = f"{article_id}__{sec_idx}__{chunk_idx}"
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": piece,
                    "article_id": article_id,
                    "article_title": article_title,
                    "category": category,
                    "section_title": section_title,
                    "source_url": source_url,
                    "entity_ids": entity_ids,
                    "season_range": season_range,
                })

    log.info(f"Produced {len(all_chunks)} semantic chunks from {len(articles)} articles")
    return all_chunks


def build_vector_store(chunks, model: SentenceTransformer):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()

    ids = [c["chunk_id"] for c in chunks]
    metadatas = []
    for c in chunks:
        season = c.get("season_range") or [0, 0]
        metadatas.append({
            "article_id": c["article_id"],
            "article_title": c["article_title"],
            "category": c["category"],
            "section_title": c["section_title"],
            "source_url": c["source_url"],
            "entity_ids": ",".join(c["entity_ids"]) if c["entity_ids"] else "",
            "season_start": season[0] if season else 0,
            "season_end": season[1] if season else 0,
        })

    # Chroma has a batch-size limit; insert in chunks of 500 to be safe.
    BATCH = 500
    for i in range(0, len(ids), BATCH):
        collection.add(
            ids=ids[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH],
            documents=texts[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
        )
    log.info(f"Vector store built: {collection.count()} chunks in Chroma collection '{COLLECTION}'")


def build_bm25_index(chunks):
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"index": bm25, "chunks": chunks}, f)
    log.info(f"BM25 index built: {len(chunks)} chunks -> {BM25_PATH}")


def build_knowledge_graph():
    G = nx.DiGraph()

    entities_path = RAW_ARTICLES_DIR / "entities.json"
    relations_path = RAW_ARTICLES_DIR / "relations.json"

    if entities_path.exists():
        with open(entities_path, "r", encoding="utf-8") as f:
            entities = json.load(f)
        for e in entities:
            G.add_node(e["id"], label=e.get("label", e["id"]), type=e.get("type", "entity"))
        log.info(f"Added {len(entities)} entities to knowledge graph")
    else:
        log.warning(f"No '{entities_path}' found — knowledge graph will have no nodes.")

    if relations_path.exists():
        with open(relations_path, "r", encoding="utf-8") as f:
            relations = json.load(f)
        for r in relations:
            G.add_edge(r["source"], r["target"], **{k: v for k, v in r.items() if k not in ("source", "target")})
        log.info(f"Added {len(relations)} relations to knowledge graph")
    else:
        log.warning(f"No '{relations_path}' found — knowledge graph will have no edges.")

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(G, f)
    log.info(f"Knowledge graph saved -> {PICKLE_PATH} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


def main():
    log.info("Starting ingest...")
    model = SentenceTransformer(EMBED_MODEL)

    articles = _load_articles()
    chunks = build_chunks(articles, model)

    build_vector_store(chunks, model)
    build_bm25_index(chunks)
    build_knowledge_graph()

    log.info("Ingest complete. You can now run: python api.py")


if __name__ == "__main__":
    main()
