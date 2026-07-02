"""
cli.py
------
Quick terminal testing, equivalent to the notebook's ask() cell.

RUN:
    python cli.py "who is he?"
    python cli.py                     (interactive loop)
"""
import sys
import time

from crag_chain import F1CRAGChain


def ask(chain: F1CRAGChain, question: str):
    start_time = time.time()
    result = chain.run(question)
    elapsed = time.time() - start_time

    print("=" * 80)
    print(f"QUESTION: {question}")
    print(f"Time Taken: {elapsed:.2f} seconds")
    print("=" * 80)
    print(f"Sub-queries Generated: {result.sub_queries}")
    print(f"Entities Detected: {result.entities if result.entities else 'none'}")
    print(f"CRAG Confidence: {result.retrieval_confidence or 'n/a (cached)'}")
    print(f"Web Search Used: {result.used_web_search}")

    print("\n-- Retrieved Docs (Top Reranked) -------------------")
    for i, doc in enumerate(result.passages, 1):
        print(f"  Doc {i}: [Score: {doc.score:.4f}] from '{doc.metadata.get('article_title', 'Unknown')}' - {doc.metadata.get('section_title', 'Unknown')}")
        snippet = doc.text.replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:120] + "..."
        print(f"    Snippet: {snippet}")

    print("\n-- Answer ------------------------------------------")
    print(result.answer)
    print("\n-- Sources Cited -----------------------------------")
    for s in result.sources:
        print(f"  [{s['index']}] {s['title']} - {s['section']}")
        print(f"        {s['url']}")
    print("=" * 80 + "\n")


def main():
    chain = F1CRAGChain()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        ask(chain, question)
        return

    print("F1 Historian CRAG — interactive mode. Type 'quit' to exit.\n")
    while True:
        question = input("Ask> ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        ask(chain, question)


if __name__ == "__main__":
    main()
