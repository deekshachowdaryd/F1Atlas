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

    print("-" * 60)
    print(f"q: {question}")
    print(f"took: {elapsed:.2f}s")
    print("-" * 60)
    print(f"sub-queries: {result.sub_queries}")
    print(f"entities: {result.entities if result.entities else 'none'}")
    print(f"crag confidence: {result.retrieval_confidence.lower() if result.retrieval_confidence else 'cached'}")
    print(f"web search used: {str(result.used_web_search).lower()}")

    print("\n-- retrieved documents --")
    for i, doc in enumerate(result.passages, 1):
        print(f"  doc {i}: [score: {doc.score:.4f}] '{doc.metadata.get('article_title', 'unknown')}' - {doc.metadata.get('section_title', 'unknown')}")
        snippet = doc.text.replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:120] + "..."
        print(f"    snippet: {snippet}")

    print("\n-- answer --")
    print(result.answer)
    print("\n-- sources --")
    for s in result.sources:
        print(f"  [{s['index']}] {s['title']} - {s['section']}")
        print(f"        {s['url']}")
    print("-" * 60 + "\n")


def main():
    chain = F1CRAGChain()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        ask(chain, question)
        return

    print("f1 rag cli - interactive mode (type 'quit' to exit)\n")
    while True:
        question = input("ask> ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        ask(chain, question)


if __name__ == "__main__":
    main()
