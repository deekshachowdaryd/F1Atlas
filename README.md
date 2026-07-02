# F1 Historian 🏎️ — Simple Corrective RAG (CRAG) Engine

F1 Historian is a Formula One Q&A assistant built using a Corrective Retrieval-Augmented Generation (CRAG) backend pipeline. It uses hybrid retrieval (Vector Search + Keyword Matching + Knowledge Graph) fused via Reciprocal Rank Fusion (RRF), re-ranks results with a Cross-Encoder, and generates accurate answers with inline source citations. When no matching context is found, it safely responds with a fallback message: *"I am sorry, I don't have enough information in the indexed documents."*

---

## 🌟 Key Features

* **Data Scraping:** Automated script fetches raw Wikipedia article data, formatted with sections and F1 tags.
* **Custom Semantic Chunking:** Splits text at natural meaning boundaries (using sentence embeddings) to keep chunks contextually coherent.
* **Hybrid Retrieval:** Fuses Chroma Vector Search, BM25 Keyword Search, and NetworkX Knowledge Graph lookups.
* **Cross-Encoder Re-ranking:** Re-scores context passages for high-fidelity ranking.
* **Local Fallback:** Prevents hallucination by gracefully failing when the requested topic is missing from the database.

---

## 📂 Project Directory Structure

```
f1_rag_backend/
├── config.py                  # Settings, thresholds, and entity maps
├── data_scrapper.py           # Scrapes raw F1 articles from Wikipedia into structured .txt files
├── chunking.py                # Semantic chunker using sentence embeddings
├── wiki_txt_parser.py         # Parsers for scraped Wikipedia text documents
├── ingest.py                  # Builds indices (Chroma vector store, BM25, Graph)
├── graph_utils.py             # NetworkX knowledge graph helper functions
├── retriever.py               # Hybrid retriever fusing Vector, BM25, and Graph with RRF
├── crag_chain.py              # CRAG Pipeline: Decomposition, Reranking, Generation
├── api.py                     # FastAPI web server serving the /query endpoint
├── cli.py                     # Interactive terminal debugging utility
├── requirements.txt           # Python dependencies
└── .env.example               # Example environment configuration (requires GEMINI_API_KEY)
```

---

## 🚀 Setup & Execution

### 1. Installation & Environment Setup
Clone the repository and set up a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Configure your environment variables:
```bash
cp .env.example .env
# Open .env and add your GEMINI_API_KEY
```

### 2. Fetch the Data (Run Scraper)
Scrape F1 articles from Wikipedia and output them directly into the raw articles directory:
```bash
python data_scrapper.py --out data/raw_articles
```

### 3. Build the Indexes
Index the scraped text files into ChromaDB, BM25, and the Knowledge Graph:
```bash
python ingest.py
```

### 4. Test in the Terminal (CLI)
Start an interactive terminal chat or query a specific question:
```bash
python cli.py "How did the Senna and Prost rivalry affect McLaren in 1989?"
# Or for interactive mode:
python cli.py
```

### 5. Run the Web Server
Start the FastAPI server:
```bash
python api.py
```
Open `f1_historian_frontend.html` directly in your browser. Configure the endpoint URL to point to `http://localhost:8000/query` and begin exploring.
