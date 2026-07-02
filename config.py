"""
config.py
---------
Central place for constants, environment loading, and logging setup.
Every other module imports from here so there's exactly one source of truth.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
    force=True
)
log = logging.getLogger("f1_rag")

# --------------------------------------------------------------------------
# API keys
# --------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Optional: if set, the FastAPI server requires "Authorization: Bearer <token>"
# This matches the "API Key" field in the frontend's config panel.
API_AUTH_TOKEN = os.environ.get("API_AUTH_TOKEN", "")

if not GEMINI_API_KEY:
    log.warning("GEMINI_API_KEY is not set. Put it in a .env file (see .env.example).")

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_DIR = Path("data")
RAW_ARTICLES_DIR = DATA_DIR / "raw_articles"   # your input JSON files go here
GRAPH_DIR = DATA_DIR / "graph"
PICKLE_PATH = GRAPH_DIR / "f1_knowledge.pkl"
CHROMA_DIR = DATA_DIR / "chroma"
BM25_PATH = DATA_DIR / "bm25_index.pkl"
CACHE_PATH = DATA_DIR / "answer_cache.json"
COLLECTION = "f1_knowledge"

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
SPACY_MODEL = "en_core_web_sm"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --------------------------------------------------------------------------
# Retrieval / fusion
# --------------------------------------------------------------------------
VECTOR_CANDIDATES = 10
BM25_CANDIDATES = 10
GRAPH_CANDIDATES = 6
RRF_K = 60
RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 5

# --------------------------------------------------------------------------
# CRAG confidence thresholds
# --------------------------------------------------------------------------
CRAG_UPPER_THRESHOLD = 0.60   # best passage score >= this  -> CORRECT
CRAG_LOWER_THRESHOLD = 0.30   # best passage score <  this  -> INCORRECT
                               # anything in between          -> AMBIGUOUS
STRIP_KEEP_THRESHOLD = 0.35   # a sentence "strip" must clear this to survive
MAX_STRIPS_PER_DOC = 6        # cap on how many sentences we keep per passage

# --------------------------------------------------------------------------
# Semantic chunking (used by ingest.py)
# --------------------------------------------------------------------------
CHUNK_SIMILARITY_THRESHOLD = 0.55   # below this, adjacent sentences start a new chunk
CHUNK_MAX_CHARS = 1200              # hard cap per chunk
CHUNK_MIN_CHARS = 200               # don't break a chunk before this many chars

# --------------------------------------------------------------------------
# .txt ingestion (used by wiki_txt_parser.py)
# --------------------------------------------------------------------------
# Section headings we drop entirely — pure citation/link dumps with no prose
# value, and they add noise to retrieval.
SKIP_SECTION_TITLES = {
    "references", "external links", "see also", "notes",
    "bibliography", "further reading",
}

# --------------------------------------------------------------------------
# Entity map (surface form -> canonical entity id used in the knowledge graph)
# --------------------------------------------------------------------------
ENTITY_MAP = {
    "senna": "ayrton_senna", "ayrton senna": "ayrton_senna",
    "prost": "alain_prost", "alain prost": "alain_prost",
    "schumacher": "michael_schumacher",
    "hamilton": "lewis_hamilton", "lewis hamilton": "lewis_hamilton",
    "verstappen": "max_verstappen", "max verstappen": "max_verstappen",
    "vettel": "sebastian_vettel",
    "alonso": "fernando_alonso",
    "räikkönen": "kimi_raikkonen", "raikkonen": "kimi_raikkonen",
    "häkkinen": "mika_hakkinen", "hakkinen": "mika_hakkinen",
    "mansell": "nigel_mansell",
    "piquet": "nelson_piquet",
    "lauda": "niki_lauda",
    "hunt": "james_hunt",
    "hill": "damon_hill",
    "button": "jenson_button",
    "norris": "lando_norris",
    "leclerc": "charles_leclerc",
    "russell": "george_russell",
    "piastri": "oscar_piastri",
    "webber": "mark_webber",
    "coulthard": "david_coulthard",
    "barrichello": "rubens_barrichello",
    "mclaren": "mclaren",
    "ferrari": "ferrari",
    "williams": "williams",
    "red bull": "red_bull",
    "mercedes": "mercedes_amg",
    "renault": "renault_f1",
    "benetton": "benetton",
    "brawn": "brawn_gp", "brawn gp": "brawn_gp",
    "aston martin": "aston_martin",
}

NER_TYPES = {"PERSON", "ORG"}
