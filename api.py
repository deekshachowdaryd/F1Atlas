"""
api.py

fastapi app to expose the crag chain.
loads model and data on startup and handles queries via POST /query.
run with: python api.py (or uvicorn api:app --reload)
"""
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import API_AUTH_TOKEN, log
from crag_chain import F1CRAGChain

app = FastAPI(title="F1 Historian CRAG API")

# allow all origins for local static html frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# loaded on startup
chain: Optional[F1CRAGChain] = None


@app.on_event("startup")
def load_chain():
    global chain
    log.info("Loading F1CRAGChain at startup (this takes a few seconds)...")
    chain = F1CRAGChain()
    log.info("F1CRAGChain loaded and ready.")


class Message(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    history: List[Message] = []


def _check_auth(authorization: Optional[str]):
    """Mirrors the frontend's optional 'API Key' field, sent as a Bearer token."""
    if not API_AUTH_TOKEN:
        return  # no auth check needed
    expected = f"Bearer {API_AUTH_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health():
    return {"status": "ok", "chain_loaded": chain is not None}


@app.post("/query")
def query(req: QueryRequest, authorization: Optional[str] = Header(default=None)):
    _check_auth(authorization)

    if chain is None:
        raise HTTPException(status_code=503, detail="Chain is still loading, try again shortly.")

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="'query' must not be empty")

    try:
        result = chain.run(req.query)
    except Exception as e:
        log.error(f"Error handling query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "answer": result.answer,
        "sources": result.sources,
        "sub_queries": result.sub_queries,
        "entities": result.entities,
        "retrieval_confidence": result.retrieval_confidence,
        "used_web_search": result.used_web_search,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
