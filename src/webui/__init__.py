"""webui — a zero-dependency console over runs/ (the RAG traceability deliverable).

Two layers:
  store.py   pure filesystem read-model over runs/ (no network, no Qdrant, no keys)
  server.py  a stdlib http.server that exposes store.py as JSON + drives the pipeline

Everything the frontend renders is already on disk (rag/trace.py writes it); replaying a
trace needs nothing but the filesystem, so this runs on a fresh clone with no API keys.
"""
