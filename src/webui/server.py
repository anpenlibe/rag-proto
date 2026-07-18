"""A zero-dependency console over ``runs/`` — the traceability + citations deliverable.

    PYTHONPATH=src .venv/bin/python -m webui.server [--port 8000] [--host 127.0.0.1]

Reads every hop of a logged query straight off the filesystem (no Qdrant, no keys, works
on a fresh clone), and can drive the live pipeline for a custom question or run the eval
set. Built on ``http.server`` — nothing to install beyond what the pipeline already needs.

Design notes:
  * **Replay is free.** All ``GET /api/*`` endpoints read ``rag/trace.py``'s output via
    ``webui.store``. The pipeline (Qdrant + fastembed + Groq) is imported *lazily*, only
    when a live query arrives — so the console starts, and replays traces, with none of
    that present.
  * **Live work is serialised.** Ad-hoc queries share one lazily-built ``RagPipeline``
    under a lock (fastembed/Qdrant clients aren't guaranteed thread-safe, and the key
    budget is shared). Interactive one-at-a-time use makes that free.
  * **Eval runs in a subprocess.** ``run eval`` shells out to ``rag.eval.harness`` so a
    long pass is isolated from the server and inherits the same ``QDRANT_URL``. Default
    is ``--retrieve-only``: the 240-query headline panel for **0 tokens**.
  * **Bind to localhost.** It exposes a live LLM/Qdrant trigger; it is not for the LAN.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from rag.config import EVAL_DIR, ROOT, Config
from webui import store

_GOLD_SET_RE = re.compile(r"^[A-Za-z0-9_]+$")

STATIC_DIR = Path(__file__).resolve().parent / "static"
_CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript",
                  ".css": "text/css", ".ico": "image/x-icon"}
MAX_QUERY_CHARS = 2000

# -- lazily-built live pipeline (import Qdrant/fastembed/Groq only on first live query) -
_pipeline = None
_pipeline_lock = threading.Lock()      # serialises live queries; also guards the build


def _get_pipeline():
    """Build the RagPipeline once, on first use. Caller must hold ``_pipeline_lock``."""
    global _pipeline
    if _pipeline is None:
        from rag.pipeline import RagPipeline    # heavy import, deferred to first query
        _pipeline = RagPipeline()
    return _pipeline


# -- eval jobs (in-memory registry; one pass at a time) -------------------------------
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _run_eval_job(job_id: str, argv: list[str]) -> None:
    """Run ``rag.eval.harness`` as a subprocess; record the run(s) it created."""
    import subprocess
    import sys

    before = {r["run_id"] for r in store.list_runs()}
    env = {**os.environ, "PYTHONPATH": "src"}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "rag.eval.harness", *argv],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800,
        )
        new_runs = sorted({r["run_id"] for r in store.list_runs()} - before)
        result = {
            "returncode": proc.returncode,
            "new_run_ids": new_runs,
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        }
        status = "done" if proc.returncode == 0 else "error"
    except Exception as e:               # timeout, spawn failure, …
        result = {"error": f"{type(e).__name__}: {e}"}
        status = "error"
    with _jobs_lock:
        _jobs[job_id].update(status=status, result=result, finished_at=time.time())


def _qdrant_status() -> dict:
    """Which vector-store backend a live query would use, and whether it's reachable."""
    url = os.environ.get("QDRANT_URL", "")
    if not url:
        return {"mode": "local", "url": None, "reachable": None,
                "note": "local path — single process at a time (file lock)"}
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/collections", timeout=1.5) as r:
            data = json.load(r)
        names = [c["name"] for c in data.get("result", {}).get("collections", [])]
        return {"mode": "server", "url": url, "reachable": True, "collections": names}
    except Exception as e:
        return {"mode": "server", "url": url, "reachable": False,
                "note": f"unreachable: {type(e).__name__}"}


def _has_keys() -> bool:
    """A live generation needs Groq keys — reported so the UI can disable the box."""
    return bool(os.environ.get("GROQ_API_KEYS") or os.environ.get("k1")
                or (ROOT / ".env").exists())


class Handler(BaseHTTPRequestHandler):
    server_version = "ragwebui/1.0"

    # -- response helpers ----------------------------------------------------------
    def _send(self, body: bytes, status: int = 200, ctype: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, status: int = 200):
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"), status)

    def _err(self, message: str, status: int = 400, **extra):
        self._json({"error": message, **extra}, status)

    def log_message(self, fmt, *args):        # quieter, single-line access log
        print(f"  {self.address_string()} {fmt % args}")

    # -- routing -------------------------------------------------------------------
    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path.startswith("/api/"):
                return self._get_api(path)
            return self._get_static(path)
        except store.NotFound as e:
            self._err(str(e), 404)
        except BrokenPipeError:
            pass
        except Exception as e:               # never leak a traceback to the browser
            self._err(f"{type(e).__name__}: {e}", 500)

    do_HEAD = do_GET

    def do_DELETE(self):
        path = urlsplit(self.path).path
        parts = path.strip("/").split("/")          # ["api", "runs", <run_id>]
        try:
            if len(parts) == 3 and parts[:2] == ["api", "runs"]:
                return self._json(store.delete_run(parts[2]))
            self._err("unknown endpoint", 404)
        except store.NotFound as e:
            self._err(str(e), 404)
        except store.NotDeletable as e:
            self._err(str(e), 403)
        except BrokenPipeError:
            pass
        except Exception as e:                       # never leak a traceback to the browser
            self._err(f"{type(e).__name__}: {e}", 500)

    def _get_static(self, path: str):
        if path in ("/", "/index.html"):
            rel = "index.html"
        elif path == "/favicon.ico":
            return self._send(b"", 204, "image/x-icon")
        else:
            rel = path.lstrip("/")
        target = (STATIC_DIR / rel).resolve()
        if target.parent != STATIC_DIR.resolve() or not target.is_file():
            return self._err("not found", 404)
        ctype = _CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self._send(target.read_bytes(), 200, ctype)

    def _get_api(self, path: str):
        parts = path.strip("/").split("/")     # ["api", ...]
        if parts == ["api", "status"]:
            return self._json(self._status())
        if parts == ["api", "runs"]:
            return self._json({"runs": store.list_runs()})
        if parts == ["api", "gold-sets"]:
            return self._json({"gold_sets": store.list_gold_sets()})
        if len(parts) == 3 and parts[1] == "runs":
            return self._json(store.load_run(parts[2]))
        if len(parts) == 4 and parts[1] == "runs" and parts[3] == "eval":
            ev = store.load_eval(parts[2])
            return self._json(ev) if ev else self._err("run not scored", 404)
        if len(parts) == 5 and parts[1] == "runs" and parts[3] == "trace":
            return self._json(store.load_trace(parts[2], parts[4]))
        if len(parts) == 4 and parts[1:3] == ["eval", "jobs"]:
            with _jobs_lock:
                job = _jobs.get(parts[3])
            return self._json(job) if job else self._err("no such job", 404)
        self._err("unknown endpoint", 404)

    def _status(self) -> dict:
        cfg = Config().to_dict()
        cfg.pop("qdrant_url", None)             # shown separately via _qdrant_status
        return {
            "qdrant": _qdrant_status(),
            "has_keys": _has_keys(),
            "config": cfg,
            "budget": store.token_spend_today(),
            "pipeline_ready": _pipeline is not None,
        }

    def do_POST(self):
        path = urlsplit(self.path).path
        try:
            body = self._read_json_body()
            if path == "/api/query":
                return self._post_query(body)
            if path == "/api/eval":
                return self._post_eval(body)
            self._err("unknown endpoint", 404)
        except ValueError as e:                # bad JSON / bad input
            self._err(str(e), 400)
        except Exception as e:
            self._err(f"{type(e).__name__}: {e}", 500)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError("request body is not valid JSON")

    # -- live pipeline (a custom ad-hoc question) ----------------------------------
    def _post_query(self, body: dict):
        query = (body.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query too long (max {MAX_QUERY_CHARS} chars)")
        if not _has_keys():
            return self._err("no Groq keys configured (set GROQ_API_KEYS or .env)", 503)

        before = {r["run_id"] for r in store.list_runs()}
        with _pipeline_lock:                    # one live query at a time
            pipe = _get_pipeline()
            try:
                _ans, trace_path = pipe.answer(query)
                run_id, trace_id = trace_path.parents[1].name, trace_path.stem
                return self._json({"run_id": run_id, "trace_id": trace_id,
                                   "trace": store.load_trace(run_id, trace_id)})
            except Exception as e:
                # A failed generation still wrote a full error trace (retrieved/selected
                # survive) — surface it so the failure itself is inspectable.
                new = sorted({r["run_id"] for r in store.list_runs()} - before)
                trace = None
                if new:
                    run = store.load_run(new[-1])
                    if run["queries"]:
                        trace = store.load_trace(new[-1], run["queries"][-1]["trace_id"])
                self._json({"error": f"{type(e).__name__}: {e}",
                            "run_id": new[-1] if new else None, "trace": trace}, 502)

    # -- run the eval set ----------------------------------------------------------
    def _post_eval(self, body: dict):
        mode = body.get("mode", "retrieve_only")
        gold_args, gold_tag = [], ""
        gold_set = body.get("gold_set")
        if gold_set:                                 # optional: score a specific gold set
            if not _GOLD_SET_RE.match(gold_set) or not (EVAL_DIR / f"{gold_set}.jsonl").exists():
                raise ValueError(f"unknown gold set: {gold_set!r}")
            gold_args, gold_tag = ["--gold-set", gold_set], f" [{gold_set}]"
        if mode == "retrieve_only":
            argv = ["--retrieve-only", *gold_args,
                    "--label", body.get("label") or f"webui retrieval panel{gold_tag}"]
        elif mode == "smoke":
            limit = int(body.get("limit") or 3)
            if not 1 <= limit <= 40:
                raise ValueError("limit must be 1..40")
            if not _has_keys():
                return self._err("smoke eval generates answers but no Groq keys are set", 503)
            argv = ["--limit", str(limit), *gold_args,
                    "--label", body.get("label") or f"webui smoke x{limit}{gold_tag}"]
        else:
            raise ValueError(f"unknown eval mode: {mode!r} (use retrieve_only | smoke)")

        with _jobs_lock:
            running = [j for j in _jobs.values() if j["status"] == "running"]
            if running:
                return self._err("an eval pass is already running", 409,
                                 job_id=running[0]["job_id"])
            job_id = uuid.uuid4().hex[:8]
            _jobs[job_id] = {"job_id": job_id, "status": "running", "mode": mode,
                             "argv": argv, "started_at": time.time(), "result": None}
        threading.Thread(target=_run_eval_job, args=(job_id, argv), daemon=True).start()
        self._json({"job_id": job_id, "status": "running", "mode": mode}, 202)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="webui.server", description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default localhost)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    q = _qdrant_status()
    backend = q["url"] if q["mode"] == "server" else "local path (single-process)"
    print("rag-proto console")
    print(f"  http://{args.host}:{args.port}")
    print(f"  qdrant : {backend}"
          + ("" if q.get("reachable") is not False else "  ⚠ UNREACHABLE"))
    print(f"  keys   : {'present' if _has_keys() else 'MISSING (replay only)'}")
    print("  ctrl-c to stop", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        httpd.shutdown()


if __name__ == "__main__":
    main()
