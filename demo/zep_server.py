"""
Interactive demo server for the Zep/Graphiti → HugeGraph adapter.

Wraps ZepHugeGraphEngine behind a small Flask REST API and serves the
single-page frontend (zep_frontend.html). Demonstrates the full Graphiti
interaction flow visually: add_episode → temporal graph → hybrid search →
point-in-time query.

Run:
  HF_HUB_OFFLINE=1 PYTHONPATH=. \
  /Users/mac/.workbuddy/binaries/python/envs/hg-llm/bin/python3.10 \
  demo/zep_server.py
"""

import logging
import os
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_file

from zep_hg import HugeGraphClient, LocalEmbedder, ZepHugeGraphEngine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("zep_demo")

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))

# --- init engine (real HugeGraph + DeepSeek LLM) ---
hg = HugeGraphClient()
embedder = LocalEmbedder()
engine = ZepHugeGraphEngine(hg, embedder)


@app.route("/")
def index():
    return send_file(os.path.join(app.static_folder, "zep_frontend.html"))


@app.route("/api/reset", methods=["POST"])
def reset():
    hg.init_schema()  # rebuild = clear all data + recreate schema
    engine._entities.clear()
    engine._edges.clear()
    return jsonify({"ok": True, "msg": "graph reset"})


@app.route("/api/add_episode", methods=["POST"])
def add_episode():
    d = request.json or {}
    body = d.get("body", "").strip()
    if not body:
        return jsonify({"error": "empty body"}), 400
    date = d.get("date") or datetime.now(timezone.utc).date().isoformat()
    ref = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    r = engine.add_episode(
        name=d.get("name", "episode"),
        body=body,
        source_description="demo conversation",
        reference_time=ref,
        group_id="demo",
        source="message",
    )
    return jsonify(r)


@app.route("/api/search", methods=["POST"])
def search():
    d = request.json or {}
    q = d.get("query", "").strip()
    if not q:
        return jsonify({"error": "empty query"}), 400
    return jsonify(engine.search(q, group_id="demo", num_results=8))


@app.route("/api/temporal", methods=["POST"])
def temporal():
    d = request.json or {}
    q = d.get("query", "").strip()
    pit_str = d.get("datetime", "").strip()
    if not q or not pit_str:
        return jsonify({"error": "need query + datetime"}), 400
    pit = datetime.fromisoformat(pit_str).replace(tzinfo=timezone.utc)
    return jsonify(engine.temporal_query(q, pit, group_id="demo", num_results=8))


@app.route("/api/graph")
def graph():
    """Return the current temporal graph as Cytoscape elements."""
    nodes, edges = [], []
    seen = set()
    for e in engine._entities.values():
        nodes.append({"data": {"id": e["uuid"], "label": e.get("name", "?"),
                               "type": "entity",
                               "summary": e.get("summary", "")}})
        seen.add(e["uuid"])
    for ed in engine._edges.values():
        su = ed.get("source_node_uuid")
        tu = ed.get("target_node_uuid")
        if su not in seen:
            nodes.append({"data": {"id": su, "label": su[:6], "type": "entity"}})
            seen.add(su)
        if tu not in seen:
            nodes.append({"data": {"id": tu, "label": tu[:6], "type": "entity"}})
            seen.add(tu)
        invalid = ed.get("invalid_at")
        edges.append({"data": {
            "id": "e_" + ed.get("uuid", "")[:8],
            "source": su, "target": tu,
            "label": (ed.get("fact") or "")[:30],
            "fact": ed.get("fact", ""),
            "valid_at": ed.get("valid_at"),
            "invalid_at": invalid,
            "expired": bool(invalid),
        }})
    return jsonify({"nodes": nodes, "edges": edges,
                    "stats": {"entities": len(engine._entities),
                              "edges": len(engine._edges)}})


@app.route("/api/stats")
def stats():
    return jsonify({"entities": hg.count("Entity"),
                    "episodes": hg.count("Episodic"),
                    "cached_edges": len(engine._edges)})


if __name__ == "__main__":
    log.info("Zep-HugeGraph demo: http://127.0.0.1:8767")
    app.run(host="127.0.0.1", port=8767, debug=False, threaded=True)
