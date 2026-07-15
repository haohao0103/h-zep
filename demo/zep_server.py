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
    gid = d.get("group_id", "demo")
    return jsonify(engine.search(q, group_id=gid, num_results=8))


@app.route("/api/temporal", methods=["POST"])
def temporal():
    d = request.json or {}
    q = d.get("query", "").strip()
    pit_str = d.get("datetime", "").strip()
    if not q or not pit_str:
        return jsonify({"error": "need query + datetime"}), 400
    pit = datetime.fromisoformat(pit_str).replace(tzinfo=timezone.utc)
    gid = d.get("group_id", "demo")
    return jsonify(engine.temporal_query(q, pit, group_id=gid, num_results=8))


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


# --- LOCOMO benchmark dataset (real agent-memory eval data) ---
import json as _json
from datetime import datetime as _dt

_LOCOMO_PATH = os.path.join(os.path.dirname(__file__), "..",
    "incubator-hugegraph-ai", "hugegraph-llm", "tests", "locomo_data", "locomo10.json")
try:
    with open(_LOCOMO_PATH) as _f:
        LOCOMO = _json.load(_f)
    log.info("LOCOMO loaded: %d sessions", len(LOCOMO))
except Exception as _e:
    LOCOMO = []
    log.warning("LOCOMO not loaded: %s", _e)


def _parse_locomo_dt(s: str) -> _dt:
    """Parse '1:56 pm on 8 May, 2023' → datetime."""
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return _dt.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return _dt.now(timezone.utc)


@app.route("/api/locomo/sessions")
def locomo_sessions():
    """List 10 LOCOMO sessions (real Zep eval data)."""
    out = []
    for i, s in enumerate(LOCOMO):
        c = s.get("conversation", {})
        n_sub = sum(1 for k in c if k.startswith("session_") and not k.endswith("_date_time"))
        ss = s.get("session_summary", {})
        summary = ss.get("session_1_summary", "") if isinstance(ss, dict) else str(ss)
        out.append({
            "idx": i, "sample_id": s.get("sample_id", f"conv-{i}"),
            "speaker_a": c.get("speaker_a", "?"), "speaker_b": c.get("speaker_b", "?"),
            "n_qa": len(s.get("qa", [])), "n_sessions": n_sub,
            "summary": (summary or "")[:120],
        })
    return jsonify(out)


@app.route("/api/locomo/load", methods=["POST"])
def locomo_load():
    """Load first N sub-sessions of a LOCOMO session as episodes into HugeGraph."""
    d = request.json or {}
    idx = int(d.get("idx", 0))
    n = int(d.get("n_sessions", 3))
    if idx >= len(LOCOMO):
        return jsonify({"error": "bad idx"}), 400
    s = LOCOMO[idx]
    c = s["conversation"]
    results = []
    loaded = 0
    try:
        for k in sorted([k for k in c if k.startswith("session_") and not k.endswith("_date_time")],
                        key=lambda x: int(x.split("_")[1])):
            if loaded >= n:
                break
            dt_key = k + "_date_time"
            dt_str = c.get(dt_key, "")
            ref = _parse_locomo_dt(dt_str)
            turns = c[k]
            body = " ".join(f"{t.get('speaker','?')}: {t.get('text','')}"
                            for t in turns if t.get("text"))
            if not body.strip():
                continue
            r = engine.add_episode(
                name=f"locomo_{s['sample_id']}_{k}",
                body=body[:2000],
                source_description=f"LOCOMO {s['sample_id']} {k} ({dt_str})",
                reference_time=ref,
                group_id="locomo",
                source="message",
            )
            results.append({"session": k, "date": dt_str, "turns": len(turns),
                            "entities": r.get("entities", 0), "edges": r.get("edges", 0)})
            loaded += 1
        return jsonify({"sample_id": s["sample_id"], "loaded": loaded, "episodes": results})
    except Exception as ex:
        log.exception("locomo_load failed")
        raw = str(ex)
        if "402" in raw or "Insufficient Balance" in raw or "insufficient_quota" in raw:
            return jsonify({"error": "LLM 余额不足（DeepSeek 402 Insufficient Balance），请充值 DeepSeek 账户后再试。",
                            "raw": raw, "loaded": loaded, "episodes": results}), 503
        return jsonify({"error": f"摄入失败: {raw}", "loaded": loaded, "episodes": results}), 500


@app.route("/api/locomo/qa")
def locomo_qa():
    """Return QA list for a LOCOMO session (for retrieval eval)."""
    idx = int(request.args.get("idx", 0))
    if idx >= len(LOCOMO):
        return jsonify({"error": "bad idx"}), 400
    s = LOCOMO[idx]
    return jsonify({"sample_id": s["sample_id"],
                   "qa": [{"q": x.get("question", ""), "a": x.get("answer", ""),
                           "category": x.get("category", "")} for x in s.get("qa", [])]})


if __name__ == "__main__":
    log.info("Zep-HugeGraph demo: http://127.0.0.1:8767")
    app.run(host="127.0.0.1", port=8767, debug=False, threaded=True)
