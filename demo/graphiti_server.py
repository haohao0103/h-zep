"""
Graphiti-native demo server: Graphiti(graph_driver=HugeGraphDriver) behind Flask.

Unlike zep_server.py (which used the simplified ZepHugeGraphEngine), this server
runs the REAL Graphiti pipeline — entity resolution, bi-temporal invalidation,
hybrid retrieval all come from Graphiti core, only graph storage is HugeGraph.

API (Graphiti-compatible):
  POST /api/messages       -> Graphiti.add_episode (native)
  POST /api/search         -> Graphiti.search (native)
  GET  /api/graph          -> current graph as Cytoscape elements
  GET  /api/stats          -> HugeGraph vertex/edge counts
  POST /api/reset          -> clear graph + rebuild schema
  GET  /api/locomo/sessions|load|qa  -> LOCOMO/Chinese dataset (reuse from zep_server)

Run:
  HF_HUB_OFFLINE=1 OPENAI_API_KEY=... LLM_API_KEY=... PYTHONPATH=.:graphiti-source \
  python demo/graphiti_server.py
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_file

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "graphiti-source"))

from graphiti_core import Graphiti
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.client import EmbedderClient
from zep_hg.hugegraph_driver import HugeGraphDriver
from zep_hg.embedder import LocalEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("graphiti_demo")

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))

DEEPSEEK_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""


class LocalEmbedderClient(EmbedderClient):
    def __init__(self, emb):
        self._emb = emb
    async def create(self, input_data):
        if isinstance(input_data, list):
            input_data = " ".join(str(x) for x in input_data)
        return self._emb.embed(input_data).tolist()
    async def create_batch(self, input_data_list):
        return [self._emb.embed(t if isinstance(t, str) else " ".join(t)).tolist() for t in input_data_list]


# --- init Graphiti native ---
_emb = LocalEmbedder()
_driver = HugeGraphDriver(embedder=_emb)
_driver._database = "hugegraph"
_llm = OpenAIGenericClient(config=LLMConfig(
    api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1",
    model="deepseek-chat"), structured_output_mode='json_object')
_embedder_client = LocalEmbedderClient(_emb)
_graphiti = Graphiti(graph_driver=_driver, llm_client=_llm, embedder=_embedder_client)
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def _run(coro):
    return _loop.run_until_complete(coro)


# --- LOCOMO datasets (reuse) ---
import json as _json
_LOCOMO_PATH = os.path.join(os.path.dirname(__file__), "..",
    "incubator-hugegraph-ai", "hugegraph-llm", "tests", "locomo_data", "locomo10.json")
_LOCOMO_ZH_PATH = os.path.join(os.path.dirname(__file__), "data", "locomo_zh.json")
DATASETS = {}
for _name, _path in (("en", _LOCOMO_PATH), ("zh", _LOCOMO_ZH_PATH)):
    try:
        with open(_path) as _f:
            DATASETS[_name] = _json.load(_f)
        log.info("dataset[%s] loaded: %d sessions", _name, len(DATASETS[_name]))
    except Exception as _e:
        DATASETS[_name] = []
        log.warning("dataset[%s] not loaded: %s", _name, _e)

# Graphiti built-in dataset: Wizard of Oz (chapter-segmented narrative)
_WIZARD_PATH = os.path.join(os.path.dirname(__file__), "..",
    "graphiti-source", "examples", "wizard_of_oz", "woo.txt")
try:
    import re as _re
    with open(_WIZARD_PATH, encoding="utf-8") as _f:
        _woo = _f.read()
    _chapters = _re.split(r"\n\n+Chapter [IVX]+\n", _woo)[1:]
    _wizard_conv = {"speaker_a": "Narrator", "speaker_b": "Characters"}
    for _i, _ch in enumerate(_chapters[:10], 1):
        _tm = _re.match(r"(.*?)\n\n", _ch)
        _title = _tm.group(1) if _tm else f"Chapter {_i}"
        _body = _ch[len(_title):].strip() if _tm else _ch.strip()
        _wizard_conv[f"session_{_i}_date_time"] = f"1900-{_i:02d}-01"
        _wizard_conv[f"session_{_i}"] = [{"speaker": "Narrator", "text": _body[:1500]}]
    DATASETS["wizard"] = [{
        "sample_id": "wizard-oz", "conversation": _wizard_conv,
        "session_summary": {"session_1_summary": "Dorothy's journey from Kansas to Oz"},
        "qa": [{"question": "Where does Dorothy live?", "answer": "Kansas", "category": "single-hop"},
               {"question": "Who are Dorothy's companions?", "answer": "Scarecrow, Tin Woodman, Cowardly Lion", "category": "single-hop"}],
    }]
    log.info("dataset[wizard] loaded: Wizard of Oz, %d chapters", len(_chapters))
except Exception as _e:
    DATASETS["wizard"] = []
    log.warning("dataset[wizard] not loaded: %s", _e)


def _parse_dt(s):
    from datetime import datetime as _dt
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y", "%Y-%m-%d", "%Y-%m-%d %H:%M"):
        try:
            return _dt.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def _dataset(req):
    ds = "en"
    if isinstance(req, dict):
        ds = req.get("datasource", "en")
    elif hasattr(req, "args"):
        ds = req.args.get("datasource", "en")
    return DATASETS.get(ds, DATASETS.get("en", []))


@app.route("/")
def index():
    return send_file(os.path.join(app.static_folder, "zep_frontend.html"))


@app.route("/api/reset", methods=["POST"])
def reset():
    # 1. wipe all data via HugeGraph /clear API (truncate backend)
    _driver.hg.clear_graph()
    # 2. rebuild schema (labels + properties + edge labels)
    _driver.hg.init_schema(rebuild=False)
    # 3. clear in-process embedding cache
    _driver.embedder._cache.clear()
    return jsonify({"ok": True, "msg": "graph reset (data cleared + schema rebuilt)"})


@app.route("/api/messages", methods=["POST"])
def add_episode():
    """Graphiti native add_episode."""
    d = request.json or {}
    body = d.get("body", "").strip() or d.get("episode_body", "").strip()
    if not body:
        return jsonify({"error": "empty body"}), 400
    date = d.get("date") or datetime.now(timezone.utc).date().isoformat()
    ref = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    try:
        r = _run(_graphiti.add_episode(
            name=d.get("name", "episode"),
            episode_body=body,
            source_description=d.get("source_description", "demo"),
            reference_time=ref,
            group_id="hugegraph",
        ))
        invalidated = sum(1 for e in r.edges if getattr(e, "invalid_at", None))
        return jsonify({
            "entities": len(r.nodes), "edges": len(r.edges),
            "invalidated_edges": invalidated,
            "native": True,
        })
    except Exception as ex:
        log.exception("add_episode failed")
        raw = str(ex)
        if "402" in raw or "Insufficient Balance" in raw:
            return jsonify({"error": "LLM 余额不足（DeepSeek 402），请充值后再试。"}), 503
        return jsonify({"error": f"摄入失败: {raw}"}), 500


@app.route("/api/add_episode", methods=["POST"])
def add_episode_alias():
    """Alias for frontend compatibility."""
    return add_episode()


@app.route("/api/search", methods=["POST"])
def search():
    """Graphiti native search."""
    d = request.json or {}
    q = d.get("query", "").strip()
    if not q:
        return jsonify({"error": "empty query"}), 400
    try:
        res = _run(_graphiti.search(q, group_ids=["hugegraph"], num_results=8))
        results = []
        for edge in res:
            results.append({
                "edge_uuid": getattr(edge, "uuid", ""),
                "score": 0.0,
                "source": "", "target": "",
                "fact": getattr(edge, "fact", ""),
                "valid_at": str(getattr(edge, "valid_at", "")) if getattr(edge, "valid_at", None) else None,
                "invalid_at": str(getattr(edge, "invalid_at", "")) if getattr(edge, "invalid_at", None) else None,
            })
        return jsonify({"query": q, "results": results, "native": True,
                        "stats": {"edges_returned": len(results)}})
    except Exception as ex:
        log.exception("search failed")
        return jsonify({"error": f"检索失败: {ex}"}), 500


@app.route("/api/graph")
def graph():
    """Return current graph as Cytoscape elements (from HugeGraph)."""
    nodes, edges = [], []
    seen = set()
    for v in _driver.hg.get_vertices_by_label("Entity", limit=10000):
        p = v.get("properties", {})
        nodes.append({"data": {"id": p.get("uuid", v.get("id")), "label": p.get("name", "?"),
                               "type": "entity", "summary": p.get("summary", "")}})
        seen.add(p.get("uuid"))
    for v in _driver.hg.get_vertices_by_label("Episodic", limit=10000):
        p = v.get("properties", {})
        nodes.append({"data": {"id": p.get("uuid", v.get("id")), "label": "episode",
                               "type": "episode"}})
        seen.add(p.get("uuid"))
    for v in _driver.hg.get_vertices_by_label("Entity", limit=10000):
        for e in _driver.hg.get_edges_of(v.get("id"), direction="OUT"):
            if e.get("label") != "RELATES_TO":
                continue
            p = e.get("properties", {})
            invalid = p.get("invalid_at")
            edges.append({"data": {
                "id": "e_" + (p.get("uuid", "") or "")[:8],
                "source": e.get("outV"), "target": e.get("inV"),
                "label": (p.get("fact") or "")[:30], "fact": p.get("fact", ""),
                "valid_at": p.get("valid_at"), "invalid_at": invalid,
                "expired": bool(invalid),
            }})
    return jsonify({"nodes": nodes, "edges": edges,
                    "stats": {"entities": len([n for n in nodes if n["data"]["type"] == "entity"]),
                              "edges": len(edges)}})


@app.route("/api/stats")
def stats():
    return jsonify({"entities": _driver.hg.count("Entity"),
                    "episodes": _driver.hg.count("Episodic"),
                    "cached_edges": len([e for v in _driver.hg.get_vertices_by_label("Entity", limit=10000)
                                         for e in _driver.hg.get_edges_of(v.get("id"), "OUT")
                                         if e.get("label") == "RELATES_TO"]),
                    "native": True})


@app.route("/api/locomo/sessions")
def locomo_sessions():
    data = _dataset(request)
    out = []
    for i, s in enumerate(data):
        c = s.get("conversation", {})
        n_sub = sum(1 for k in c if k.startswith("session_") and not k.endswith("_date_time"))
        ss = s.get("session_summary", {})
        summary = ss.get("session_1_summary", "") if isinstance(ss, dict) else str(ss)
        out.append({"idx": i, "sample_id": s.get("sample_id", f"conv-{i}"),
                    "speaker_a": c.get("speaker_a", "?"), "speaker_b": c.get("speaker_b", "?"),
                    "n_qa": len(s.get("qa", [])), "n_sessions": n_sub,
                    "summary": (summary or "")[:120]})
    return jsonify(out)


@app.route("/api/locomo/load", methods=["POST"])
def locomo_load():
    d = request.json or {}
    idx = int(d.get("idx", 0)); n = int(d.get("n_sessions", 3))
    data = _dataset(d)
    if idx >= len(data):
        return jsonify({"error": "bad idx"}), 400
    s = data[idx]; c = s["conversation"]; results = []; loaded = 0
    try:
        for k in sorted([k for k in c if k.startswith("session_") and not k.endswith("_date_time")],
                        key=lambda x: int(x.split("_")[1])):
            if loaded >= n:
                break
            dt_str = c.get(k + "_date_time", ""); ref = _parse_dt(dt_str)
            turns = c[k]
            body = " ".join(f"{t.get('speaker','?')}: {t.get('text','')}" for t in turns if t.get("text"))
            if not body.strip():
                continue
            r = _run(_graphiti.add_episode(
                name=f"locomo_{s['sample_id']}_{k}", episode_body=body[:2000],
                source_description=f"LOCOMO {s['sample_id']} {k} ({dt_str})",
                reference_time=ref, group_id="hugegraph"))
            invalidated = sum(1 for e in r.edges if getattr(e, "invalid_at", None))
            results.append({"session": k, "date": dt_str, "turns": len(turns),
                            "entities": len(r.nodes), "edges": len(r.edges),
                            "invalidated": invalidated})
            loaded += 1
        return jsonify({"sample_id": s["sample_id"], "loaded": loaded, "episodes": results, "native": True})
    except Exception as ex:
        log.exception("locomo_load failed")
        raw = str(ex)
        if "402" in raw or "Insufficient Balance" in raw:
            return jsonify({"error": "LLM 余额不足（DeepSeek 402），请充值后再试。"}), 503
        return jsonify({"error": f"摄入失败: {raw}", "loaded": loaded, "episodes": results}), 500


@app.route("/api/locomo/qa")
def locomo_qa():
    idx = int(request.args.get("idx", 0))
    data = _dataset(request)
    if idx >= len(data):
        return jsonify({"error": "bad idx"}), 400
    s = data[idx]
    return jsonify({"sample_id": s["sample_id"],
                   "qa": [{"q": x.get("question", ""), "a": x.get("answer", ""),
                           "category": x.get("category", "")} for x in s.get("qa", [])]})


@app.route("/api/temporal", methods=["POST"])
def temporal():
    """Graphiti native search (temporal filtering happens in Graphiti search recipe)."""
    return search()


if __name__ == "__main__":
    log.info("building indices...")
    _run(_driver.build_indices_and_constraints(delete_existing=True))
    log.info("Graphiti-native demo: http://127.0.0.1:8768")
    app.run(host="127.0.0.1", port=8768, debug=False, threaded=True)
