"""
Zep/Graphiti-style temporal knowledge-graph engine on HugeGraph.

This is a faithful, dependency-light re-implementation of the Graphiti
interaction flow (add_episode → extract → resolve → store with temporal
invalidation; search → hybrid vector+fulltext+graph retrieval) adapted onto
HugeGraph 1.7.0 via the REST client in driver.py.

Why a re-implementation rather than Graphiti's Neo4jDriver subclass:
Graphiti's Operations layer and node/edge models are Cypher-string based
(every save/get/delete is a Cypher template in node_db_queries.py), and
HugeGraph 1.7.0 has no Cypher engine. The temporal-invalidation logic and
hybrid-retrieval recipe, however, are pure-Python in Graphiti and are mirrored
here. See docs/ZEP_HUGEGRAPH_ADAPTER.md § "Adaptation approach".
"""

from __future__ import annotations

import json
import logging
import re
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .driver import HugeGraphClient, HugeGraphError
from .embedder import LocalEmbedder

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_json_loose(text: str) -> Any:
    """Extract the first JSON object/array from an LLM response."""
    text = text.strip()
    # try direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # strip code fences
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # find first { ... } or [ ... ]
    for pat in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pat, text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    raise ValueError(f"cannot parse JSON from LLM: {text[:200]}")


class ZepHugeGraphEngine:
    """Temporal knowledge-graph engine: add_episode + hybrid search."""

    def __init__(self, hg: HugeGraphClient, embedder: LocalEmbedder,
                 llm_api_key: str | None = None, llm_base_url: str | None = None,
                 llm_model: str = "deepseek-chat"):
        self.hg = hg
        self.embedder = embedder
        # DeepSeek via OpenAI-compatible SDK
        from openai import OpenAI
        self.llm = OpenAI(
            api_key=llm_api_key or os.environ.get("LLM_API_KEY",
                                                  "sk-624b175e39f543429cd402555021e7f9"),
            base_url=llm_base_url or os.environ.get("LLM_BASE_URL",
                                                     "https://api.deepseek.com/v1"),
        )
        self.llm_model = os.environ.get("LLM_MODEL", llm_model)
        # in-process edge cache (uuid -> edge dict) for vector search
        self._edges: dict[str, dict] = {}
        self._entities: dict[str, dict] = {}

    # ----- LLM ------------------------------------------------------------- #
    def _llm(self, system: str, user: str, max_tokens: int = 2048) -> str:
        r = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.1, max_tokens=max_tokens,
        )
        return r.choices[0].message.content or ""

    # ----- add_episode ----------------------------------------------------- #
    def add_episode(self, name: str, body: str, source_description: str,
                    reference_time: datetime, group_id: str = "default",
                    source: str = "message") -> dict:
        """Graphiti-style: extract entities & edges, store, invalidate stale."""
        now = utc_now()
        ref_iso = reference_time.isoformat()
        episode_uuid = uuid4().hex

        # 1. LLM extract entities
        ent_sys = ("You are a knowledge-graph extractor. Extract entities "
                   "(people, places, orgs, concepts) from the episode. "
                   "Return ONLY JSON: {\"entities\":[{\"name\":\"\",\"summary\":\"\"}]}")
        raw = self._llm(ent_sys, f"Episode:\n{body}")
        ents = _parse_json_loose(raw).get("entities", [])
        ents = [e for e in ents if e.get("name")]
        logger.info("extracted %d entities", len(ents))

        # 2. LLM extract edges
        ent_names = ", ".join(e["name"] for e in ents)
        edge_sys = ("You are a knowledge-graph extractor. Extract factual "
                    "relationships between the known entities from the episode. "
                    "Each edge: source, target (entity names), fact (one sentence), "
                    "valid_at (ISO datetime). Return ONLY JSON: "
                    "{\"edges\":[{\"source\":\"\",\"target\":\"\",\"fact\":\"\","
                    "\"valid_at\":\"\"}]}")
        raw = self._llm(edge_sys,
                        f"Episode:\n{body}\n\nKnown entities: {ent_names}\n"
                        f"Reference time: {ref_iso}")
        edges = _parse_json_loose(raw).get("edges", [])
        edges = [e for e in edges if e.get("source") and e.get("target")]
        logger.info("extracted %d edges", len(edges))

        # 3. store Episodic vertex
        self.hg.upsert_vertex("Episodic", episode_uuid, {
            "uuid": episode_uuid, "name": name, "group_id": group_id,
            "source": source, "source_description": source_description,
            "content": body, "valid_at": ref_iso, "created_time": now,
            "entity_edges": [e.get("uuid", "") for e in edges],
        })

        # 4. store Entity vertices + MENTIONS edges
        name_to_uuid: dict[str, str] = {}
        for e in ents:
            euuid = uuid4().hex
            name_to_uuid[e["name"]] = euuid
            summary = e.get("summary", "")
            self.hg.upsert_vertex("Entity", euuid, {
                "uuid": euuid, "name": e["name"], "group_id": group_id,
                "summary": summary, "labels": ["Entity"],
                "attributes": {}, "created_time": now,
            })
            # embed + cache
            self.embedder.embed(e["name"], key=euuid)
            self._entities[euuid] = {"uuid": euuid, "name": e["name"],
                                     "summary": summary, "group_id": group_id}
            # MENTIONS: Episodic -> Entity
            self.hg.upsert_edge("MENTIONS", episode_uuid, euuid,
                                "Episodic", "Entity",
                                {"uuid": uuid4().hex, "group_id": group_id,
                                 "created_time": now})

        # 5. store RELATES_TO edges with temporal invalidation
        new_edges = []
        for ed in edges:
            su = name_to_uuid.get(ed["source"])
            tu = name_to_uuid.get(ed["target"])
            if not su or not tu:
                continue
            euuid = uuid4().hex
            fact = ed["fact"]
            valid_at = ed.get("valid_at") or ref_iso
            # temporal invalidation: check existing edges su->tu
            self._invalidate_conflicts(su, tu, valid_at, now)
            props = {
                "uuid": euuid, "group_id": group_id,
                "fact": fact, "valid_at": valid_at,
                "invalid_at": None, "expired_at": None,
                "reference_time": ref_iso, "episodes": [episode_uuid],
                "created_time": now, "attributes": {},
                "source_node_uuid": su, "target_node_uuid": tu,
            }
            self.hg.upsert_edge("RELATES_TO", su, tu, "Entity", "Entity", props)
            self.embedder.embed(fact, key=euuid)
            self._edges[euuid] = props
            new_edges.append(props)

        return {"episode_uuid": episode_uuid, "entities": len(ents),
                "edges": len(new_edges), "edges_detail": new_edges}

    def _invalidate_conflicts(self, src_uuid: str, tgt_uuid: str,
                              new_valid_at: str, now: str) -> None:
        """If an existing su->tu edge exists, mark it invalid (temporal)."""
        existing = self.hg.get_edges_of(src_uuid, direction="OUT")
        for e in existing:
            if e.get("label") != "RELATES_TO":
                continue
            if e.get("inV") != tgt_uuid:
                continue
            old_invalid = e.get("properties", {}).get("invalid_at")
            if old_invalid and old_invalid not in ("", "null", "None"):
                continue  # already invalidated
            euuid = e.get("properties", {}).get("uuid")
            eid = e.get("id")
            if not eid or not euuid:
                continue
            try:
                self.hg.update_edge(eid, "RELATES_TO",
                                    {"invalid_at": new_valid_at, "expired_at": now})
            except Exception as ex:
                logger.warning("edge update failed: %s", ex)
            if euuid in self._edges:
                self._edges[euuid]["invalid_at"] = new_valid_at
                self._edges[euuid]["expired_at"] = now
            logger.info("invalidated edge %s (su=%s tu=%s) @ %s",
                        euuid, src_uuid, tgt_uuid, new_valid_at)

    # ----- search ---------------------------------------------------------- #
    def search(self, query: str, group_id: str = "default",
               top_k: int = 10, num_results: int = 5) -> dict:
        """Hybrid: vector (cosine) + fulltext (token overlap) + graph (traverse)."""
        qvec = self.embedder.embed(query)

        # --- vector channel ---
        ent_scores: dict[str, float] = {}
        for euuid, ent in self._entities.items():
            if ent.get("group_id") != group_id:
                continue
            v = self.embedder._cache.get(euuid)
            if v is not None:
                ent_scores[euuid] = self.embedder.cosine(qvec, v)

        edge_scores: dict[str, float] = {}
        for euuid, ed in self._edges.items():
            if ed.get("group_id") != group_id:
                continue
            v = self.embedder._cache.get(euuid)
            if v is not None:
                edge_scores[euuid] = self.embedder.cosine(qvec, v)

        # --- fulltext channel (token overlap) ---
        qtokens = set(re.findall(r"\w+", query.lower()))
        ent_ft: dict[str, float] = {}
        for euuid, ent in self._entities.items():
            if ent.get("group_id") != group_id:
                continue
            txt = (ent.get("name", "") + " " + ent.get("summary", "")).lower()
            ttoks = set(re.findall(r"\w+", txt))
            if qtokens and ttoks:
                ent_ft[euuid] = len(qtokens & ttoks) / len(qtokens | ttoks)

        edge_ft: dict[str, float] = {}
        for euuid, ed in self._edges.items():
            if ed.get("group_id") != group_id:
                continue
            txt = ed.get("fact", "").lower()
            ttoks = set(re.findall(r"\w+", txt))
            if qtokens and ttoks:
                edge_ft[euuid] = len(qtokens & ttoks) / len(qtokens | ttoks)

        # --- graph channel (traverse from top vector entities) ---
        graph_edge_boost: dict[str, float] = {}
        top_ents = sorted(ent_scores, key=lambda x: -ent_scores[x])[:top_k]
        for euuid in top_ents:
            try:
                nbr_edges = self.hg.get_edges_of(euuid, direction="OUT")
            except HugeGraphError:
                continue
            for e in nbr_edges:
                if e.get("label") != "RELATES_TO":
                    continue
                eu = e.get("properties", {}).get("uuid")
                if eu:
                    graph_edge_boost[eu] = max(graph_edge_boost.get(eu, 0), 0.3)

        # --- fusion (additive, Graphiti-style) ---
        all_edge_uuids = (set(edge_scores) | set(edge_ft) | set(graph_edge_boost))
        fused: list[tuple[str, float]] = []
        for euuid in all_edge_uuids:
            score = (0.4 * edge_scores.get(euuid, 0)
                     + 0.3 * edge_ft.get(euuid, 0)
                     + 0.3 * graph_edge_boost.get(euuid, 0))
            fused.append((euuid, score))
        fused.sort(key=lambda x: -x[1])

        results = []
        for euuid, score in fused[:num_results]:
            ed = self._edges.get(euuid, {})
            su = ed.get("source_node_uuid")
            tu = ed.get("target_node_uuid")
            sname = self._entities.get(su, {}).get("name", "?")
            tname = self._entities.get(tu, {}).get("name", "?")
            results.append({
                "edge_uuid": euuid, "score": round(score, 4),
                "source": sname, "target": tname,
                "fact": ed.get("fact", ""), "valid_at": ed.get("valid_at"),
                "invalid_at": ed.get("invalid_at"),
            })
        return {"query": query, "results": results,
                "stats": {"entities_scanned": len(ent_scores),
                          "edges_scanned": len(edge_scores),
                          "graph_boosted": len(graph_edge_boost)}}

    # --- temporal point-in-time query ---
    def temporal_query(self, query: str, point_in_time: datetime,
                       group_id: str = "default", num_results: int = 5) -> dict:
        pit = point_in_time.isoformat()
        res = self.search(query, group_id=group_id, num_results=num_results * 3)
        filtered = []
        for r in res["results"]:
            va = r.get("valid_at")
            ia = r.get("invalid_at")
            if va and va <= pit and (not ia or ia > pit):
                filtered.append(r)
        res["results"] = filtered[:num_results]
        res["point_in_time"] = pit
        return res
