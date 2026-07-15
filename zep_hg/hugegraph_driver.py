"""
HugeGraph driver for Graphiti — lets Graphiti run natively on HugeGraph 1.7.0.

Implements GraphDriver + GraphOperationsInterface + SearchInterface so that
`Graphiti(graph_driver=HugeGraphDriver(...))` reuses Graphiti's full pipeline:
entity resolution (resolve_extracted_nodes), bi-temporal invalidation
(edge_operations.py), and hybrid retrieval (search recipe).

Only the graph *storage* is swapped to HugeGraph REST; all Graphiti core
logic runs untouched. Vector search is in-process (HugeGraph 1.7 has no
vector index); fulltext + graph traversal hit HugeGraph.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.driver.query_executor import Transaction
from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.nodes import EntityNode, EpisodicNode

from .driver import HugeGraphClient, _json_str
from .embedder import LocalEmbedder

logger = logging.getLogger(__name__)


# ---------------- helpers ---------------- #
def _g(obj, k, d=None):
    """Get attr (object) or key (dict)."""
    if isinstance(obj, dict):
        return obj.get(k, d)
    return getattr(obj, k, d)


def _dt2s(dt) -> str | None:
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    return dt.isoformat()


def _s2dt(s):
    if not s or s in ("", "None", "null"):
        return None
    if isinstance(s, datetime):
        return s
    try:
        return datetime.fromisoformat(str(s))
    except Exception:
        return None


def _j2s(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _s2j(s):
    if not s or s in ("", "None", "null"):
        return None
    if isinstance(s, (list, dict)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return s


# ---------------- GraphOperations ---------------- #
class HugeGraphGraphOperations(GraphOperationsInterface):
    """HugeGraph REST implementation of GraphOperationsInterface (P0 subset)."""
    d: Any = None

    def __init__(self, driver: "HugeGraphDriver"):
        super().__init__()
        self.d = driver

    # -- node save -- #
    async def node_save_bulk(self, _cls, driver, tx, nodes, batch_size=100):
        for n in nodes:
            uuid = _g(n, "uuid")
            if not uuid:
                continue
            self.d.hg.upsert_vertex("Entity", uuid, {
                "uuid": uuid, "name": _g(n, "name", ""), "group_id": _g(n, "group_id", ""),
                "summary": _g(n, "summary", "") or "", "labels": _j2s(_g(n, "labels") or ["Entity"]),
                "attributes": _j2s(_g(n, "attributes") or {}), "created_time": _dt2s(_g(n, "created_at")),
            })
            # cache embedding if present
            emb = _g(n, "name_embedding")
            if emb is not None:
                self.d.embedder._cache[uuid] = self.d.embedder._to_arr(emb)

    async def episodic_node_save_bulk(self, _cls, driver, tx, nodes, batch_size=100):
        for n in nodes:
            uuid = _g(n, "uuid")
            if not uuid:
                continue
            self.d.hg.upsert_vertex("Episodic", uuid, {
                "uuid": uuid, "name": _g(n, "name", ""), "group_id": _g(n, "group_id", ""),
                "source": _g(n, "source", "") or "", "source_description": _g(n, "source_description", "") or "",
                "content": _g(n, "content", "") or "", "valid_at": _dt2s(_g(n, "valid_at")),
                "created_time": _dt2s(_g(n, "created_at")), "entity_edges": _j2s(_g(n, "entity_edges") or []),
            })

    async def episodic_edge_save_bulk(self, _cls, driver, tx, episodic_edges, batch_size=100):
        for e in episodic_edges:
            uuid = _g(e, "uuid")
            if not uuid:
                continue
            src = _g(e, "source_node_uuid"); tgt = _g(e, "target_node_uuid")
            if not src or not tgt:
                continue
            self.d.hg.upsert_edge("MENTIONS", src, tgt, "Episodic", "Entity", {
                "uuid": uuid, "group_id": _g(e, "group_id", ""),
                "created_time": _dt2s(_g(e, "created_at")),
            })

    async def edge_save_bulk(self, _cls, driver, tx, edges, batch_size=100):
        for e in edges:
            uuid = _g(e, "uuid")
            if not uuid:
                continue
            src = _g(e, "source_node_uuid"); tgt = _g(e, "target_node_uuid")
            if not src or not tgt:
                continue
            self.d.hg.upsert_edge("RELATES_TO", src, tgt, "Entity", "Entity", {
                "uuid": uuid, "group_id": _g(e, "group_id", ""), "name": _g(e, "name", "") or "",
                "fact": _g(e, "fact", "") or "", "valid_at": _dt2s(_g(e, "valid_at")),
                "invalid_at": _dt2s(_g(e, "invalid_at")), "expired_at": _dt2s(_g(e, "expired_at")),
                "reference_time": _dt2s(_g(e, "reference_time")), "episodes": _j2s(_g(e, "episodes") or []),
                "created_time": _dt2s(_g(e, "created_at")), "attributes": _j2s(_g(e, "attributes") or {}),
                "source_node_uuid": src, "target_node_uuid": tgt,
            })
            emb = _g(e, "fact_embedding")
            if emb is not None:
                self.d.embedder._cache[uuid] = self.d.embedder._to_arr(emb)

    async def edge_save(self, edge, driver):
        await self.edge_save_bulk(None, driver, None, [edge])

    async def node_save(self, node, driver):
        await self.node_save_bulk(None, driver, None, [node])

    async def episodic_node_save(self, node, driver):
        await self.episodic_node_save_bulk(None, driver, None, [node])

    # -- node read -- #
    async def node_get_by_uuid(self, _cls, driver, uuid):
        p = self.d.hg.get_vertex(uuid)
        return self._props_to_entity(p) if p else None

    async def node_get_by_uuids(self, _cls, driver, uuids, group_id=None):
        out = []
        for u in uuids:
            p = self.d.hg.get_vertex(u)
            if p:
                out.append(self._props_to_entity(p))
        return out

    async def episodic_node_get_by_uuid(self, _cls, driver, uuid):
        p = self.d.hg.get_vertex(uuid)
        return self._props_to_episodic(p) if p else None

    async def episodic_node_get_by_uuids(self, _cls, driver, uuids):
        out = []
        for u in uuids:
            p = self.d.hg.get_vertex(u)
            if p:
                out.append(self._props_to_episodic(p))
        return out

    async def retrieve_episodes(self, driver, reference_time, last_n=3,
                                group_ids=None, source=None, saga=None):
        eps = self.d.hg.get_vertices_by_label("Episodic", limit=10000)
        ref_s = _dt2s(reference_time)
        out = []
        for v in eps:
            p = v.get("properties", {})
            if group_ids and p.get("group_id") not in group_ids:
                continue
            va = p.get("valid_at")
            if va and ref_s and va > ref_s:
                continue
            out.append(self._props_to_episodic(p))
        out.sort(key=lambda e: _g(e, "valid_at") or datetime.min, reverse=True)
        out = out[:last_n]
        out.reverse()  # oldest first
        return out

    async def edge_get_between_nodes(self, _cls, driver, src_uuid, tgt_uuid):
        es = self.d.hg.get_edges_of(src_uuid, direction="OUT")
        out = []
        for e in es:
            if e.get("label") != "RELATES_TO":
                continue
            if e.get("inV") != tgt_uuid:
                continue
            out.append(self._edge_to_object(e))
        return out

    async def edge_get_by_node_uuid(self, _cls, driver, node_uuid):
        out = []
        for d in ("OUT", "IN"):
            es = self.d.hg.get_edges_of(node_uuid, direction=d)
            for e in es:
                if e.get("label") == "RELATES_TO":
                    out.append(self._edge_to_object(e))
        return out

    # -- episode/entity edges (simplified) -- #
    async def next_episode_edge_save(self, edge, driver):
        src = _g(edge, "source_node_uuid"); tgt = _g(edge, "target_node_uuid")
        if src and tgt:
            self.d.hg.upsert_edge("NEXT_EPISODE", src, tgt, "Episodic", "Episodic",
                                  {"uuid": _g(edge, "uuid", ""), "group_id": _g(edge, "group_id", ""),
                                   "created_time": _dt2s(_g(edge, "created_at"))})

    async def has_episode_edge_save(self, edge, driver):
        pass  # saga=None path skips; implement if saga used

    async def saga_node_save(self, node, driver):
        pass  # saga=None path skips

    async def saga_get_previous_episode_uuid(self, driver, saga_uuid, current_uuid):
        return None

    # -- maintenance -- #
    async def clear_data(self, driver, group_ids=None):
        self.d.hg.clear_graph()

    # -- converters -- #
    def _props_to_entity(self, p):
        p = p.get("properties", p)
        return EntityNode(uuid=p.get("uuid", ""), name=p.get("name", ""),
                          group_id=p.get("group_id", ""), summary=_s2j(p.get("summary")) or "",
                          labels=_s2j(p.get("labels")) or ["Entity"],
                          attributes=_s2j(p.get("attributes")) or {},
                          created_at=_s2dt(p.get("created_time")))

    def _props_to_episodic(self, p):
        p = p.get("properties", p)
        return EpisodicNode(uuid=p.get("uuid", ""), name=p.get("name", ""),
                            group_id=p.get("group_id", ""),
                            source=p.get("source", "message"),
                            source_description=p.get("source_description", ""),
                            content=p.get("content", ""),
                            valid_at=_s2dt(p.get("valid_at")),
                            created_at=_s2dt(p.get("created_time")),
                            entity_edges=_s2j(p.get("entity_edges")) or [])

    def _edge_to_object(self, e):
        p = e.get("properties", {})
        return EntityEdge(uuid=p.get("uuid", ""), group_id=p.get("group_id", ""),
                          source_node_uuid=e.get("outV", ""), target_node_uuid=e.get("inV", ""),
                          name=p.get("name", ""), fact=p.get("fact", ""),
                          valid_at=_s2dt(p.get("valid_at")), invalid_at=_s2dt(p.get("invalid_at")),
                          expired_at=_s2dt(p.get("expired_at")),
                          reference_time=_s2dt(p.get("reference_time")),
                          episodes=_s2j(p.get("episodes")) or [],
                          created_at=_s2dt(p.get("created_time")),
                          attributes=_s2j(p.get("attributes")) or {})


# ---------------- SearchInterface ---------------- #
class HugeGraphSearchInterface(SearchInterface):
    """HugeGraph + in-process vector search."""
    d: Any = None

    def __init__(self, driver: "HugeGraphDriver"):
        super().__init__()
        self.d = driver

    async def node_similarity_search(self, driver, search_vector, search_filter=None,
                                     group_ids=None, limit=100, min_score=0.7):
        import numpy as np
        qv = self.d.embedder._to_arr(search_vector)
        out = []
        for v in self.d.hg.get_vertices_by_label("Entity", limit=100000):
            p = v.get("properties", {})
            if group_ids and p.get("group_id") not in group_ids:
                continue
            uid = p.get("uuid")
            emb = self.d.embedder._cache.get(uid)
            if emb is None:
                continue
            score = float(np.dot(qv, emb))
            if min_score and score < min_score:
                continue
            out.append((score, self.d.graph_operations_interface._props_to_entity(p)))
        out.sort(key=lambda x: -x[0])
        return [n for _, n in out[:limit]]

    async def node_fulltext_search(self, driver, query, search_filter=None, group_ids=None, limit=10):
        import re
        qtokens = set(re.findall(r"\w+", query.lower()))
        out = []
        for v in self.d.hg.get_vertices_by_label("Entity", limit=100000):
            p = v.get("properties", {})
            if group_ids and p.get("group_id") not in group_ids:
                continue
            txt = (p.get("name", "") + " " + (p.get("summary") or "")).lower()
            ttoks = set(re.findall(r"\w+", txt))
            score = len(qtokens & ttoks) / max(len(qtokens | ttoks), 1)
            if score > 0:
                out.append((score, self.d.graph_operations_interface._props_to_entity(p)))
        out.sort(key=lambda x: -x[0])
        return [n for _, n in out[:limit]]

    async def edge_fulltext_search(self, driver, query, search_filter=None, group_ids=None, limit=10):
        import re
        qtokens = set(re.findall(r"\w+", query.lower()))
        out = []
        for v in self.d.hg.get_vertices_by_label("Entity", limit=100000):
            es = self.d.hg.get_edges_of(v.get("id"), direction="OUT")
            for e in es:
                if e.get("label") != "RELATES_TO":
                    continue
                p = e.get("properties", {})
                if group_ids and p.get("group_id") not in group_ids:
                    continue
                txt = (p.get("fact", "") + " " + p.get("name", "")).lower()
                ttoks = set(re.findall(r"\w+", txt))
                score = len(qtokens & ttoks) / max(len(qtokens | ttoks), 1)
                if score > 0:
                    out.append((score, self.d.graph_operations_interface._edge_to_object(e)))
        # dedupe by uuid
        seen = set(); res = []
        for s, e in sorted(out, key=lambda x: -x[0]):
            if e.uuid in seen:
                continue
            seen.add(e.uuid); res.append(e)
        return res[:limit]

    async def edge_similarity_search(self, driver, search_vector, source_node_uuid=None,
                                     target_node_uuid=None, search_filter=None,
                                     group_ids=None, limit=100, min_score=0.7):
        import numpy as np
        qv = self.d.embedder._to_arr(search_vector)
        out = []
        for v in self.d.hg.get_vertices_by_label("Entity", limit=100000):
            es = self.d.hg.get_edges_of(v.get("id"), direction="OUT")
            for e in es:
                if e.get("label") != "RELATES_TO":
                    continue
                p = e.get("properties", {})
                if group_ids and p.get("group_id") not in group_ids:
                    continue
                uid = p.get("uuid")
                emb = self.d.embedder._cache.get(uid)
                if emb is None:
                    continue
                score = float(np.dot(qv, emb))
                if min_score and score < min_score:
                    continue
                out.append((score, self.d.graph_operations_interface._edge_to_object(e)))
        out.sort(key=lambda x: -x[0])
        seen = set(); res = []
        for s, e in out:
            if e.uuid in seen:
                continue
            seen.add(e.uuid); res.append(e)
        return res[:limit]

    async def edge_bfs_search(self, driver, origin_uuids, max_depth=3, search_filter=None,
                              group_ids=None, limit=10):
        visited = set(origin_uuids); out = []
        frontier = list(origin_uuids)
        for _ in range(max_depth):
            nxt = []
            for uid in frontier:
                es = self.d.hg.get_edges_of(uid, direction="OUT")
                for e in es:
                    if e.get("label") != "RELATES_TO":
                        continue
                    obj = self.d.graph_operations_interface._edge_to_object(e)
                    if obj.uuid not in visited:
                        out.append(obj); visited.add(obj.uuid)
                    tgt = e.get("inV")
                    if tgt and tgt not in visited:
                        nxt.append(tgt)
            frontier = nxt
            if len(out) >= limit:
                break
        return out[:limit]

    async def node_bfs_search(self, driver, origin_uuids, max_depth=3, search_filter=None,
                              group_ids=None, limit=10):
        visited = set(origin_uuids); out = []
        frontier = list(origin_uuids)
        for _ in range(max_depth):
            nxt = []
            for uid in frontier:
                es = self.d.hg.get_edges_of(uid, direction="OUT")
                for e in es:
                    tgt = e.get("inV")
                    if tgt and tgt not in visited:
                        visited.add(tgt); nxt.append(tgt)
                        p = self.d.hg.get_vertex(tgt)
                        if p:
                            out.append(self.d.graph_operations_interface._props_to_entity(p))
            frontier = nxt
            if len(out) >= limit:
                break
        return out[:limit]

    async def episode_fulltext_search(self, driver, query, search_filter=None, group_ids=None, limit=10):
        import re
        qtokens = set(re.findall(r"\w+", query.lower()))
        out = []
        for v in self.d.hg.get_vertices_by_label("Episodic", limit=100000):
            p = v.get("properties", {})
            if group_ids and p.get("group_id") not in group_ids:
                continue
            txt = (p.get("content", "") + " " + p.get("name", "")).lower()
            ttoks = set(re.findall(r"\w+", txt))
            score = len(qtokens & ttoks) / max(len(qtokens | ttoks), 1)
            if score > 0:
                out.append((score, self.d.graph_operations_interface._props_to_episodic(p)))
        out.sort(key=lambda x: -x[0])
        return [n for _, n in out[:limit]]

    async def node_distance_reranker(self, driver, node_uuids, center_uuid, min_score=0):
        # BFS distance from center
        if not center_uuid:
            return []
        dist = {center_uuid: 0}; frontier = [center_uuid]; d = 0
        while frontier and d < 5:
            d += 1; nxt = []
            for uid in frontier:
                es = self.d.hg.get_edges_of(uid, direction="OUT")
                for e in es:
                    tgt = e.get("inV")
                    if tgt and tgt not in dist:
                        dist[tgt] = d; nxt.append(tgt)
            frontier = nxt
        out = []
        for uid in node_uuids:
            if uid in dist:
                p = self.d.hg.get_vertex(uid)
                if p:
                    out.append(self.d.graph_operations_interface._props_to_entity(p))
        return out

    async def episode_mentions_reranker(self, driver, node_uuids, min_score=0):
        # rank by number of episodes mentioning
        out = []
        for uid in node_uuids:
            es = self.d.hg.get_edges_of(uid, direction="IN")  # MENTIONS target = entity
            n = sum(1 for e in es if e.get("label") == "MENTIONS")
            if n >= min_score:
                p = self.d.hg.get_vertex(uid)
                if p:
                    out.append((n, self.d.graph_operations_interface._props_to_entity(p)))
        out.sort(key=lambda x: -x[0])
        return [n for _, n in out]


# ---------------- Driver ---------------- #
class _HGSession(GraphDriverSession):
    def __init__(self, driver):
        self.d = driver

    async def run(self, query, **kwargs):
        raise NotImplementedError("HugeGraph has no Cypher; use graph_operations_interface")

    async def execute_write(self, func, *args, **kwargs):
        return await func(self, *args, **kwargs)

    async def close(self):
        pass

    async def __aexit__(self, *a):
        pass


class _HGTransaction(Transaction):
    def __init__(self, driver):
        self.d = driver

    async def run(self, query, **kwargs):
        raise NotImplementedError("HugeGraph has no Cypher")


class HugeGraphDriver(GraphDriver):
    """Graphiti GraphDriver backed by HugeGraph 1.7.0 REST."""

    provider = GraphProvider.NEO4J  # reuse to satisfy get_default_group_id; override _database

    def __init__(self, url="http://127.0.0.1:8080", graph="hugegraph",
                 user="admin", pwd="admin", embedder: LocalEmbedder | None = None):
        self.hg = HugeGraphClient(url, graph, user, pwd)
        self.embedder = embedder or LocalEmbedder()
        self._database = graph
        self.default_group_id = graph
        self._graph_ops = HugeGraphGraphOperations(self)
        self._search_iface = HugeGraphSearchInterface(self)

    @property
    def graph_operations_interface(self):
        return self._graph_ops

    @property
    def search_interface(self):
        return self._search_iface

    async def execute_query(self, cypher_query_, **kwargs):
        raise NotImplementedError("HugeGraph has no Cypher engine")

    def session(self, database=None):
        return _HGSession(self)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Transaction]:
        yield _HGTransaction(self)

    async def close(self):
        pass

    async def delete_all_indexes(self):
        pass  # init_schema handles rebuild

    async def build_indices_and_constraints(self, delete_existing=False):
        self.hg.init_schema(rebuild=delete_existing)
        # add NEXT_EPISODE edge label if missing
        try:
            self.hg._safe_create("/schema/edgelabels", {
                "name": "NEXT_EPISODE", "source_label": "Episodic", "target_label": "Episodic",
                "properties": ["uuid", "group_id", "created_time"], "frequency": "MULTIPLE",
                "sort_keys": ["uuid"]})
        except Exception:
            pass

    def with_database(self, database):
        import copy
        c = copy.copy(self)
        c._database = database
        c.hg = HugeGraphClient(self.hg.base, database, self.hg.user, self.hg.pwd)
        return c

    def clone(self, database):
        return self.with_database(database)
