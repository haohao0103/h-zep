"""
Unit tests for HugeGraphDriver / HugeGraphGraphOperations / HugeGraphSearchInterface.

Uses fake HugeGraph client + fake embedder so tests run offline (no LLM, no
HugeGraph server). Covers the P0 methods that Graphiti's add_episode/search
call链 depends on.
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "graphiti-source"))

from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.edges import EntityEdge
from zep_hg.hugegraph_driver import (
    HugeGraphDriver, HugeGraphGraphOperations, HugeGraphSearchInterface,
    _g, _dt2s, _s2dt, _j2s, _s2j,
)


# ----------------- fakes ----------------- #
class FakeHugeGraphClient:
    def __init__(self):
        self.vertices = {}   # vid -> {"label":..., "properties":{...}}
        self.edges = []      # list of {"label","outV","inV","properties"}

    def upsert_vertex(self, label, vid, props):
        self.vertices[vid] = {"label": label, "properties": {k: v for k, v in props.items() if v is not None}}

    def get_vertex(self, vid):
        v = self.vertices.get(vid)
        if not v:
            return None
        return {"id": vid, "label": v["label"], "properties": v["properties"]}

    def delete_vertex(self, vid):
        self.vertices.pop(vid, None)

    def get_vertices_by_label(self, label, limit=1000):
        return [{"id": k, "label": label, "properties": v["properties"]}
                for k, v in self.vertices.items() if v["label"] == label][:limit]

    def upsert_edge(self, label, src, tgt, slabel, tlabel, props):
        self.edges.append({"label": label, "outV": src, "inV": tgt,
                           "outVLabel": slabel, "inVLabel": tlabel,
                           "properties": {k: v for k, v in props.items() if v is not None}})

    def update_edge(self, eid, label, props):
        for e in self.edges:
            if e.get("id") == eid or e["properties"].get("uuid") == eid:
                e["properties"].update({k: v for k, v in props.items() if v is not None})

    def get_edges_of(self, vid, direction="OUT", limit=1000):
        out = []
        for e in self.edges:
            if direction in ("OUT", "BOTH") and e["outV"] == vid:
                out.append(e)
            if direction in ("IN", "BOTH") and e["inV"] == vid:
                out.append(e)
        return out[:limit]

    def count(self, label):
        return sum(1 for v in self.vertices.values() if v["label"] == label)

    def init_schema(self, rebuild=True): pass
    def clear_graph(self):
        self.vertices.clear(); self.edges.clear()


class FakeEmbedder:
    def __init__(self):
        self._cache = {}
        self.dim = 4
    def embed(self, text, key=None):
        k = key or text
        if k not in self._cache:
            rng = np.random.default_rng(hash(k) % 2**32)
            v = rng.standard_normal(self.dim).astype(np.float32)
            v /= np.linalg.norm(v)
            self._cache[k] = v
        return self._cache[k]
    def _to_arr(self, vec):
        if isinstance(vec, np.ndarray):
            return vec
        arr = np.array(vec, dtype=np.float32)
        n = np.linalg.norm(arr)
        return arr / n if n > 0 else arr
    def cosine(self, a, b):
        return float(np.dot(a, b))


def make_driver():
    hg = FakeHugeGraphClient()
    emb = FakeEmbedder()
    d = HugeGraphDriver(hg_client=hg, embedder=emb)
    return d, hg, emb


# ----------------- helper tests ----------------- #
class TestHelpers:
    def test_g_object(self):
        class O: name = "x"
        assert _g(O(), "name") == "x"
        assert _g(O(), "missing", "d") == "d"

    def test_g_dict(self):
        assert _g({"name": "y"}, "name") == "y"
        assert _g({}, "missing", "d") == "d"

    def test_dt2s_none(self):
        assert _dt2s(None) is None

    def test_dt2s_datetime(self):
        dt = datetime(2024, 3, 1, tzinfo=timezone.utc)
        assert _dt2s(dt) == "2024-03-01T00:00:00+00:00"

    def test_dt2s_passthrough_str(self):
        assert _dt2s("2024-03-01") == "2024-03-01"

    def test_s2dt_none(self):
        assert _s2dt(None) is None
        assert _s2dt("") is None
        assert _s2dt("None") is None

    def test_s2dt_valid(self):
        dt = _s2dt("2024-03-01T00:00:00+00:00")
        assert dt.year == 2024 and dt.month == 3

    def test_s2dt_passthrough(self):
        dt = datetime(2025, 1, 1)
        assert _s2dt(dt) is dt

    def test_j2s_list(self):
        assert json.loads(_j2s(["a", "b"])) == ["a", "b"]

    def test_j2s_dict(self):
        assert json.loads(_j2s({"k": "v"})) == {"k": "v"}

    def test_j2s_none(self):
        assert _j2s(None) is None

    def test_s2j_none(self):
        assert _s2j(None) is None
        assert _s2j("") is None

    def test_s2j_valid(self):
        assert _s2j('["a","b"]') == ["a", "b"]


# ----------------- GraphOperations tests ----------------- #
class TestGraphOperations:
    @pytest.mark.asyncio
    async def test_node_save_bulk(self):
        d, hg, emb = make_driver()
        n = EntityNode(uuid="u1", name="张明", group_id="g1", created_at=datetime(2024,3,1,tzinfo=timezone.utc))
        await d.graph_operations_interface.node_save_bulk(None, d, None, [n])
        assert "u1" in hg.vertices
        assert hg.vertices["u1"]["properties"]["name"] == "张明"
        assert hg.vertices["u1"]["label"] == "Entity"

    @pytest.mark.asyncio
    async def test_episodic_node_save_bulk(self):
        d, hg, emb = make_driver()
        ep = EpisodicNode(uuid="ep1", name="ep", group_id="g1", source="message",
                          source_description="test", content="hello", valid_at=datetime(2024,3,1,tzinfo=timezone.utc),
                          created_at=datetime(2024,3,1,tzinfo=timezone.utc))
        await d.graph_operations_interface.episodic_node_save_bulk(None, d, None, [ep])
        assert hg.vertices["ep1"]["label"] == "Episodic"
        assert hg.vertices["ep1"]["properties"]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_edge_save_bulk(self):
        d, hg, emb = make_driver()
        # pre-create nodes so edge endpoints exist
        hg.upsert_vertex("Entity", "s1", {"uuid":"s1","name":"A","group_id":"g","created_time":"2024-01-01"})
        hg.upsert_vertex("Entity", "t1", {"uuid":"t1","name":"B","group_id":"g","created_time":"2024-01-01"})
        e = EntityEdge(uuid="e1", group_id="g", source_node_uuid="s1", target_node_uuid="t1",
                       name="works_at", fact="A works at B", created_at=datetime(2024,3,1,tzinfo=timezone.utc),
                       episodes=["ep1"])
        await d.graph_operations_interface.edge_save_bulk(None, d, None, [e])
        assert len(hg.edges) == 1
        assert hg.edges[0]["label"] == "RELATES_TO"
        assert hg.edges[0]["properties"]["fact"] == "A works at B"

    @pytest.mark.asyncio
    async def test_episodic_edge_save_bulk(self):
        d, hg, emb = make_driver()
        hg.upsert_vertex("Episodic", "ep1", {"uuid":"ep1"})
        hg.upsert_vertex("Entity", "e1", {"uuid":"e1"})
        edge_dict = {"uuid":"me1","source_node_uuid":"ep1","target_node_uuid":"e1",
                     "group_id":"g","created_at":datetime(2024,3,1,tzinfo=timezone.utc)}
        await d.graph_operations_interface.episodic_edge_save_bulk(None, d, None, [edge_dict])
        assert any(e["label"] == "MENTIONS" for e in hg.edges)

    @pytest.mark.asyncio
    async def test_node_get_by_uuid(self):
        d, hg, emb = make_driver()
        hg.upsert_vertex("Entity", "u1", {"uuid":"u1","name":"张明","group_id":"g",
                                          "summary":"s","labels":'["Entity"]',"attributes":'{}',
                                          "created_time":"2024-03-01T00:00:00+00:00"})
        n = await d.graph_operations_interface.node_get_by_uuid(None, d, "u1")
        assert n is not None
        assert n.name == "张明"
        assert n.uuid == "u1"

    @pytest.mark.asyncio
    async def test_node_get_by_uuids(self):
        d, hg, emb = make_driver()
        hg.upsert_vertex("Entity", "u1", {"uuid":"u1","name":"A","group_id":"g","created_time":"2024-01-01","summary":"","labels":'[]',"attributes":'{}'})
        hg.upsert_vertex("Entity", "u2", {"uuid":"u2","name":"B","group_id":"g","created_time":"2024-01-01","summary":"","labels":'[]',"attributes":'{}'})
        ns = await d.graph_operations_interface.node_get_by_uuids(None, d, ["u1","u2","missing"])
        assert len(ns) == 2

    @pytest.mark.asyncio
    async def test_retrieve_episodes(self):
        d, hg, emb = make_driver()
        for i, dt in enumerate(["2024-01-01","2024-06-01","2025-01-01"]):
            hg.upsert_vertex("Episodic", f"ep{i}", {"uuid":f"ep{i}","name":"ep","group_id":"g",
                "content":f"c{i}","valid_at":dt,"created_time":dt,
                "source":"message","source_description":"","entity_edges":"[]"})
        eps = await d.graph_operations_interface.retrieve_episodes(
            d, datetime(2024,12,1,tzinfo=timezone.utc), last_n=2, group_ids=["g"])
        assert len(eps) == 2
        # oldest first
        assert eps[0].valid_at.year == 2024 and eps[0].valid_at.month == 1

    @pytest.mark.asyncio
    async def test_edge_get_between_nodes(self):
        d, hg, emb = make_driver()
        hg.upsert_edge("RELATES_TO", "s1", "t1", "Entity", "Entity",
                       {"uuid":"e1","fact":"A->B"})
        hg.upsert_edge("RELATES_TO", "s1", "t2", "Entity", "Entity",
                       {"uuid":"e2","fact":"A->C"})
        es = await d.graph_operations_interface.edge_get_between_nodes(None, d, "s1", "t1")
        assert len(es) == 1
        assert es[0].fact == "A->B"

    def test_props_to_entity(self):
        d, hg, emb = make_driver()
        p = {"uuid":"u1","name":"A","group_id":"g","summary":"s","labels":'["Entity"]',
             "attributes":'{"k":"v"}',"created_time":"2024-03-01T00:00:00+00:00"}
        n = d.graph_operations_interface._props_to_entity(p)
        assert n.uuid == "u1" and n.name == "A"
        assert n.labels == ["Entity"]
        assert n.attributes == {"k": "v"}

    def test_props_to_episodic(self):
        d, hg, emb = make_driver()
        p = {"uuid":"ep1","name":"ep","group_id":"g","source":"message",
             "source_description":"desc","content":"hello",
             "valid_at":"2024-03-01","created_time":"2024-03-01","entity_edges":'["e1"]'}
        ep = d.graph_operations_interface._props_to_episodic(p)
        assert ep.content == "hello"
        assert ep.entity_edges == ["e1"]

    def test_edge_to_object(self):
        d, hg, emb = make_driver()
        e = {"label":"RELATES_TO","outV":"s1","inV":"t1",
             "properties":{"uuid":"e1","fact":"works","valid_at":"2024-01-01",
                           "invalid_at":"2025-01-01","episodes":'["ep1"]'}}
        obj = d.graph_operations_interface._edge_to_object(e)
        assert obj.fact == "works"
        assert obj.source_node_uuid == "s1"
        assert obj.invalid_at.year == 2025


# ----------------- SearchInterface tests ----------------- #
class TestSearchInterface:
    @pytest.mark.asyncio
    async def test_node_similarity_search(self):
        d, hg, emb = make_driver()
        hg.upsert_vertex("Entity", "u1", {"uuid":"u1","name":"张明","group_id":"g","summary":"","labels":'[]',"attributes":'{}',"created_time":"2024-01-01"})
        emb._cache["u1"] = np.array([1,0,0,0], dtype=np.float32)
        qv = np.array([1,0,0,0], dtype=np.float32)
        res = await d.search_interface.node_similarity_search(d, qv, None, ["g"], 10, 0.5)
        assert len(res) == 1
        assert res[0].uuid == "u1"

    @pytest.mark.asyncio
    async def test_node_fulltext_search(self):
        d, hg, emb = make_driver()
        hg.upsert_vertex("Entity", "u1", {"uuid":"u1","name":"张明 腾讯","group_id":"g","summary":"engineer","labels":'[]',"attributes":'{}',"created_time":"2024-01-01"})
        res = await d.search_interface.node_fulltext_search(d, "张明 腾讯", None, ["g"], 10)
        assert len(res) >= 1
        assert res[0].name == "张明 腾讯"

    @pytest.mark.asyncio
    async def test_edge_fulltext_search(self):
        d, hg, emb = make_driver()
        hg.upsert_vertex("Entity", "s1", {"uuid":"s1","name":"A","group_id":"g","summary":"","labels":'[]',"attributes":'{}',"created_time":"2024-01-01"})
        hg.upsert_edge("RELATES_TO", "s1", "t1", "Entity", "Entity",
                       {"uuid":"e1","fact":"张明在腾讯工作","group_id":"g","name":"works_at","valid_at":"2024-01-01","episodes":"[]"})
        res = await d.search_interface.edge_fulltext_search(d, "腾讯", None, ["g"], 10)
        assert len(res) >= 1

    @pytest.mark.asyncio
    async def test_edge_bfs_search(self):
        d, hg, emb = make_driver()
        hg.upsert_edge("RELATES_TO", "a", "b", "Entity", "Entity", {"uuid":"e1","fact":"a-b"})
        hg.upsert_edge("RELATES_TO", "b", "c", "Entity", "Entity", {"uuid":"e2","fact":"b-c"})
        res = await d.search_interface.edge_bfs_search(d, ["a"], max_depth=2, limit=10)
        assert len(res) >= 2

    @pytest.mark.asyncio
    async def test_node_distance_reranker(self):
        d, hg, emb = make_driver()
        hg.upsert_vertex("Entity", "c", {"uuid":"c","name":"C","group_id":"g","summary":"","labels":'[]',"attributes":'{}',"created_time":"2024-01-01"})
        hg.upsert_edge("RELATES_TO", "c", "x", "Entity", "Entity", {"uuid":"e1"})
        res = await d.search_interface.node_distance_reranker(d, ["c"], "c", 0)
        assert len(res) == 1


# ----------------- Driver tests ----------------- #
class TestDriver:
    def test_init(self):
        d, hg, emb = make_driver()
        assert d.provider is not None
        assert isinstance(d.graph_operations_interface, HugeGraphGraphOperations)
        assert isinstance(d.search_interface, HugeGraphSearchInterface)

    @pytest.mark.asyncio
    async def test_build_indices(self):
        d, hg, emb = make_driver()
        await d.build_indices_and_constraints(delete_existing=True)
        # fake client init_schema is no-op; just verify no exception
        assert True

    @pytest.mark.asyncio
    async def test_session_raises_on_cypher(self):
        d, hg, emb = make_driver()
        s = d.session()
        with pytest.raises(NotImplementedError):
            await s.run("MATCH (n) RETURN n")

    @pytest.mark.asyncio
    async def test_transaction_context(self):
        d, hg, emb = make_driver()
        async with d.transaction() as tx:
            assert tx is not None
