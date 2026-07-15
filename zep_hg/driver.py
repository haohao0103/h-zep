"""
HugeGraph 1.7.0 REST client for the Zep/Graphiti adapter.

Design notes (verified against HugeGraph 1.7.0 @ localhost:8080):
- Gremlin script engine (gremlin-groovy) is NOT available in this build, so all
  reads/writes go through the REST API + traverser endpoints.
- Vertex id strategy = CUSTOMIZE_STRING; the vertex id IS the entity uuid, so a
  "get by uuid" is just a GET by id (no secondary index required).
- Vertex update requires the id be wrapped in double quotes in the URL path and
  `?action=append` to merge properties (documented HugeGraph 1.7 quirk).
- list/dict model fields (labels, attributes, entity_edges, episodes) are stored
  as JSON strings in TEXT properties (HugeGraph has no native list-of-mixed type).
- Embeddings (name_embedding / fact_embedding) are NOT persisted to HugeGraph;
  they live in the in-process embedder cache (see engine.py). HugeGraph 1.7 lacks
  a usable vector index on par with Neo4j/FalkorDB, so vector search is done
  in-process over the cache while graph traversal / fulltext go to HugeGraph.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Schema definitions
# --------------------------------------------------------------------------- #
# All properties are TEXT; structured fields are JSON-encoded strings.
_PROPERTY_KEYS: list[str] = [
    "uuid", "name", "group_id", "summary", "content", "source",
    "source_description", "valid_at", "created_time", "expired_at",
    "invalid_at", "fact", "reference_time", "labels", "attributes",
    "entity_edges", "episodes", "source_node_uuid", "target_node_uuid",
]

_VERTEX_LABELS: dict[str, list[str]] = {
    "Entity": ["uuid", "name", "group_id", "summary", "labels",
               "attributes", "created_time"],
    "Episodic": ["uuid", "name", "group_id", "source", "source_description",
                 "content", "valid_at", "created_time", "entity_edges"],
}

_EDGE_LABELS: dict[tuple[str, str, str], list[str]] = {
    # (label, source_label, target_label): properties
    ("RELATES_TO", "Entity", "Entity"): [
        "uuid", "group_id", "name", "fact", "valid_at", "invalid_at", "expired_at",
        "reference_time", "episodes", "created_time", "attributes",
        "source_node_uuid", "target_node_uuid"],
    ("MENTIONS", "Episodic", "Entity"): [
        "uuid", "group_id", "created_time"],
    ("NEXT_EPISODE", "Episodic", "Episodic"): [
        "uuid", "group_id", "created_time"],
}


def _json_str(v: Any) -> str | None:
    """Encode list/dict model fields as JSON strings for HugeGraph TEXT props."""
    if v is None:
        return None
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


class HugeGraphClient:
    """Synchronous HugeGraph 1.7.0 REST client."""

    def __init__(self, url: str = "http://127.0.0.1:8080",
                 graph: str = "hugegraph",
                 user: str = "admin", pwd: str = "admin"):
        self.base = url.rstrip("/")
        self.graph = graph
        self.user = user
        self.pwd = pwd
        self._g = f"{self.base}/graphs/{graph}"
        self._s = requests.Session()
        # HugeGraph may gzip; requests auto-decompresses. Ask for identity to
        # avoid any transfer-encoding surprises on some endpoints.
        self._s.headers.update({"Accept-Encoding": "identity"})

    # ----- low-level ------------------------------------------------------- #
    def _req(self, method: str, path: str, **kw) -> Any:
        url = f"{self._g}{path}"
        r = self._s.request(method, url, timeout=30, **kw)
        if r.status_code >= 400:
            # Try to surface the HugeGraph exception message
            try:
                body = r.json()
                msg = body.get("message", body.get("exception", r.text))
            except Exception:
                msg = r.text[:200]
            raise HugeGraphError(f"{method} {path} -> {r.status_code}: {msg}")
        if not r.content:
            return None
        try:
            return r.json()
        except ValueError:
            return r.text

    @staticmethod
    def _qid(vid: str) -> str:
        """Wrap a CUSTOMIZE_STRING vertex id in double quotes for URL path."""
        return f'"{vid}"'

    # ----- schema ---------------------------------------------------------- #
    def init_schema(self, rebuild: bool = True) -> None:
        """Create property keys, vertex/edge labels (idempotent).

        With rebuild=True (default for clean PoC runs): drop existing labels
        after clearing their data, then recreate with the full property set.
        This sidesteps the case where a label was created earlier with a
        partial property list (HugeGraph cannot alter a label's properties
        in place).
        """
        if rebuild:
            self.clear_graph()
            # drop edge labels first (they reference vertex labels)
            for (name, _, _) in _EDGE_LABELS:
                self._safe_delete(f"/schema/edgelabels/{name}")
            for name in _VERTEX_LABELS:
                self._safe_delete(f"/schema/vertexlabels/{name}")
        for pk in _PROPERTY_KEYS:
            self._safe_create(
                "/schema/propertykeys",
                {"name": pk, "data_type": "TEXT", "cardinality": "SINGLE"})
        for name, props in _VERTEX_LABELS.items():
            self._safe_create(
                "/schema/vertexlabels",
                {"name": name, "id_strategy": "CUSTOMIZE_STRING",
                 "properties": props, "nullable_keys": props,
                 "enable_label_index": True})
        for (name, slabel, tlabel), props in _EDGE_LABELS.items():
            self._safe_create(
                "/schema/edgelabels",
                {"name": name, "source_label": slabel, "target_label": tlabel,
                 "properties": props, "frequency": "MULTIPLE",
                 "sort_keys": ["uuid"],
                 "nullable_keys": [p for p in props if p != "uuid"]})

    def _safe_create(self, path: str, body: dict) -> None:
        try:
            self._req("POST", path, json=body)
        except HugeGraphError as e:
            low = str(e).lower()
            if "already exist" in low or "existed" in low or "has exist" in low:
                logger.debug("schema exists: %s", body.get("name"))
            else:
                raise

    def _safe_delete(self, path: str) -> None:
        try:
            self._req("DELETE", path)
        except HugeGraphError:
            pass

    def clear_graph(self) -> None:
        """Delete all vertices & edges (for a clean PoC run)."""
        # HugeGraph has no 'drop all' REST; iterate labels.
        for label in list(_VERTEX_LABELS):
            try:
                vs = self.get_vertices_by_label(label, limit=10000)
                for v in vs:
                    self.delete_vertex(v["id"])
            except HugeGraphError as e:
                logger.warning("clear %s: %s", label, e)

    # ----- vertex ---------------------------------------------------------- #
    def upsert_vertex(self, label: str, vid: str, props: dict) -> dict:
        """Create vertex; if it exists, append-merge properties."""
        body = {"label": label, "id": vid,
                "properties": {k: _json_str(v) for k, v in props.items() if v is not None}}
        try:
            return self._req("POST", "/graph/vertices", json=body)
        except HugeGraphError:
            return self.update_vertex(label, vid, props)

    def update_vertex(self, label: str, vid: str, props: dict) -> dict:
        body = {"label": label,
                "properties": {k: _json_str(v) for k, v in props.items() if v is not None}}
        # HugeGraph 1.7 quirk: id wrapped in double quotes + action=append
        return self._req("PUT", f"/graph/vertices/{self._qid(vid)}?action=append",
                         json=body)

    def get_vertex(self, vid: str) -> dict | None:
        try:
            return self._req("GET", f"/graph/vertices/{self._qid(vid)}")
        except HugeGraphError as e:
            if "404" in str(e) or "not exist" in str(e).lower():
                return None
            raise

    def delete_vertex(self, vid: str) -> None:
        try:
            self._req("DELETE", f"/graph/vertices/{self._qid(vid)}")
        except HugeGraphError as e:
            logger.debug("delete %s: %s", vid, e)

    def get_vertices_by_label(self, label: str, limit: int = 1000) -> list[dict]:
        r = self._req("GET", f"/graph/vertices?label={label}&limit={limit}")
        return r.get("vertices", []) if r else []

    # ----- edge ------------------------------------------------------------ #
    def upsert_edge(self, label: str, src: str, tgt: str, slabel: str,
                    tlabel: str, props: dict) -> dict:
        body = {"label": label, "outV": src, "inV": tgt,
                "outVLabel": slabel, "inVLabel": tlabel,
                "properties": {k: _json_str(v) for k, v in props.items() if v is not None}}
        return self._req("POST", "/graph/edges", json=body)

    def update_edge(self, edge_id: str, label: str, props: dict) -> dict:
        """Append-merge properties onto an existing edge (by edge id)."""
        body = {"label": label,
                "properties": {k: _json_str(v) for k, v in props.items() if v is not None}}
        return self._req("PUT", f"/graph/edges/{self._qid(edge_id)}?action=append",
                         json=body)

    def get_edges_of(self, vid: str, direction: str = "OUT",
                     limit: int = 1000) -> list[dict]:
        """direction: OUT / IN / BOTH."""
        r = self._req("GET",
                      f"/graph/edges?vertex_id={self._qid(vid)}"
                      f"&direction={direction}&limit={limit}")
        return r.get("edges", []) if r else []

    # ----- traversal ------------------------------------------------------- #
    def neighbors(self, vid: str, direction: str = "OUT",
                  edge_label: str | None = None, max_depth: int = 1) -> list[dict]:
        """One-hop neighbors via the kout traverser, returns vertex objects."""
        lbl = f"&edge_label={edge_label}" if edge_label else ""
        r = self._req("GET",
                      f"/traversers/kneighbor?source={self._qid(vid)}"
                      f"&direction={direction}&max_depth={max_depth}{lbl}")
        return r.get("vertices", []) if r else []

    # ----- stats ----------------------------------------------------------- #
    def count(self, label: str) -> int:
        return len(self.get_vertices_by_label(label, limit=100000))


class HugeGraphError(RuntimeError):
    pass
