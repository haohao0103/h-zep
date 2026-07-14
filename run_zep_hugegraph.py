#!/usr/bin/env python3
"""
End-to-end: Zep/Graphiti interaction flow on HugeGraph 1.7.0.

Demonstrates the full Graphiti-style flow adapted to HugeGraph:
  init schema → add_episode (×3, incl. a fact change to trigger temporal
  invalidation) → hybrid search → temporal point-in-time query.

Run:
  PYTHONPATH=graphiti-source:. \
  /Users/mac/.workbuddy/binaries/python/envs/hg-llm/bin/python3.10 \
  run_zep_hugegraph.py
"""

import json
import logging
import sys
from datetime import datetime, timezone

from zep_hg import HugeGraphClient, LocalEmbedder, ZepHugeGraphEngine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("zep_hg.run")


def main():
    # --- 1. init HugeGraph + embedder + engine ---
    hg = HugeGraphClient("http://127.0.0.1:8080", "hugegraph")
    log.info("initializing HugeGraph schema (rebuild) ...")
    hg.init_schema()

    emb = LocalEmbedder("all-MiniLM-L6-v2", dim=384)
    engine = ZepHugeGraphEngine(hg, emb,
                                llm_model="deepseek-chat")

    # --- 2. add episodes (with a mid-stream fact change) ---
    episodes = [
        ("ep1", "张明在2024年加入Acme公司担任后端工程师，主要用Python开发风控系统。",
         "2024-03-01"),
        ("ep2", "2025年张明跳槽到了Globex公司，开始负责图数据库方向的运维。",
         "2025-06-15"),
        ("ep3", "李华是张明在Globex的同事，擅长关系图谱和HugeGraph集群运维。",
         "2025-09-20"),
    ]
    for name, body, date in episodes:
        ref = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        log.info("=== add_episode %s (ref=%s) ===", name, date)
        r = engine.add_episode(
            name=name, body=body,
            source_description="PoC conversation", reference_time=ref,
            group_id="poc_zep", source="message")
        log.info("  -> entities=%d edges=%d", r["entities"], r["edges"])
        for ed in r["edges_detail"]:
            log.info("     edge: %s --[%s]--> %s  (valid=%s invalid=%s)",
                     ed.get("source_node_uuid", "?")[:8],
                     ed.get("fact", "")[:40], ed.get("target_node_uuid", "?")[:8],
                     ed.get("valid_at"), ed.get("invalid_at"))

    # --- 3. hybrid search (current state) ---
    log.info("=== hybrid search: 张明现在在哪家公司工作 ===")
    res = engine.search("张明现在在哪家公司工作", group_id="poc_zep", num_results=5)
    _print_search(res)

    log.info("=== hybrid search: 谁擅长图数据库运维 ===")
    res = engine.search("谁擅长图数据库运维", group_id="poc_zep", num_results=5)
    _print_search(res)

    # --- 4. temporal point-in-time query (should see the OLD employer) ---
    pit = datetime(2024, 10, 1, tzinfo=timezone.utc)
    log.info("=== temporal query @ 2024-10-01: 张明在哪家公司 ===")
    tres = engine.temporal_query("张明在哪家公司工作", point_in_time=pit,
                                 group_id="poc_zep", num_results=5)
    _print_search(tres)

    # --- 5. summary ---
    log.info("=== HugeGraph stats ===")
    log.info("  Entity vertices: %d", hg.count("Entity"))
    log.info("  Episodic vertices: %d", hg.count("Episodic"))
    log.info("DONE. Zep/Graphiti flow adapted to HugeGraph 1.7.0 successfully.")


def _print_search(res: dict):
    log.info("  query: %s", res.get("query"))
    log.info("  stats: %s", res.get("stats"))
    for i, r in enumerate(res.get("results", []), 1):
        log.info("  #%d [%.4f] %s --[%s]--> %s  valid=%s invalid=%s",
                 i, r["score"], r["source"], r["fact"][:50],
                 r["target"], r.get("valid_at"), r.get("invalid_at"))


if __name__ == "__main__":
    sys.exit(main())
