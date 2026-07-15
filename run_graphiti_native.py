#!/usr/bin/env python3
"""Test Graphiti native add_episode on HugeGraph via HugeGraphDriver."""
import asyncio, os, sys
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('OPENAI_API_KEY', 'sk-624b175e39f543429cd402555021e7f9')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'graphiti-source'))

from datetime import datetime, timezone
from graphiti_core import Graphiti
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.client import EmbedderClient
from zep_hg.hugegraph_driver import HugeGraphDriver
from zep_hg.embedder import LocalEmbedder

DEEPSEEK_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""


class LocalEmbedderClient(EmbedderClient):
    """Adapt LocalEmbedder to Graphiti's EmbedderClient interface."""
    def __init__(self, emb):
        self._emb = emb
    async def create(self, input_data):
        # Graphiti may pass str or list[str]; normalize to str
        if isinstance(input_data, list):
            input_data = " ".join(str(x) for x in input_data)
        return self._emb.embed(input_data).tolist()
    async def create_batch(self, input_data_list):
        return [self._emb.embed(t if isinstance(t, str) else " ".join(t)).tolist() for t in input_data_list]


async def main():
    emb = LocalEmbedder()
    driver = HugeGraphDriver(embedder=emb)
    driver._database = "hugegraph"  # match group_id to avoid clone
    print("building indices...")
    await driver.build_indices_and_constraints(delete_existing=True)

    llm = OpenAIGenericClient(config=LLMConfig(
        api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1",
        model="deepseek-chat"), structured_output_mode='json_object')
    embedder_client = LocalEmbedderClient(emb)

    print("init Graphiti (native)...")
    g = Graphiti(graph_driver=driver, llm_client=llm, embedder=embedder_client)

    print("add_episode 1 (张明在腾讯, 2024-03)...")
    r1 = await g.add_episode(
        name="ep1", episode_body="张明2024年在腾讯做后端开发，用Java写支付模块。",
        source_description="native test",
        reference_time=datetime(2024, 3, 1, tzinfo=timezone.utc),
        group_id="hugegraph")
    print(f"  -> nodes={len(r1.nodes)} edges={len(r1.edges)}")

    print("add_episode 2 (张明跳槽字节, 2025-03)...")
    r2 = await g.add_episode(
        name="ep2", episode_body="张明2025年从腾讯跳槽到字节跳动做算法，改用Python。",
        source_description="native test",
        reference_time=datetime(2025, 3, 1, tzinfo=timezone.utc),
        group_id="hugegraph")
    print(f"  -> nodes={len(r2.nodes)} edges={len(r2.edges)}")
    print(f"  invalidated edges: {len(r2.edges) - len([e for e in r2.edges if not getattr(e,'invalid_at',None)])}")

    print("search: 张明现在在哪家公司...")
    res = await g.search("张明现在在哪家公司", group_ids=["hugegraph"], num_results=5)
    for i, edge in enumerate(res[:3], 1):
        print(f"  #{i} {getattr(edge,'fact','')[:50]} valid={getattr(edge,'valid_at',None)} invalid={getattr(edge,'invalid_at',None)}")

    print("\nGraphiti native on HugeGraph: OK")
    await g.close()


if __name__ == "__main__":
    asyncio.run(main())
