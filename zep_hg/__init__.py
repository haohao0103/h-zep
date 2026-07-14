"""
Zep/Graphiti → HugeGraph adapter.

Adapts Zep's Graphiti temporal-knowledge-graph interaction flow onto a real
HugeGraph 1.7.0 backend via REST API (Gremlin is unavailable in this build).

See docs/ZEP_HUGEGRAPH_ADAPTER.md for the architecture & benchmark.
"""

from .driver import HugeGraphClient
from .embedder import LocalEmbedder
from .engine import ZepHugeGraphEngine

__all__ = ["HugeGraphClient", "LocalEmbedder", "ZepHugeGraphEngine"]
