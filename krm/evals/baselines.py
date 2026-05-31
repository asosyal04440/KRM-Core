from __future__ import annotations

from pathlib import Path
from typing import Any

from krm.index.hybrid_retriever import HybridRetriever
from krm.index.lexical_index import LexicalIndex
from krm.pipeline import run_query
from krm.reasoning.router import QueryRouter
from krm.runtime.memory_budget import MemoryBudget
from krm.runtime.telemetry import Timer
from krm.shards.shard_store import ShardStore


def _concepts(mind_dir: Path):
    store = ShardStore(mind_dir)
    concepts = store.load_concepts("general")
    if concepts:
        return concepts
    all_concepts = []
    for manifest in store.load_manifests():
        all_concepts.extend(store.load_concepts(manifest.shard_id))
    return all_concepts


def lexical_only_baseline(mind_dir: Path, query: str, limit: int = 10) -> dict[str, Any]:
    timer = Timer.start()
    concepts = _concepts(mind_dir)
    by_id = {card.concept_id: card for card in concepts}
    hits = LexicalIndex.from_concepts(concepts).search(query, limit)
    top = [by_id[cid].canonical_name for cid in hits if cid in by_id]
    return {"top_concepts": top, "timing_ms": round(timer.elapsed_ms(), 3)}


def retrieval_without_resonance_baseline(mind_dir: Path, query: str, profile: str = "local_core", limit: int = 10) -> dict[str, Any]:
    timer = Timer.start()
    budget = MemoryBudget.for_profile(profile)
    intent = QueryRouter().classify(query)
    result = HybridRetriever(_concepts(mind_dir)).search(query, budget, limit=limit, intent=intent)
    return {
        "top_concepts": [hit.concept.canonical_name for hit in result.hits[:limit]],
        "estimated_ram_bytes": result.estimated_ram_bytes,
        "timing_ms": round(timer.elapsed_ms(), 3),
    }


def krm_resonance_answer(mind_dir: Path, query: str, profile: str = "local_core") -> dict[str, Any]:
    result = run_query(mind_dir, query, profile)
    return {
        "top_concepts": [item["name"] for item in result["top_activated_concepts"][:10]],
        "answer": result["final_answer"],
        "estimated_ram_bytes": result["estimated_ram_bytes"],
        "timing_ms": result["timing_ms"],
    }
