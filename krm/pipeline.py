from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from krm.concepts.concept_card import ConceptCard
from krm.concepts.concept_store import ConceptStore
from krm.concepts.extractor import ConceptExtractor
from krm.graph.ghost_edges import GhostEdgeGenerator, SourceContext
from krm.graph.resonance import ResonanceConfig, ResonanceEngine
from krm.evals.rubric import evaluate_demo_answer, is_demo_query
from krm.index.hybrid_retriever import HybridRetriever
from krm.llm.composer import Composer
from krm.reasoning.answer_plan import AnswerPlanner
from krm.reasoning.policy_seed import PolicySeedGenerator
from krm.reasoning.router import QueryRouter
from krm.runtime.memory_budget import MemoryBudget
from krm.runtime.telemetry import Timer
from krm.shards.shard_builder import ShardBuilder
from krm.shards.shard_loader import ShardLoader
from krm.shards.shard_store import ShardStore
from krm.source.source_pointer import SourceArticle, SourcePointer
from krm.source.zim_reader import CsvSourceReader, JsonlSourceReader, MarkdownSourceReader, PlainTextSourceReader, strip_html


def ensure_mind_dirs(mind_dir: Path) -> None:
    for name in ["mind.skel", "mind.index", "mind.shards", "mind.seeds", "mind.cache"]:
        (mind_dir / name).mkdir(parents=True, exist_ok=True)


def iter_source_articles(source: Path) -> list[SourceArticle]:
    readers = [PlainTextSourceReader(source), MarkdownSourceReader(source)]
    articles: list[SourceArticle] = []
    for reader in readers:
        articles.extend(reader.iter_articles())
    return sorted(articles, key=lambda article: (article.source_type, article.article_id))


def ingest_source(source: Path, mind_dir: Path) -> dict[str, Any]:
    return ingest_articles(iter_source_articles(source), mind_dir)


def ingest_articles(articles: list[SourceArticle], mind_dir: Path) -> dict[str, Any]:
    ensure_mind_dirs(mind_dir)
    extractor = ConceptExtractor()
    store = ConceptStore()
    pointer_path = mind_dir / "mind.skel" / "sources.jsonl"
    edges_path = mind_dir / "mind.skel" / "candidate_edges.jsonl"
    warnings: list[str] = []
    with pointer_path.open("w", encoding="utf-8") as pf, edges_path.open("w", encoding="utf-8") as ef:
        for article in articles:
            pointer = article.pointer()
            pf.write(json.dumps(pointer.to_compact_dict(), ensure_ascii=True) + "\n")
            result = extractor.extract(article)
            warnings.extend(result.warnings)
            name_to_id: dict[str, int] = {}
            for card in result.concepts:
                stored = store.add(card)
                name_to_id[stored.canonical_name] = stored.concept_id
            for edge in result.candidate_edges:
                ef.write(json.dumps(asdict(edge), ensure_ascii=True) + "\n")
    concepts_path = mind_dir / "mind.skel" / "concepts.jsonl"
    store.save_jsonl(concepts_path)
    _write_indexes(store.all(), mind_dir / "mind.index")
    return {
        "articles": len(articles),
        "concepts": len(store.all()),
        "concepts_path": str(concepts_path),
        "sources_path": str(pointer_path),
        "warnings": sorted(set(warnings)),
    }


def build_shards(mind_dir: Path, profile: str) -> dict[str, Any]:
    ensure_mind_dirs(mind_dir)
    budget = MemoryBudget.for_profile(profile)
    concepts = ConceptStore.load_jsonl(mind_dir / "mind.skel" / "concepts.jsonl").all()
    manifests = ShardBuilder().build(concepts, mind_dir, budget)
    seeds_dir = mind_dir / "mind.seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    (seeds_dir / "policy.json").write_text(
        json.dumps({"version": "0.1", "profile": profile, "seed_type": "policy_seed_defaults"}, indent=2),
        encoding="utf-8",
    )
    return {
        "profile": profile,
        "shards": [manifest.to_dict() for manifest in manifests],
        "concept_count": len(concepts),
    }


def run_query(
    mind_dir: Path,
    query: str,
    profile: str = "local_core",
    max_concepts: int | None = None,
    max_edges: int | None = None,
    rounds: int | None = None,
) -> dict[str, Any]:
    timer = Timer.start()
    budget = MemoryBudget.for_profile(profile)
    router = QueryRouter()
    intent = router.classify(query)
    shard_store = ShardStore(mind_dir)
    manifests = shard_store.load_manifests()
    selected = ShardLoader().select(manifests, intent, budget)
    concepts: list[ConceptCard] = []
    for manifest in selected:
        concepts.extend(shard_store.load_concepts(manifest.shard_id))
    if not concepts:
        elapsed = round(timer.elapsed_ms(), 3)
        message = "No sufficient local concepts found. Add more ingestible local files or lower thresholds."
        return {
            "query": query,
            "intent": intent.intent_type.value,
            "profile": profile,
            "selected_shards": [manifest.shard_id for manifest in selected],
            "candidate_concept_count": 0,
            "retrieved_concepts": [],
            "ghost_edge_count": 0,
            "resonance_rounds": 0,
            "top_activated_concepts": [],
            "answer_plan": {},
            "final_answer": message,
            "resonance_trace": {"rounds": [], "top_edges_used": [], "pruning_decisions": [], "degradation_decisions": []},
        "quality_rubric": None,
            "estimated_ram_bytes": 0,
            "timing_ms": elapsed,
            "degradation_decisions": [message],
            "source_snippet_count": 0,
        }
    retriever = HybridRetriever(concepts)
    candidate_limit = max_concepts or budget.profile.default_candidates
    retrieval = retriever.search(query, budget, limit=candidate_limit, intent=intent)
    seed_concepts = [hit.concept for hit in retrieval.hits]
    relevant_sources = {source for card in seed_concepts for source in card.source_refs}
    active_concepts = _filter_source_refs(
        _expand_by_source(seed_concepts, concepts, budget.profile.default_candidates),
        relevant_sources,
    )
    source_contexts = _source_contexts(active_concepts, mind_dir)
    edge_limit = max_edges or budget.profile.default_ghost_edges
    ghost_edges = GhostEdgeGenerator().generate(active_concepts, source_contexts, intent, budget, max_edges=edge_limit)
    initial_scores = {hit.concept.concept_id: hit.score for hit in retrieval.hits}
    for card in active_concepts:
        initial_scores.setdefault(card.concept_id, 0.35)
    graph = ResonanceEngine().run(
        query=query,
        concepts=active_concepts,
        initial_scores=initial_scores,
        ghost_edges=ghost_edges,
        query_intent=intent,
        budget=budget,
        config=ResonanceConfig(
            max_rounds=rounds or budget.profile.default_rounds,
            max_active_concepts=candidate_limit,
            max_edges=edge_limit,
        ),
    )
    policy = PolicySeedGenerator().from_intent(intent)
    plan = AnswerPlanner().plan(query, graph, intent, policy)
    snippets = load_source_snippets(mind_dir, plan.source_references, budget.snippet_chars())
    answer = Composer().compose(query, plan, intent, snippets)
    estimated_ram = budget.ensure_query_fits(len(active_concepts), len(ghost_edges), len(selected))
    top_activated = [
        {"concept_id": card.concept_id, "name": card.canonical_name, "score": graph.activation_scores.get(card.concept_id, 0.0)}
        for card in graph.top_concepts(12)
    ]
    return {
        "query": query,
        "intent": intent.intent_type.value,
        "profile": profile,
        "selected_shards": [manifest.shard_id for manifest in selected],
        "candidate_concept_count": len(active_concepts),
        "retrieved_concepts": [
            {
                "concept_id": hit.concept.concept_id,
                "name": hit.concept.canonical_name,
                "score": hit.score,
                "matched_terms": hit.matched_terms,
                "reasons": hit.reasons,
                "source_pointer_ids": hit.source_pointer_ids,
                "domain": hit.domain,
                "confidence": hit.confidence,
            }
            for hit in retrieval.hits[:20]
        ],
        "ghost_edge_count": len(ghost_edges),
        "resonance_rounds": graph.resonance_rounds,
        "top_activated_concepts": top_activated,
        "answer_plan": plan.to_dict(),
        "final_answer": answer,
        "resonance_trace": graph.trace,
        "quality_rubric": evaluate_demo_answer(query, answer, plan.to_dict()).to_dict() if is_demo_query(query) else None,
        "estimated_ram_bytes": estimated_ram,
        "timing_ms": round(timer.elapsed_ms(), 3),
        "degradation_decisions": sorted(set(budget.degradation_decisions + retrieval.degradation_decisions + graph.warnings)),
        "source_snippet_count": len(snippets),
    }


def inspect_shard(mind_dir: Path, shard_id: str) -> dict[str, Any]:
    store = ShardStore(mind_dir)
    manifest = next((m for m in store.load_manifests() if m.shard_id == shard_id), None)
    concepts = store.load_concepts(shard_id)
    if manifest is None:
        return {"error": f"shard not found: {shard_id}", "available": [m.shard_id for m in store.load_manifests()]}
    top = sorted(concepts, key=lambda card: (-card.importance, card.canonical_name.lower()))[:12]
    return {
        "manifest": manifest.to_dict(),
        "top_concepts": [card.canonical_name for card in top],
    }


def load_source_snippets(mind_dir: Path, pointer_ids: list[str], max_chars: int) -> list[str]:
    pointers = load_source_pointers(mind_dir)
    snippets: list[str] = []
    for pointer_id in pointer_ids:
        pointer = pointers.get(pointer_id)
        if pointer is None:
            continue
        raw_path = pointer.extra.get("path")
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        if pointer.source_type == "jsonl":
            snippets.append(JsonlSourceReader(path, pointer.extra.get("title_field", "title"), pointer.extra.get("text_field", "text")).get_text(pointer)[:max_chars])
            continue
        if pointer.source_type == "csv":
            snippets.append(CsvSourceReader(path, pointer.extra.get("title_field", "title"), pointer.extra.get("text_field", "text")).get_text(pointer)[:max_chars])
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if pointer.source_type == "html":
            text = strip_html(text)
        start = pointer.char_start or 0
        end = pointer.char_end if pointer.char_end is not None else len(text)
        snippets.append(text[start:end].strip()[:max_chars])
    return snippets


def load_source_pointers(mind_dir: Path) -> dict[str, SourcePointer]:
    path = mind_dir / "mind.skel" / "sources.jsonl"
    pointers: dict[str, SourcePointer] = {}
    if not path.exists():
        return pointers
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                pointer = SourcePointer.from_compact_dict(json.loads(line))
                pointers[pointer.pointer_id] = pointer
    return pointers


def _source_contexts(concepts: list[ConceptCard], mind_dir: Path | None = None) -> list[SourceContext]:
    grouped: dict[str, list[int]] = {}
    for card in concepts:
        for source_ref in card.source_refs:
            grouped.setdefault(source_ref, []).append(card.concept_id)
    texts = load_source_snippets(mind_dir, list(grouped), 1000) if mind_dir is not None else []
    text_by_source = dict(zip(grouped, texts, strict=False))
    return [
        SourceContext(source_pointer_id=source, concept_ids=ids, text=text_by_source.get(source, ""))
        for source, ids in sorted(grouped.items())
    ]


def _expand_by_source(seed_concepts: list[ConceptCard], all_concepts: list[ConceptCard], limit: int) -> list[ConceptCard]:
    by_id = {card.concept_id: card for card in seed_concepts}
    seed_sources = {source for card in seed_concepts for source in card.source_refs}
    related = [
        card
        for card in all_concepts
        if card.concept_id not in by_id and seed_sources.intersection(card.source_refs)
    ]
    related.sort(key=lambda card: (-card.importance, card.canonical_name.lower(), card.concept_id))
    for card in related:
        if len(by_id) >= limit:
            break
        by_id[card.concept_id] = card
    return list(by_id.values())


def _filter_source_refs(concepts: list[ConceptCard], allowed_sources: set[str]) -> list[ConceptCard]:
    if not allowed_sources:
        return concepts
    filtered: list[ConceptCard] = []
    for card in concepts:
        refs = [source for source in card.source_refs if source in allowed_sources]
        if refs:
            filtered.append(replace(card, source_refs=refs))
    return filtered


def _write_indexes(concepts: list[ConceptCard], index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    lexical: dict[str, list[int]] = {}
    fingerprints: dict[str, int] = {}
    from krm.index.lexical_index import tokenize

    for card in concepts:
        fingerprints[str(card.concept_id)] = card.short_fingerprint
        for term in sorted(set(tokenize(card.canonical_name + " " + " ".join(card.aliases)))):
            lexical.setdefault(term, []).append(card.concept_id)
    (index_dir / "lexical.json").write_text(json.dumps(lexical, indent=2, ensure_ascii=True), encoding="utf-8")
    (index_dir / "fingerprints.json").write_text(json.dumps(fingerprints, indent=2, ensure_ascii=True), encoding="utf-8")
