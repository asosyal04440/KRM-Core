from krm.runtime.memory_budget import MemoryBudget, ResourceProfile


def test_memory_budget_degrades_when_too_low() -> None:
    profile = ResourceProfile("micro", 64 * 1024 * 1024, 0, 1024 * 1024, 100, 100, 5, 500)
    budget = MemoryBudget(profile)
    assert budget.clamp_rounds(5) == 1
    assert budget.clamp_edges(1000) == 64
    assert budget.clamp_candidates(100) == 16
    assert budget.clamp_loaded_shards(3) == 1
    assert budget.snippet_chars() == 180
    assert budget.degradation_decisions
