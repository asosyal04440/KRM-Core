from __future__ import annotations

from krm.training.record_format import TrainingRecord


SPECIAL_TOKENS: tuple[str, ...] = (
    "<|SOURCE|>",
    "<|QUERY|>",
    "<|CONTEXT|>",
    "<|CONCEPTS|>",
    "<|EDGES|>",
    "<|HOT_GRAPH|>",
    "<|POLICY|>",
    "<|ANSWER_PLAN|>",
    "<|ANSWER|>",
    "<|INTENT|>",
    "<|DOMAIN|>",
    "<|SHARD|>",
    "<|UNCERTAIN|>",
    "<|SPECULATIVE|>",
    "<|DO_NOT_CLAIM|>",
    "<|SUPPORTED_BY_SOURCE|>",
    "<|MISSING_INFO|>",
    "<|END|>",
)


def list_special_tokens() -> list[str]:
    return list(SPECIAL_TOKENS)


def validate_no_collision(extra_tokens: list[str] | None = None, *, full_list: bool = False) -> None:
    if full_list:
        tokens = list(extra_tokens or [])
    else:
        tokens = list(SPECIAL_TOKENS) + list(extra_tokens or [])
    if len(tokens) != len(set(tokens)):
        raise ValueError("special token collision detected")
    for token in tokens:
        if not (token.startswith("<|") and token.endswith("|>")):
            raise ValueError(f"invalid special token shape: {token}")


def format_training_prompt(record: TrainingRecord) -> str:
    domains = ", ".join(str(domain) for domain in (record.metadata.get("domains") or []))
    intent = str(record.metadata.get("intent") or "")
    source_refs = ", ".join(str(source_ref) for source_ref in (record.metadata.get("source_refs") or []))
    flags: list[str] = []
    if record.task.value == "COUNTERFACTUAL_LABELING":
        flags.append("<|SPECULATIVE|>")
    if record.task.value == "DO_NOT_CLAIM":
        flags.append("<|DO_NOT_CLAIM|>")
    if record.metadata.get("confidence", 1.0) < 0.5:
        flags.append("<|UNCERTAIN|>")
    prefix = " ".join(flags)
    target_token = "<|ANSWER_PLAN|>" if record.task.value == "ANSWER_PLANNING" else "<|ANSWER|>"
    parts = [
        f"<|QUERY|>\n{record.input}",
        f"<|INTENT|>\n{intent}",
        f"<|DOMAIN|>\n{domains}",
        f"<|SOURCE|>\n{source_refs}",
        f"<|CONTEXT|>\n{prefix}".rstrip(),
        f"{target_token}\n{record.target}",
        "<|END|>",
    ]
    return "\n".join(parts)
