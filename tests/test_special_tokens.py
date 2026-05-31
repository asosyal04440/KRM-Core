from krm.training.record_format import TrainingTaskType, make_record
from krm.training.special_tokens import format_training_prompt, list_special_tokens, validate_no_collision
from krm.training.tokenizer import CharByteTokenizer


def test_special_tokens_unique_and_formatting_preserved() -> None:
    validate_no_collision()
    tokens = list_special_tokens()
    assert len(tokens) == len(set(tokens))
    record = make_record(TrainingTaskType.ANSWER_PLANNING, "q", "plan", {"confidence": 0.4, "domains": ["history"]})
    prompt = format_training_prompt(record)
    assert "<|QUERY|>" in prompt
    assert "<|ANSWER_PLAN|>" in prompt
    assert "<|UNCERTAIN|>" in prompt
    tokenizer = CharByteTokenizer()
    ids = tokenizer.encode(prompt)
    assert tokenizer.token_to_id["<|ANSWER_PLAN|>"] in ids
    assert tokenizer.decode(ids) == prompt
