from krm.training.tokenizer import CharByteTokenizer


def test_tokenizer_round_trip_special_tokens_and_turkish(tmp_path) -> None:
    tokenizer = CharByteTokenizer()
    text = "<|QUERY|> kavram ağı çalışıyor mu? İstanbul <|END|>"
    ids = tokenizer.encode(text)
    assert tokenizer.token_to_id["<|QUERY|>"] in ids
    assert tokenizer.decode(ids) == text
    tokenizer.save(tmp_path / "tok")
    loaded = CharByteTokenizer.load(tmp_path / "tok")
    assert loaded.decode(loaded.encode(text)) == text
    assert loaded.vocab_size == tokenizer.vocab_size
