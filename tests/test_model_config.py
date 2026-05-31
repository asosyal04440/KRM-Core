from krm.training.model_config import TinyModelConfig, get_model_config


def test_model_configs_validate_and_save_load(tmp_path) -> None:
    cfg10 = get_model_config("10m")
    cfg30 = get_model_config("30m")
    assert cfg10.parameter_estimate > 0
    assert cfg30.parameter_estimate > cfg10.parameter_estimate
    path = tmp_path / "config.json"
    cfg10.save(path)
    loaded = TinyModelConfig.load(path)
    assert loaded.model_name == cfg10.model_name
