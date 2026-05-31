from krm.training.resource_estimator import estimate_by_name


def test_resource_estimator_scales_with_config() -> None:
    small = estimate_by_name("10m", batch_size=2, seq_len=128)
    bigger = estimate_by_name("30m", batch_size=2, seq_len=128)
    assert small.parameter_count > 0
    assert bigger.total_training_memory_bytes > small.total_training_memory_bytes
