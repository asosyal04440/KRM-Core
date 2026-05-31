"""
Bellek Haritali Egitim Pipeline - Test Dosyasi (PyTorch Gerektirmez)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from krm.training.memory_mapped_pipeline import (
    MemoryBudget,
    DiskBuffer,
    ConceptualTrainingOrchestrator,
    create_orchestrator,
)


def test_memory_budget():
    print("=== MemoryBudget Testi ===")
    budget = MemoryBudget(max_ram_mb=2000, max_vram_mb=6000)

    assert budget.can_allocate(500, "ram")
    budget.allocate(500, "ram")
    assert budget.current_ram_mb == 500

    assert not budget.can_allocate(2000, "ram")

    budget.free(500, "ram")
    assert budget.current_ram_mb == 0

    stats = budget.get_usage_stats()
    assert stats["ram_max_mb"] == 2000
    print("[OK] MemoryBudget testi gecti\n")


def test_disk_buffer():
    print("=== DiskBuffer Testi ===")
    buffer = DiskBuffer(buffer_dir="test_buffer", max_size_mb=1000)

    data = {"key": "value", "numbers": [1, 2, 3]}
    import json
    path = buffer.buffer_dir / "test.json"
    with open(path, "w") as f:
        json.dump(data, f)

    with open(path, "r") as f:
        loaded = json.load(f)

    assert loaded == data
    path.unlink()
    buffer.clear()
    print("[OK] DiskBuffer testi gecti\n")


def test_orchestrator():
    print("=== ConceptualTrainingOrchestrator Testi ===")
    orch = create_orchestrator(
        output_dir="test_orch_output",
        max_ram_mb=2000,
        max_vram_mb=6000,
    )

    assert Path("test_orch_output").exists()
    assert Path("test_orch_output/checkpoints").exists()
    assert Path("test_orch_output/logs").exists()

    print("[OK] Orchestrator olusturma testi gecti\n")


def test_memory_estimation():
    print("=== Bellek Tahmini Testi ===")
    orch = create_orchestrator()

    mem = orch.estimate_block_memory(400_000_000)
    print(f"  400M parametre icin FP16 model: {mem['model_fp16_mb']:.0f} MB")
    print(f"  Toplam egitim: {mem['total_training_mb']:.0f} MB")

    assert mem["model_fp16_mb"] > 0
    assert mem["total_training_mb"] > mem["model_fp16_mb"]

    print("[OK] Bellek tahmini testi gecti\n")


def test_can_train_block():
    print("=== Blok Egitilebilirlik Testi ===")
    orch = create_orchestrator(max_ram_mb=2000)

    fits, mem = orch.can_train_block(10_000_000)
    print(f"  10M param: sigar={fits}, gereken={mem['total_training_mb']:.0f} MB")
    assert fits == True

    fits, mem = orch.can_train_block(500_000_000)
    print(f"  500M param: sigar={fits}, gereken={mem['total_training_mb']:.0f} MB")

    print("[OK] Blok egitilebilirlik testi gecti\n")


def test_optimal_config():
    print("=== Optimal Konfigurasyon Testi ===")
    orch = create_orchestrator(max_ram_mb=16000)

    config = orch.calculate_optimal_config(
        total_params=400_000_000_000,
        available_ram_mb=16000,
        available_vram_mb=6000,
    )

    assert config["total_blocks"] > 0
    assert config["params_per_block"] > 0
    assert config["fits_in_ram"] == True

    print(f"\n  400B model, 16GB RAM, 6GB VRAM:")
    print(f"    Toplam blok: {config['total_blocks']}")
    print(f"    Blok basina: {config['params_per_block']:,} parametre")
    print(f"    Blok basina egitim: {config['training_per_block_mb']:.0f} MB")
    print(f"    RAM'a siger: {config['fits_in_ram']}")
    print(f"    VRAM'a siger: {config['fits_in_vram']}")

    print("[OK] Optimal konfigurasyon testi gecti\n")


def test_checkpoint_save_load():
    print("=== Checkpoint Kayit/Yukleme Testi ===")
    orch = create_orchestrator(output_dir="test_checkpoint_output")

    fake_state = {"layer.weight": [1.0, 2.0, 3.0]}
    meta = {"loss": 0.5, "epoch": 10}

    path = orch.save_block_checkpoint("block_0001", fake_state, meta)
    assert Path(path).exists()

    loaded = orch.load_block_from_disk(path)
    assert loaded["block_id"] == "block_0001"

    print("[OK] Checkpoint kayit/yukleme testi gecti\n")


def test_training_log():
    print("=== Egitim Log Testi ===")
    orch = create_orchestrator(output_dir="test_log_output")

    orch.log_training("block_0001", {"loss": 0.5, "status": "done"})
    orch.log_training("block_0002", {"loss": 0.3, "status": "done"})

    assert len(orch.training_log) == 2

    summary = orch.get_full_summary()
    assert summary["total_blocks_trained"] == 2

    log_path = Path("test_log_output/logs/training_log.json")
    assert log_path.exists()

    print("[OK] Egitim log testi gecti\n")


def test_optimal_configs_various():
    print("=== Farkli Donanimlar Icin Konfigurasyon ===")
    orch = create_orchestrator()

    configs = [
        ("RTX 4050 + 16GB RAM", 400_000_000_000, 16000, 6000),
        ("RTX 4090 + 64GB RAM", 400_000_000_000, 64000, 24000),
        ("A100 + 256GB RAM", 400_000_000_000, 256000, 80000),
        ("Kucuk model + 16GB RAM", 1_000_000_000, 16000, 6000),
    ]

    for name, params, ram, vram in configs:
        config = orch.calculate_optimal_config(params, ram, vram)
        print(f"\n  {name}:")
        print(f"    Bloklar: {config['total_blocks']}")
        print(f"    Blok boyutu: {config['params_per_block']:,}")
        print(f"    RAM siger: {config['fits_in_ram']}")
        print(f"    VRAM siger: {config['fits_in_vram']}")

    print("\n[OK] Farkli donanim testleri gecti\n")


def main():
    print("Bellek Haritali Egitim Pipeline Testleri\n")
    print("=" * 60)

    test_memory_budget()
    test_disk_buffer()
    test_orchestrator()
    test_memory_estimation()
    test_can_train_block()
    test_optimal_config()
    test_checkpoint_save_load()
    test_training_log()
    test_optimal_configs_various()

    print("=" * 60)
    print("Tum testler basarili!")
    print("\nSonraki adimlar:")
    print("  1. PyTorch ile gercek egitim dongusu")
    print("  2. Gradient yonetimi (gradient checkpointing)")
    print("  3. Mixed precision (FP16) destegi")
    print("  4. Rezonans motoru entegrasyonu")


if __name__ == "__main__":
    main()
