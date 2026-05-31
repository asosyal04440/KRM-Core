"""
Bellek Haritali Egitim Pipeline'i

Dusuk donanimda buyuk modelleri egitmek icin disk'i genisletilmis bellek olarak kullanir.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    Dataset = None
    TORCH_AVAILABLE = False


@dataclass
class MemoryBudget:
    """Butce yonetimi (RAM/VRAM)."""

    max_ram_mb: int = 2000
    max_vram_mb: int = 6000

    current_ram_mb: float = 0
    current_vram_mb: float = 0

    def can_allocate(self, size_mb: float, target: str = "ram") -> bool:
        if target == "ram":
            return self.current_ram_mb + size_mb <= self.max_ram_mb
        elif target == "vram":
            return self.current_vram_mb + size_mb <= self.max_vram_mb
        return False

    def allocate(self, size_mb: float, target: str = "ram") -> None:
        if target == "ram":
            self.current_ram_mb += size_mb
        elif target == "vram":
            self.current_vram_mb += size_mb

    def free(self, size_mb: float, target: str = "ram") -> None:
        if target == "ram":
            self.current_ram_mb = max(0, self.current_ram_mb - size_mb)
        elif target == "vram":
            self.current_vram_mb = max(0, self.current_vram_mb - size_mb)

    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            "ram_used_mb": self.current_ram_mb,
            "ram_max_mb": self.max_ram_mb,
            "ram_usage_percent": (self.current_ram_mb / self.max_ram_mb) * 100,
            "vram_used_mb": self.current_vram_mb,
            "vram_max_mb": self.max_vram_mb,
            "vram_usage_percent": (self.current_vram_mb / self.max_vram_mb) * 100,
        }


class DiskBuffer:
    """Disk tabanli tampon bellek."""

    def __init__(self, buffer_dir: str, max_size_mb: int = 10000):
        self.buffer_dir = Path(buffer_dir)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.current_size_mb = 0

    def save_tensor(self, tensor: Any, name: str) -> str:
        path = self.buffer_dir / f"{name}.pt"
        if TORCH_AVAILABLE and torch is not None:
            torch.save(tensor, path)
            size_mb = tensor.numel() * tensor.element_size() / 1024 / 1024
        else:
            import pickle
            with open(str(path) + ".pkl", "wb") as f:
                pickle.dump(tensor, f)
            size_mb = os.path.getsize(str(path) + ".pkl") / 1024 / 1024
        self.current_size_mb += size_mb
        return str(path)

    def load_tensor(self, name: str) -> Any:
        pkl_path = self.buffer_dir / f"{name}.pt.pkl"
        pt_path = self.buffer_dir / f"{name}.pt"
        if TORCH_AVAILABLE and torch is not None and pt_path.exists():
            return torch.load(pt_path, weights_only=False)
        elif pkl_path.exists():
            import pickle
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        raise FileNotFoundError(f"Tensor '{name}' not found")

    def delete_tensor(self, name: str) -> None:
        for suffix in [".pt", ".pt.pkl"]:
            path = self.buffer_dir / f"{name}{suffix}"
            if path.exists():
                path.unlink()

    def clear(self) -> None:
        if self.buffer_dir.exists():
            shutil.rmtree(self.buffer_dir)
            self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.current_size_mb = 0


class ConceptualTrainingOrchestrator:
    """
    Kavramsal Parcali Egitim Orkestrasyonu.

    Kavram bloklarini sirayla yukler, egitir, kaydeder ve cikar.
    Tum islemler butce kontrolu altinda yapilir.
    """

    def __init__(
        self,
        output_dir: str,
        max_ram_mb: int = 2000,
        max_vram_mb: int = 6000,
        buffer_dir: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        self.budget = MemoryBudget(max_ram_mb=max_ram_mb, max_vram_mb=max_vram_mb)
        self.buffer = DiskBuffer(
            buffer_dir=buffer_dir or str(self.output_dir / "buffer"),
        )

        self.current_block_id: Optional[str] = None
        self.training_log: List[Dict[str, Any]] = []

    def estimate_block_memory(self, param_count: int) -> Dict[str, float]:
        """Tek blok icin bellek tahmini (MB)."""
        fp32_model = param_count * 4 / 1024 / 1024
        fp16_model = param_count * 2 / 1024 / 1024
        gradient = fp32_model
        optimizer = fp32_model * 2
        activations = fp32_model * 0.5

        return {
            "model_fp32_mb": fp32_model,
            "model_fp16_mb": fp16_model,
            "gradient_mb": gradient,
            "optimizer_mb": optimizer,
            "activations_mb": activations,
            "total_training_mb": fp16_model + gradient + optimizer + activations,
        }

    def can_train_block(self, param_count: int) -> Tuple[bool, Dict[str, float]]:
        """Bir blok egitilebilir mi kontrol et."""
        mem = self.estimate_block_memory(param_count)
        total_needed = mem["total_training_mb"]

        fits_ram = self.budget.can_allocate(total_needed, "ram")

        if TORCH_AVAILABLE and torch.cuda.is_available():
            fits_vram = self.budget.can_allocate(mem["model_fp16_mb"], "vram")
        else:
            fits_vram = True

        return fits_ram and fits_vram, mem

    def load_block_from_disk(self, block_path: str) -> Dict[str, Any]:
        """Blok parametrelerini diskten yukle."""
        path = Path(block_path)
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif TORCH_AVAILABLE and torch is not None and path.suffix == ".pt":
            return torch.load(path, weights_only=False)
        elif path.suffix == ".pkl" or str(path).endswith(".pt.pkl"):
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Desteklenmeyen format: {path.suffix}")

    def save_block_checkpoint(
        self,
        block_id: str,
        state_dict: Any,
        metadata: Dict[str, Any],
    ) -> str:
        """Blok checkpoint'ini kaydet."""
        checkpoint_path = self.output_dir / "checkpoints" / f"{block_id}.pt"

        checkpoint = {
            "block_id": block_id,
            "state_dict": state_dict,
            "metadata": metadata,
        }

        if TORCH_AVAILABLE and torch is not None:
            torch.save(checkpoint, checkpoint_path)
        else:
            import pickle
            with open(str(checkpoint_path) + ".pkl", "wb") as f:
                pickle.dump(checkpoint, f)
            checkpoint_path = Path(str(checkpoint_path) + ".pkl")

        return str(checkpoint_path)

    def log_training(self, block_id: str, result: Dict[str, Any]) -> None:
        """Egitim sonucunu kaydet."""
        entry = {
            "block_id": block_id,
            "budget_before": self.budget.get_usage_stats(),
            "result": result,
        }
        self.training_log.append(entry)

        log_path = self.output_dir / "logs" / "training_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.training_log, f, indent=2, ensure_ascii=False)

    def get_full_summary(self) -> Dict[str, Any]:
        """Tam egitim ozeti."""
        return {
            "total_blocks_trained": len(self.training_log),
            "budget_stats": self.budget.get_usage_stats(),
            "buffer_size_mb": self.buffer.current_size_mb,
            "output_dir": str(self.output_dir),
            "training_log": self.training_log,
        }

    def calculate_optimal_config(
        self,
        total_params: int,
        available_ram_mb: int = 16000,
        available_vram_mb: int = 6000,
    ) -> Dict[str, Any]:
        """
        Verilen donanim icin optimal konfigurasyonu hesapla.

        Modeli kucuk bloklara boler, her blok RAM'a sigacak sekilde.
        
        Args:
            total_params: Toplam model parametresi
            available_ram_mb: Mevcut RAM (MB)
            available_vram_mb: Mevcut VRAM (MB)

        Returns:
            Optimal konfigurasyon
        """
        print(f"\nHesaplaniyor: {total_params:,} parametre")
        print(f"  Mevcut RAM: {available_ram_mb} MB")
        print(f"  Mevcut VRAM: {available_vram_mb} MB")

        # Her byte icin gerekli bellek:
        # FP16 model: 2 byte/param
        # FP32 gradient: 4 byte/param
        # Adam optimizer (m + v): 8 byte/param
        # Activations: ~2 byte/param (tahmini)
        # Toplam: ~16 byte/param
        bytes_per_param = 16
        mb_per_param = bytes_per_param / (1024 * 1024)

        # Mevcut RAM'a sigan maksimum parametre sayisi
        available_bytes = available_ram_mb * 1024 * 1024
        max_params_per_block = int(available_bytes / bytes_per_param * 0.85)

        # VRAM kontrolu (INT4 model: 0.5 byte/param)
        vram_available = available_vram_mb * 1024 * 1024
        max_params_per_block_vram = int(vram_available / 0.5 * 0.85)

        # Ikinin kucugu
        max_params_per_block = min(max_params_per_block, max_params_per_block_vram)

        # Toplam blok sayisi
        total_blocks = max(1, -(-total_params // max_params_per_block))

        # Gercek blok boyutu
        params_per_block = total_params // total_blocks

        # Bir blok icin bellek tuketimi
        block_training_mb = params_per_block * mb_per_param

        config = {
            "total_parameters": total_params,
            "total_blocks": total_blocks,
            "params_per_block": params_per_block,
            "max_concurrent_blocks": 1,
            "bytes_per_param": bytes_per_param,
            "model_int4_mb": params_per_block * 0.5 / 1024 / 1024,
            "model_int8_mb": params_per_block * 1 / 1024 / 1024,
            "model_fp16_mb": params_per_block * 2 / 1024 / 1024,
            "training_per_block_mb": block_training_mb,
            "total_training_memory_mb": block_training_mb,
            "fits_in_ram": block_training_mb <= available_ram_mb,
            "fits_in_vram": (params_per_block * 0.5 / 1024 / 1024) <= available_vram_mb,
        }

        print(f"\nSonuclar:")
        print(f"  Toplam blok: {total_blocks}")
        print(f"  Blok basina parametre: {params_per_block:,}")
        print(f"  Egitim basina bellek: {block_training_mb:.0f} MB")
        print(f"  RAM'a sigar: {config['fits_in_ram']}")
        print(f"  VRAM'a sigar: {config['fits_in_vram']}")

        return config


def create_orchestrator(
    output_dir: str = "training_output",
    max_ram_mb: int = 2000,
    max_vram_mb: int = 6000,
) -> ConceptualTrainingOrchestrator:
    """Orkestrator olustur."""
    return ConceptualTrainingOrchestrator(
        output_dir=output_dir,
        max_ram_mb=max_ram_mb,
        max_vram_mb=max_vram_mb,
    )
