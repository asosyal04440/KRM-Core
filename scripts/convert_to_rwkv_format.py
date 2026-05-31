"""
echOS Spec->Code verisini RWKV eğitim formatına dönüştürür.

RWKV eğitim formatı:
- Düz metin dosyası (.txt)
- Her satır bir eğitim örneği
- Format: "User: {soru}\nAssistant: {cevap}\n\n"

Ayrıca tokenized binary format da oluşturur (RWKV için optimal).
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = Path(r"D:\yeni_ai_hiyerarsisi\krm_core\data\training_corpus\echos_spec_code_pairs.jsonl")
OUTPUT_DIR = Path(r"D:\yeni_ai_hiyerarsisi\krm_core\data\rwkv_training")

# Maksimum token sayısı (yaklaşık)
MAX_TOKENS_PER_EXAMPLE = 2000


def estimate_tokens(text: str) -> int:
    """Yaklaşık token sayısını hesapla (Türkçe için ~4 karakter/token)."""
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Metni belirli token sayısına kısalt."""
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def format_as_chat(example: Dict[str, Any]) -> str:
    """Örneği RWKV chat formatına dönüştür."""
    inp = example["input"]
    target = example["target"]

    # Spec context ve requirement'ı birleştir
    spec_ctx = inp.get("spec_context", "")
    requirement = inp.get("requirement", "")
    code = target.get("code", "")
    doc_summary = target.get("doc_summary", [])

    # Kodu kısalt
    code = truncate_to_tokens(code, MAX_TOKENS_PER_EXAMPLE // 2)

    # Dokümantasyon özeti
    doc_text = "\n".join(doc_summary[:3]) if doc_summary else ""

    # Chat formatı
    chat = f"""User: {spec_ctx}

{requirement}

Assistant: echOS kaynak kodunda bu gereksinim şöyle uygulanmıştır:

```rust
{code}
```

{doc_text}

"""
    return chat


def format_as_completion(example: Dict[str, Any]) -> str:
    """Örneği RWKV completion formatına dönüştür."""
    inp = example["input"]
    target = example["target"]

    spec_ctx = inp.get("spec_context", "")
    requirement = inp.get("requirement", "")
    code = truncate_to_tokens(target.get("code", ""), MAX_TOKENS_PER_EXAMPLE // 2)

    # Daha kısa ve öz format
    completion = f"### Spec: {spec_ctx}\n\n### Görev: {requirement}\n\n### Kod:\n```rust\n{code}\n```\n\n"
    return completion


def main():
    """Ana fonksiyon."""
    print("echOS Training Data -> RWKV Format Converter")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verileri oku
    examples = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")

    # Chat formatında kaydet
    chat_file = OUTPUT_DIR / "echos_chat_train.txt"
    with open(chat_file, "w", encoding="utf-8") as f:
        for example in examples:
            chat = format_as_chat(example)
            f.write(chat)
            f.write("\n" + "=" * 80 + "\n\n")

    print(f"Chat format: {chat_file}")

    # Completion formatında kaydet
    completion_file = OUTPUT_DIR / "echos_completion_train.txt"
    with open(completion_file, "w", encoding="utf-8") as f:
        for example in examples:
            completion = format_as_completion(example)
            f.write(completion)
            f.write("\n")

    print(f"Completion format: {completion_file}")

    # Validation split (son %10)
    split_idx = int(len(examples) * 0.9)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    # Validation dosyası
    val_file = OUTPUT_DIR / "echos_completion_valid.txt"
    with open(val_file, "w", encoding="utf-8") as f:
        for example in val_examples:
            completion = format_as_completion(example)
            f.write(completion)
            f.write("\n")

    print(f"Validation: {val_file} ({len(val_examples)} examples)")

    # İstatistikler
    total_chars = 0
    total_tokens_est = 0
    for example in examples:
        chat = format_as_chat(example)
        total_chars += len(chat)
        total_tokens_est += estimate_tokens(chat)

    print(f"\n{'=' * 60}")
    print(f"Total characters: {total_chars:,}")
    print(f"Estimated tokens: {total_tokens_est:,}")
    print(f"Average tokens/example: {total_tokens_est // len(examples):,}")

    # Küçük bir sample oluştur
    sample_file = OUTPUT_DIR / "echos_sample.txt"
    with open(sample_file, "w", encoding="utf-8") as f:
        for example in examples[:3]:
            chat = format_as_chat(example)
            f.write(chat)
            f.write("\n" + "=" * 80 + "\n\n")

    print(f"\nSample file: {sample_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
