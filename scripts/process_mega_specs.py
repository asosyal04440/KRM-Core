"""
Intel SDM (13MB), NVMe 2.0 (1.4MB), VirtIO 1.4 (1.1MB) spec'lerini 
işler ve devasa eğitim verisi oluşturur.
"""

import json, re, hashlib
from pathlib import Path
from typing import Dict, List, Any

INTEL_SDM_MD = Path(r"D:\echOS Kaynak Arşivi\OS kaynaları\pdfs-master\01_Operating_Systems_and_Kernel\Intel SDM - 64 and IA-32 Architectures Combined Volumes 1-4 (June 2025).md")
NVME_SPEC_MD = Path(r"D:\echOS Kaynak Arşivi\OS kaynaları\pdfs-master\01_Operating_Systems_and_Kernel\NVMe Base Specification 2.0.md")
VIRTIO_SPEC_MD = Path(r"D:\echOS Kaynak Arşivi\OS kaynaları\pdfs-master\01_Operating_Systems_and_Kernel\VirtIO 1.4 - Virtual I-O Device Specification.md")
ECHOS_SRC = Path(r"C:\Users\Bahadir\Desktop\dersler_ve_projeler\echOS\src")
OUTPUT_DIR = Path(r"D:\yeni_ai_hiyerarsisi\krm_core\data\training_corpus")
SPECS_DIR = Path(r"D:\yeni_ai_hiyerarsisi\krm_core\data\echos_specs")

def read_file(path: Path, max_mb: int = 20) -> str:
    try:
        if path.exists() and path.stat().st_size < max_mb * 1_000_000:
            return path.read_text(encoding="utf-8", errors="replace")
    except: pass
    return ""

def get_echos_code(keywords: list) -> dict:
    """echOS kodundan ilgili dosyaları oku."""
    result = {}
    for root, _, files in os.walk(ECHOS_SRC):
        for f in files:
            if f.endswith(".rs"):
                path = Path(root) / f
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    for kw in keywords:
                        if kw.lower() in text.lower():
                            rel = path.relative_to(ECHOS_SRC)
                            result[str(rel)] = text[:3000]
                            break
                except: pass
    return result

import os

def create_sdm_chunks(text: str) -> list:
    """SDM'i sayfa chunk'larına böl ve volume/chapter etiketi ekle."""
    chunks = []
    pages = text.split("--- Page ")

    current_vol = "Volume 1"
    current_chapter = "Intro"

    for page in pages:
        if not page.strip():
            continue

        # Volume detection
        for vol_num in range(1, 5):
            if f"Volume {vol_num}:" in page[:500]:
                current_vol = f"Volume {vol_num}"
                break

        # Chapter detection
        cm = re.search(r'CHAPTER\s+(\d+)\s*\n(.{1,100})', page[:2000])
        if cm:
            current_chapter = f"CHAPTER {cm.group(1)}: {cm.group(2).strip()}"

        # Instruction section detection
        im = re.search(r'^(INSTRUCTION\s+SET\s+REFERENCE|INSTRUCTIONS)\s*$', page[:500], re.MULTILINE | re.IGNORECASE)
        if im:
            current_vol = "Volume 2"
            current_chapter = im.group(1)

        # Extract clean text
        clean = re.sub(r'^---.*?---$', '', page, flags=re.MULTILINE)
        clean = re.sub(r'Vol\.\s+\d+\s+(iii|iv|v|vi|vii|vii|x|xi|xii|xiii|xiv|xv)', '', clean)
        clean_lines = [l for l in clean.split("\n") if l.strip() and len(l.strip()) > 5]

        if len(clean_lines) > 3:
            text_block = "\n".join(clean_lines)
            if len(text_block) > 500:
                chunks.append({
                    "volume": current_vol,
                    "chapter": current_chapter,
                    "text": text_block[:8000]
                })

    return chunks


def process_sdm(chunks: list) -> list:
    """SDM eğitim çiftleri oluştur."""
    pairs = []
    echos_code = {}

    sdm_mappings = {
        "Volume 3": ["memory", "interrupt", "apic", "paging", "gdt", "idt", "protection", "task"],
        "Volume 1": ["basic", "architecture", "register", "instruction", "data type"],
        "Volume 2": ["instruction", "opcode"],
        "Volume 4": ["msr", "model specific"],
    }

    for vol, keywords in sdm_mappings.items():
        if vol not in echos_code:
            echos_code[vol] = get_echos_code(keywords)

    for chunk in chunks:
        vol = chunk["volume"]
        text = chunk["text"]

        # Concepts
        concepts = set()
        for m in re.finditer(r'\b(Paging|Interrupt|Descriptor|Segment|Register|APIC|MSI|TLB|Cache|Fault|Exception|Gate|Task|Call|Jump|Flag|Control|Segment|Selector|Translation|Canonical|Huge Page|PAT|MTRR|VMX|SMX|SMM|MCA)\b', text, re.IGNORECASE):
            concepts.add(m.group(1))

        # echOS kodunu bul
        code = {k: v for k, v in echos_code.get(vol, {}).items()}
        if not code:
            code = get_echos_code(["memory", "interrupt", "paging"])

        # Register definitions
        regs = []
        for m in re.finditer(r'\b(CR[0-4]|DR[0-7]|EFER|GDTR|IDTR|LDTR|TR|RFLAGS|MXCSR|XCR0)\b', text):
            if m.group(1) not in regs:
                regs.append(m.group(1))

        pair = {
            "task": "CODE_GENERATION",
            "input": {
                "spec": f"Intel SDM - {vol}",
                "chapter": chunk["chapter"],
                "context": text[:1000],
                "concepts": sorted(concepts)[:15],
                "registers": regs[:10]
            },
            "target": {"code": code},
            "metadata": {
                "spec": "Intel SDM 1-4",
                "volume": vol,
                "domain": "baremetal_os_development",
                "text_len": len(text)
            }
        }
        pairs.append(pair)

    return pairs


def process_markdown_spec(text: str, spec_name: str, section_split: str = r'\n(?=#{1,3}\s)') -> tuple:
    """Genel markdown spec işleme."""
    pairs = []
    sections = re.split(section_split, text)

    echos_keywords = {
        "NVMe": ["nvme", "pci", "dma", "storage", "ssd", "queue"],
        "VirtIO": ["virtio", "virtqueue", "pci"],
    }

    for k in echos_keywords:
        if k in spec_name:
            keywords = echos_keywords[k]
            break
    else:
        keywords = []

    code = get_echos_code(keywords) if keywords else {}

    for sec in sections[:40]:
        lines = sec.strip().split("\n")
        title = lines[0].lstrip("#").strip() if lines else ""
        body = "\n".join(lines[1:])[:2000]

        if len(body) < 100:
            continue

        concepts = set()
        for m in re.finditer(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b', sec[:500], re.UNICODE):
            word = m.group(1)
            if len(word) > 3 and word.isalpha():
                concepts.add(word)

        pairs.append({
            "task": "CODE_GENERATION",
            "input": {
                "spec": spec_name,
                "section": title,
                "summary": body,
                "concepts": sorted(concepts)[:10]
            },
            "target": {"code": code},
            "metadata": {"spec": spec_name, "domain": "baremetal_os_development"}
        })

    return pairs


def save_pairs(pairs: list, spec_name: str):
    """Çiftleri kaydet."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{spec_name}_pairs.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  Saved {len(pairs)} pairs -> {path.name}")

    total_chars = sum(len(json.dumps(p)) for p in pairs)
    print(f"  Total: {total_chars:,} chars, ~{total_chars//4:,} tokens")


def main():
    print("=" * 60)
    print("MEGA SPEC INGESTOR")
    print("=" * 60)

    all_concepts = set()

    # 1. SDM
    print("\n[1] Intel SDM (13MB)...")
    text = read_file(INTEL_SDM_MD)
    if text:
        print(f"  Read {len(text):,} chars")
        chunks = create_sdm_chunks(text)
        print(f"  Created {len(chunks)} chunks")
        pairs = process_sdm(chunks)
        save_pairs(pairs, "intel_sdm")
        all_concepts.update(c for p in pairs for c in p["input"].get("concepts", []))

    # 2. NVMe 2.0
    print("\n[2] NVMe Base Spec 2.0 (1.4MB)...")
    text = read_file(NVME_SPEC_MD)
    if text:
        print(f"  Read {len(text):,} chars")
        pairs = process_markdown_spec(text, "NVMe 2.0")
        save_pairs(pairs, "nvme_2.0")
        all_concepts.update(c for p in pairs for c in p["input"].get("concepts", []))

    # 3. VirtIO 1.4
    print("\n[3] VirtIO 1.4 (1.1MB)...")
    text = read_file(VIRTIO_SPEC_MD)
    if text:
        print(f"  Read {len(text):,} chars")
        pairs = process_markdown_spec(text, "VirtIO 1.4")
        save_pairs(pairs, "virtio_1.4")
        all_concepts.update(c for p in pairs for c in p["input"].get("concepts", []))

    # Özet
    print(f"\n{'=' * 60}")
    print(f"Unique concepts: {len(all_concepts)}")
    print(f"Concepts: {sorted(all_concepts)[:30]}...")
    print("DONE!")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
