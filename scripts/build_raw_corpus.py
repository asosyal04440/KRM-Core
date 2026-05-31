"""
Tüm spec'leri RAW metin corpus'una dönüştürür.
RWKV modeli düz metin over training yapar.
Format: her dosya, her bölüm ayrı bir eğitim örneği.
"""

import json, re, os
from pathlib import Path

SDM_MD = Path(r"D:\echOS Kaynak Arşivi\OS kaynaları\pdfs-master\01_Operating_Systems_and_Kernel\Intel SDM - 64 and IA-32 Architectures Combined Volumes 1-4 (June 2025).md")
NVME_MD = Path(r"D:\echOS Kaynak Arşivi\OS kaynaları\pdfs-master\01_Operating_Systems_and_Kernel\NVMe Base Specification 2.0.md")
VIRTIO_MD = Path(r"D:\echOS Kaynak Arşivi\OS kaynaları\pdfs-master\01_Operating_Systems_and_Kernel\VirtIO 1.4 - Virtual I-O Device Specification.md")
ECHOS_SRC = Path(r"C:\Users\Bahadir\Desktop\dersler_ve_projeler\echOS\src")
OUTPUT = Path(r"D:\yeni_ai_hiyerarsisi\krm_core\data\rwkv_training")

def read(path, mb=25):
    try:
        if path.exists() and path.stat().st_size < mb*1e6:
            return path.read_text("utf-8", errors="replace")
    except: pass
    return ""

def clean_spec(text):
    """HTML entities ve gürültüyü temizle."""
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'[�]', '', text)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text

def extract_chapters(text, pattern=r'--- Page \d+ ---'):
    """SDM'i page/chapter chunk'larına ayır."""
    chunks = []
    pages = re.split(pattern, text)
    for p in pages:
        p = p.strip()
        if len(p) > 500:
            # Chapter başlığını bul
            title = "SDM"
            cm = re.search(r'CHAPTER\s+(\d+)\s*\n(.{1,80})', p[:2000])
            if cm:
                title = f"CHAPTER {cm.group(1)}: {cm.group(2).strip()}"
            chunks.append(f"## {title}\n\n{p}")
    return chunks

def extract_md_sections(text, spec_name):
    """Markdown bölümlerini çıkar."""
    sections = []
    parts = re.split(r'\n(?=#{1,3}\s)', text)
    for p in parts:
        p = p.strip()
        if len(p) > 200:
            title = p.split("\n")[0].lstrip("#").strip() if p.split("\n") else spec_name
            sections.append(f"## [{spec_name}] {title}\n\n{p}")
    return sections

def collect_echos_code():
    """Tüm echOS Rust kodunu tek bir corpus olarak topla."""
    corpus = []
    total_lines = 0

    for root, _, files in os.walk(ECHOS_SRC):
        for f in sorted(files):
            if f.endswith(".rs"):
                path = Path(root) / f
                try:
                    text = path.read_text("utf-8", errors="replace")
                    lines = text.count("\n") + 1
                    total_lines += lines
                    rel = path.relative_to(ECHOS_SRC)
                    corpus.append(f"// FILE: {rel} ({lines} lines)\n{text}")
                except: pass

    return "\n\n".join(corpus), total_lines


def main():
    print("=" * 60)
    print("RAW TEXT TRAINING CORPUS BUILDER")
    print("=" * 60)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    total_chars = 0

    # ================================================================
    # 1. INTEL SDM (13MB) - Volume 3 (System Programming) öncelikli
    # ================================================================
    print("\n[1] Intel SDM...")
    sdm_text = read(SDM_MD)
    if sdm_text:
        sdm_chunks = extract_chapters(sdm_text)
        print(f"  {len(sdm_chunks)} chapters extracted")

        # Volume 3 (System Programming Guide) = en değerli kısım
        vol3_chunks = [c for c in sdm_chunks if "CHAPTER" in c[:200]]

        with open(OUTPUT / "intel_sdm.txt", "w", encoding="utf-8") as f:
            for chunk in vol3_chunks:
                f.write(clean_spec(chunk))
                f.write("\n\n---\n\n")
                total_chars += len(chunk)
        print(f"  Saved: intel_sdm.txt ({total_chars:,} chars, ~{total_chars//4:,} tokens)")

    # ================================================================
    # 2. NVMe Base Spec 2.0
    # ================================================================
    print("\n[2] NVMe Base Spec 2.0...")
    nvme_text = read(NVME_MD)
    if nvme_text:
        nvme_sections = extract_md_sections(nvme_text, "NVMe 2.0")
        print(f"  {len(nvme_sections)} sections")
        nvme_text_clean = clean_spec("\n\n".join(nvme_sections))

        with open(OUTPUT / "nvme_2.0.txt", "w", encoding="utf-8") as f:
            f.write(nvme_text_clean)
        nvme_chars = len(nvme_text_clean)
        total_chars += nvme_chars
        print(f"  Saved: nvme_2.0.txt ({nvme_chars:,} chars, ~{nvme_chars//4:,} tokens)")

    # ================================================================
    # 3. VirtIO 1.4
    # ================================================================
    print("\n[3] VirtIO 1.4...")
    virtio_text = read(VIRTIO_MD)
    if virtio_text:
        virtio_sections = extract_md_sections(virtio_text, "VirtIO 1.4")
        print(f"  {len(virtio_sections)} sections")
        virtio_text_clean = clean_spec("\n\n".join(virtio_sections))

        with open(OUTPUT / "virtio_1.4.txt", "w", encoding="utf-8") as f:
            f.write(virtio_text_clean)
        virtio_chars = len(virtio_text_clean)
        total_chars += virtio_chars
        print(f"  Saved: virtio_1.4.txt ({virtio_chars:,} chars, ~{virtio_chars//4:,} tokens)")

    # ================================================================
    # 4. echOS Source Code
    # ================================================================
    print("\n[4] echOS Source Code...")
    code, total_lines = collect_echos_code()
    if code:
        safe_name = "echos_source_code.txt"
        with open(OUTPUT / safe_name, "w", encoding="utf-8") as f:
            f.write(code)
        code_chars = len(code)
        total_chars += code_chars
        print(f"  {total_lines} lines across all files")
        print(f"  Saved: {safe_name} ({code_chars:,} chars, ~{code_chars//4:,} tokens)")

    # ================================================================
    # 5. SINGLE MEGA CORPUS (hepsi birleşik)
    # ================================================================
    print("\n[5] Mega corpus (all combined)...")
    mega_path = OUTPUT / "echos_mega_corpus.txt"
    with open(mega_path, "w", encoding="utf-8") as f:
        f.write("# echOS Training Mega Corpus\n")
        f.write("# Intel SDM Volumes 1-4 (June 2025)\n")
        f.write("# NVMe Base Spec 2.0\n")
        f.write("# VirtIO 1.4\n")
        f.write("# echOS Source Code\n\n")

        if sdm_text:
            f.write("# === INTEL SDM ===\n#\n")
            for chunk in vol3_chunks:
                f.write(clean_spec(chunk))
                f.write("\n\n---\n\n")

        if nvme_text:
            f.write("# === NVMe 2.0 ===\n#\n")
            f.write(clean_spec(nvme_text_clean))
            f.write("\n\n---\n\n")

        if virtio_text:
            f.write("# === VirtIO 1.4 ===\n#\n")
            f.write(virtio_text_clean)
            f.write("\n\n---\n\n")

        if code:
            f.write("# === echOS SOURCE CODE ===\n#\n")
            f.write(code)

    mega_size = mega_path.stat().st_size
    print(f"  Saved: echos_mega_corpus.txt ({mega_size:,} bytes)")
    print(f"  Total chars: {total_chars:,}")
    print(f"  Est. tokens: {total_chars//4:,}")

    # Özet
    print(f"\n{'=' * 60}")
    print("FILES CREATED:")
    for f in sorted(OUTPUT.glob("*.txt")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:,.0f} KB")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
