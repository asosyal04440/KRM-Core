"""
echOS Spec->Code Training Data Builder

Bu script, echOS kaynak kodunu ve donanım spec'lerini okuyarak
egitim çiftleri (spec->code pairs) olusturur.

Egitim formati:
{
    "task": "CODE_GENERATION",
    "input": "spec_context + requirement",
    "target": "echOS Rust kodu",
    "metadata": {
        "spec_name": "NVMe Base Spec 1.4",
        "source_file": "src/drivers/nvme.rs",
        "concepts": ["NVMe", "PCIe", "DMA", "Submission Queue"]
    }
}
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib

# ============================================================================
# CONFIGURATION
# ============================================================================

ECHOS_ROOT = Path(r"C:\Users\Bahadir\Desktop\dersler_ve_projeler\echOS")
ECHOS_SRC = ECHOS_ROOT / "src"
print(f"echOS source: {ECHOS_SRC} (exists: {ECHOS_SRC.exists()})")
SPECS_DIR = Path(r"D:\yeni_ai_hiyerarsisi\krm_core\data\echos_specs")
OUTPUT_DIR = Path(r"D:\yeni_ai_hiyerarsisi\krm_core\data\training_corpus")

# Donanım spec eşleştirmeleri: spec_name -> (source_path_patterns, key_concepts)
SPEC_MAPPING = {
    "NVMe Base Spec 1.4": {
        "patterns": ["nvme", "non-volatile"],
        "source_files": ["drivers/nvme.rs"],
        "concepts": ["NVMe", "PCIe", "DMA", "Submission Queue", "Completion Queue",
                     "Admin Queue", "I/O Queue", "Namespace", "Controller", "MMIO"],
        "register_map": {
            "NVME_CAP": "Controller Capabilities",
            "NVME_CC": "Controller Configuration",
            "NVME_CSTS": "Controller Status",
            "NVME_AQA": "Admin Queue Attributes",
            "NVME_ASQ": "Admin Submission Queue Base",
            "NVME_ACQ": "Admin Completion Queue Base"
        }
    },
    "VirtIO Spec 1.2": {
        "patterns": ["virtio", "virtqueue", "virtio_blk"],
        "source_files": ["drivers/virtio_blk.rs", "drivers/virtio_net.rs",
                         "drivers/virtio_ffi.rs", "drivers/virtio_gpu.rs"],
        "concepts": ["VirtIO", "Virtqueue", "Descriptor", "Available Ring",
                     "Used Ring", "Feature Negotiation", "MMIO", "PCI Transport"],
        "structures": ["BlkReq", "BlkResp", "VirtIOBlk"]
    },
    "AHCI Spec 1.3.1": {
        "patterns": ["ahci", "sata", "ich10"],
        "source_files": ["drivers/ahci.rs"],
        "concepts": ["AHCI", "SATA", "HBA", "Command List", "FIS",
                     "PRDT", "Port Multiplier", "NCQ"],
        "register_map": {
            "AHCI_GHC": "Global HBA Control",
            "AHCI_IS": "Interrupt Status",
            "AHCI_PI": "Ports Implemented",
            "PORT_CLB": "Command List Base",
            "PORT_FB": "FIS Base",
            "PORT_CMD": "Command and Status"
        }
    },
    "Intel SDM Vol 3A": {
        "patterns": ["intel sdm", "paging", "interrupt", "gdt", "idt"],
        "source_files": ["memory/paging.rs", "memory/mod.rs",
                         "interrupts/idt.rs", "interrupts/mod.rs"],
        "concepts": ["Paging", "Page Table", "CR3", "GDT", "IDT",
                     "Interrupt Descriptor", "Segment Selector", "TSS"],
        "registers": ["CR0", "CR3", "GDTR", "IDTR"]
    },
    "ACPI Spec 6.4": {
        "patterns": ["acpi", "rsdp", "madt", "aml"],
        "source_files": ["acpi/mod.rs", "acpi/madt.rs", "acpi/aml.rs"],
        "concepts": ["ACPI", "RSDP", "RSDT", "XSDT", "MADT", "AML",
                     "DSDT", "SSDT", "GPE", "Power Management"]
    },
    "Intel APIC Spec": {
        "patterns": ["apic", "ioapic", "lapic"],
        "source_files": ["apic/lapic.rs", "apic/ioapic.rs", "apic/mod.rs"],
        "concepts": ["Local APIC", "I/O APIC", "MSI", "MSI-X",
                     "Interrupt Vector", "EOI", "ICR", "TPR"]
    },
    "PCI Local Bus Spec 3.0": {
        "patterns": ["pci", "pci express"],
        "source_files": ["drivers/pci.rs", "drivers/pci_root.rs"],
        "concepts": ["PCI", "Configuration Space", "BAR", "MSI",
                     "Interrupt Line", "Device ID", "Vendor ID"]
    },
    "F2FS Spec": {
        "patterns": ["f2fs", "flash-friendly"],
        "source_files": ["fs/f2fs.rs"],
        "concepts": ["F2FS", "Node", "Segment", "Section", "Zone",
                     "Checkpoint", "NAT", "SSA", "Superblock"]
    },
    "ext4 Spec": {
        "patterns": ["ext4", "ext2", "ext3"],
        "source_files": ["fs/ext4.rs", "fs/ext4_journal.rs"],
        "concepts": ["ext4", "Inode", "Block Group", "Journal",
                     "JBD2", "Extent Tree", "Superblock"]
    },
    "IEEE 802.3 Ethernet": {
        "patterns": ["ethernet", "802.3", "mac"],
        "source_files": ["net/ethernet.rs"],
        "concepts": ["Ethernet", "Frame", "MAC Address", "EtherType",
                     "CRC32", "Preamble", "SFD"]
    },
    "RFC 793 TCP": {
        "patterns": ["tcp", "rfc 793"],
        "source_files": ["net/tcp.rs"],
        "concepts": ["TCP", "Connection", "SYN", "ACK", "FIN",
                     "Sequence Number", "Window", "Checksum"]
    },
    "xHCI Spec 1.2": {
        "patterns": ["xhci", "usb 3"],
        "source_files": ["drivers/usb/mod.rs"],
        "concepts": ["xHCI", "USB", "Endpoint", "Transfer Ring",
                     "Event Ring", "Command Ring", "TRB"]
    }
}


def read_source_file(rel_path: str) -> Optional[str]:
    """echOS kaynak kodunu oku."""
    full_path = ECHOS_SRC / rel_path
    if full_path.exists():
        try:
            return full_path.read_text(encoding="utf-8", errors="replace")
        except:
            return None
    return None


def read_spec_content(spec_name: str) -> Optional[str]:
    """Spec content'ini JSON dosyasından oku."""
    for spec_file in SPECS_DIR.glob("*.json"):
        try:
            data = json.loads(spec_file.read_text(encoding="utf-8"))
            if data.get("name", "").lower() in spec_name.lower() or \
               spec_name.lower() in data.get("name", "").lower():
                return data.get("content", "")
        except:
            continue
    return None


def extract_code_sections(source_code: str, max_sections: int = 10) -> List[Dict[str, str]]:
    """Kaynak kodundan önemli bölümleri çıkar."""
    sections = []
    lines = source_code.split("\n")

    # Struct, impl, fn tanımlarını bul
    current_section = None
    current_lines = []
    brace_depth = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Yeni section başlangıcı
        if re.match(r"^(pub\s+)?(struct|impl|fn|const)\s+", stripped):
            if current_section and current_lines:
                sections.append({
                    "type": current_section,
                    "code": "\n".join(current_lines[:100])  # İlk 100 satır
                })
            current_section = re.match(r"^(pub\s+)?(struct|impl|fn|const)\s+(\w+)", stripped).group(1) or \
                              re.match(r"^(pub\s+)?(struct|impl|fn|const)\s+(\w+)", stripped).group(2)
            current_lines = [line]
            brace_depth = 0
        elif current_section:
            current_lines.append(line)
            brace_depth += stripped.count("{") - stripped.count("}")

            if brace_depth <= 0 and len(current_lines) > 5:
                sections.append({
                    "type": current_section,
                    "code": "\n".join(current_lines[:100])
                })
                current_section = None
                current_lines = []

    return sections[:max_sections]


def extract_doc_comments(source_code: str) -> List[str]:
    """Dokümantasyon yorumlarını çıkar."""
    comments = []
    in_doc_comment = False
    current_comment = []

    for line in source_code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("//!"):
            in_doc_comment = True
            current_comment.append(stripped[3:].strip())
        elif in_doc_comment:
            if stripped.startswith("//!"):
                current_comment.append(stripped[3:].strip())
            else:
                if current_comment:
                    comments.append("\n".join(current_comment))
                current_comment = []
                in_doc_comment = False

    if current_comment:
        comments.append("\n".join(current_comment))

    return comments


def create_training_pair(
    spec_name: str,
    spec_info: Dict,
    source_file: str,
    source_code: str,
    requirement: str
) -> Dict[str, Any]:
    """Tek bir eğitim çifti oluştur."""

    # Kod bölümlerini çıkar
    code_sections = extract_code_sections(source_code)
    doc_comments = extract_doc_comments(source_code)

    # Spec content'ini al
    spec_content = read_spec_content(spec_name)

    # Eğitim çifti oluştur
    training_pair = {
        "task": "CODE_GENERATION",
        "input": {
            "spec_context": f"Spec: {spec_name}\n\nKey concepts: {', '.join(spec_info['concepts'])}",
            "requirement": requirement,
            "relevant_code_sections": [s["code"][:500] for s in code_sections[:3]]
        },
        "target": {
            "code": source_code[:3000],  # İlk 3000 karakter
            "doc_summary": doc_comments[:3] if doc_comments else [],
            "key_structures": [s["type"] for s in code_sections[:5]]
        },
        "metadata": {
            "spec_name": spec_name,
            "source_file": source_file,
            "concepts": spec_info["concepts"],
            "register_map": spec_info.get("register_map", {}),
            "difficulty": "advanced",
            "domain": "baremetal_os_development"
        }
    }

    return training_pair


def generate_requirements(spec_name: str, spec_info: Dict) -> List[str]:
    """Spec'e göre gereksinim listesi oluştur."""
    requirements = []

    if "NVMe" in spec_name:
        requirements = [
            "NVMe denetleyicisini başlat (Controller Configuration register'ını ayarla)",
            "Admin kuyruğu oluştur (AQA, ASQ, ACQ register'ları)",
            "I/O kuyruğu oluştur ve okuma komutu gönder",
            "Namespace bilgisini al (Identify komutu)",
            "DMA buffer yönetimi ve PRP listesi oluştur",
            "NVMe hata yönetimini uygula (CSTS register'ı)",
        ]
    elif "VirtIO" in spec_name:
        requirements = [
            "VirtIO blok aygıt sürücüsü başlat (feature negotiation)",
            "Virtqueue oluştur ve DMA tamponu ayır",
            "Sektör okuma isteği gönder (non-blocking)",
            "Used ring'den tamamlanma bekle",
            "IOMMU domain yönetimi ile DMA yalıtımı",
        ]
    elif "AHCI" in spec_name:
        requirements = [
            "AHCI HBA'yı başlat (GHC.HR ile reset)",
            "Port command list ve FIS buffer oluştur",
            "IDENTIFY DEVICE komutu gönder",
            "DMA okuma/yazma komutları uygula",
            "FIS tabanlı kesme yönetimi",
        ]
    elif "Intel SDM" in spec_name:
        requirements = [
            "4-seviyeli sayfa tablosu kurulumu (PML4, PDPT, PD, PT)",
            "IDT (Interrupt Descriptor Table) oluştur",
            "GDT (Global Descriptor Table) ve TSS kur",
            "CR3 register'ını ayarla ve paging etkinleştir",
            "Sayfa hatası (#PF) işleyicisi yaz",
        ]
    elif "ACPI" in spec_name:
        requirements = [
            "RSDP tablosunu bul (UEFI config tablosu veya fiziksel bellek)",
            "RSDT/XSDT tablolarını ayrıştır",
            "MADT tablosunu oku (I/O APIC, Local APIC)",
            "AML yorumlayıcı başlat (DSDT/SSDT)",
            "GPE (General Purpose Events) yönetimi",
        ]
    elif "APIC" in spec_name:
        requirements = [
            "Local APIC'ı başlat (IA32_APIC_BASE MSR)",
            "I/O APIC redirection tablosunu ayarla",
            "MSI kesmesi yapılandır",
            "IPI (Inter-Processor Interrupt) gönder",
            "EOI (End of Interrupt) yönetimi",
        ]
    elif "PCI" in spec_name:
        requirements = [
            "PCI konfigürasyon uzayını tara",
            "BAR (Base Address Register) boyutunu tespit et",
            "MSI kesmesini yapılandır",
            "PCI cihaz sınıfını belirle",
            "ECAM (Enhanced Configuration Access Mechanism) kullan",
        ]
    elif "F2FS" in spec_name:
        requirements = [
            "F2FS superblock okuma ve doğrulama",
            "Node tablosunu oku (NAT)",
            "Segment yönetimi (SSA, Section, Zone)",
            "Checkpoint mekanizması uygula",
            "Dosya okuma/yazma işlemleri",
        ]
    elif "ext4" in spec_name:
        requirements = [
            "ext4 superblock okuma",
            "Block Group Descriptor tablosunu oku",
            "Inode yapısını ayrıştır",
            "JBD2 journal replay uygula",
            "Extent tree ile dosya bloğu haritalaması",
        ]
    elif "TCP" in spec_name:
        requirements = [
            "TCP connection kurulumu (three-way handshake)",
            "Sequence number yönetimi",
            "Flow control (window size)",
            "Error detection (checksum)",
            "Connection kapatma (FIN)",
        ]
    elif "Ethernet" in spec_name:
        requirements = [
            "Ethernet çerçeve oluştur ve ayrıştır",
            "MAC adresi yönetimi",
            "EtherType ayrıştırma",
            "CRC32 hesaplama",
            "Preamble ve SFD doğrulama",
        ]
    elif "xHCI" in spec_name:
        requirements = [
            "xHCI denetleyicisini başlat",
            "Transfer Ring oluştur",
            "Event Ring ve Interrupter ayarla",
            "USB cihaz keşfi (Device Context)",
            "Bulk transfer gönder/al",
        ]

    return requirements


def main():
    """Ana fonksiyon."""
    print("echOS Spec->Code Training Data Builder")
    print("=" * 60)

    all_training_pairs = []

    for spec_name, spec_info in SPEC_MAPPING.items():
        print(f"\nProcessing: {spec_name}")

        for source_file in spec_info["source_files"]:
            source_code = read_source_file(source_file)
            if not source_code:
                print(f"  [SKIP] {source_file} not found")
                continue

            print(f"  [OK] {source_file} ({len(source_code)} bytes)")

            # Her spec için gereksinimleri oluştur
            requirements = generate_requirements(spec_name, spec_info)

            for req in requirements:
                pair = create_training_pair(
                    spec_name, spec_info, source_file, source_code, req
                )
                all_training_pairs.append(pair)

    # Eğitim çiftlerini kaydet
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSONL formatında kaydet
    output_file = OUTPUT_DIR / "echos_spec_code_pairs.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for pair in all_training_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Total training pairs: {len(all_training_pairs)}")
    print(f"Output file: {output_file}")

    # Özet istatistikler
    spec_counts = {}
    for pair in all_training_pairs:
        spec = pair["metadata"]["spec_name"]
        spec_counts[spec] = spec_counts.get(spec, 0) + 1

    print("\nPer-spec counts:")
    for spec, count in sorted(spec_counts.items()):
        print(f"  {spec}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
