"""
echOS Spec Kavram Çıkarıcı

KRM-Core'un kavram çıkarma sistemini kullanarak
spec'lerden donanım ve sistem kavramlarını çıkarır.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set

# KRM-Core paths
KRM_ROOT = Path(__file__).parent.parent
KRM_DATA_DIR = KRM_ROOT / "data" / "echos_specs"
KRM_CONCEPTS_DIR = KRM_ROOT / "data" / "echos_concepts"


class HardwareConceptExtractor:
    """
    Donanım spec'lerinden kavram çıkaran sınıf.
    """
    
    def __init__(self):
        # Donanım kavram sözlükleri
        self.register_patterns = {
            # Register isimleri
            "register": r"(?:register|reg|REG)\s*(?:_?\w+)*",
            "offset": r"(?:offset|OFFSET|OFF)\s*(?:0x[0-9a-fA-F]+|\d+)",
            "field": r"(?:field|bit|FIELD|BIT)\s*(?:_?\w+)*",
        }
        
        # Donanım terimleri
        self.hardware_terms = {
            "pci": ["PCI", "VID", "DID", "BAR", "MSI", "MSI-X", "config space"],
            "nvme": ["NVMe", "admin queue", "submission queue", "completion queue", "PRP", "SGL"],
            "virtio": ["VirtIO", "virtqueue", "vring", "feature negotiation"],
            "usb": ["USB", "xHCI", "endpoint", "transfer ring", "TRB"],
            "sata": ["AHCI", "SATA", "FIS", "command list", "received FIS"],
            "acpi": ["ACPI", "DSDT", "SSDT", "MADT", "HPET", "FADT"],
            "uefi": ["UEFI", "EFI", "boot services", "runtime services", "protocol"],
            "interrupt": ["APIC", "IOAPIC", "IDT", "ISR", "IRQ", "MSI"],
            "memory": ["PMM", "VMM", "page table", "TLB", "cr3", "paging"],
        }
        
        # Bit alanı desenleri
        self.bit_field_patterns = [
            r"bit\s*(\d+)(?:\s*:\s*(\d+))?",  # bit 0, bit 3:0
            r"\[(\d+):(\d+)\]",  # [3:0]
            r"0x([0-9a-fA-F]+)",  # Hex değerler
        ]
        
        # Register haritası desenleri
        self.register_map_patterns = [
            r"(?:0x[0-9a-fA-F]+)\s*[-–]\s*(?:0x[0-9a-fA-F]+)",  # Offset aralığı
            r"(?:register|offset)\s*(?:0x[0-9a-fA-F]+|\d+)",  # Register tanımları
        ]
    
    def extract_from_text(self, text: str, source_name: str) -> Dict[str, Any]:
        """
        Metinden kavramları çıkar.
        """
        concepts = {
            "registers": [],
            "bit_fields": [],
            "hardware_terms": [],
            "protocols": [],
            "data_structures": [],
        }
        
        # Register'ları çıkar
        for pattern in self.register_patterns.values():
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts["registers"].extend(matches)
        
        # Bit alanlarını çıkar
        for pattern in self.bit_field_patterns:
            matches = re.findall(pattern, text)
            concepts["bit_fields"].extend(matches)
        
        # Donanım terimlerini çıkar
        for category, terms in self.hardware_terms.items():
            for term in terms:
                if term.lower() in text.lower():
                    concepts["hardware_terms"].append({
                        "term": term,
                        "category": category,
                    })
        
        # Protokolleri çıkar
        protocol_patterns = [
            r"(?:PCIe|PCI Express)",
            r"(?:USB\s*\d+\.\d+)",
            r"(?:SATA\s*\d+)",
            r"(?:NVMe\s*\d+)",
        ]
        for pattern in protocol_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts["protocols"].extend(matches)
        
        # Veri yapılarını çıkar
        ds_patterns = [
            r"(?:struct|class|enum)\s+(\w+)",
            r"(?:typedef)\s+\w+\s+(\w+)",
        ]
        for pattern in ds_patterns:
            matches = re.findall(pattern, text)
            concepts["data_structures"].extend(matches)
        
        # Temizle
        concepts["registers"] = list(set(concepts["registers"]))[:100]
        concepts["bit_fields"] = list(set(str(b) for b in concepts["bit_fields"]))[:100]
        concepts["protocols"] = list(set(concepts["protocols"]))
        concepts["data_structures"] = list(set(concepts["data_structures"]))[:100]
        
        return concepts
    
    def extract_register_map(self, text: str) -> List[Dict[str, Any]]:
        """
        Register haritasını çıkar.
        """
        register_map = []
        
        # Register tanımlarını ara
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            # Hex offset kontrolü
            hex_match = re.search(r'0x([0-9a-fA-F]{2,8})', line)
            if hex_match:
                offset = hex_match.group(1)
                
                # Register adını ara
                name_match = re.search(r'([A-Z][A-Z0-9_]{2,})', line)
                if name_match:
                    reg_name = name_match.group(1)
                    
                    register_map.append({
                        "offset": f"0x{offset}",
                        "name": reg_name,
                        "line": i + 1,
                        "context": line.strip()[:100],
                    })
        
        return register_map
    
    def extract_bit_fields(self, text: str) -> List[Dict[str, Any]]:
        """
        Bit alanlarını çıkar.
        """
        bit_fields = []
        
        # Bit alanı desenlerini ara
        patterns = [
            (r'(?:bit|BIT)\s+(\d+)', 'single'),
            (r'(?:bit|BIT)\s+(\d+)\s*:\s*(\d+)', 'range'),
            (r'\[(\d+)\s*:\s*(\d+)\]', 'range'),
        ]
        
        for pattern, field_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                if field_type == 'single':
                    bit_fields.append({
                        "bit": int(match.group(1)),
                        "type": "single",
                    })
                else:
                    bit_fields.append({
                        "high": int(match.group(1)),
                        "low": int(match.group(2)),
                        "type": "range",
                    })
        
        return bit_fields


class SpecConceptPipeline:
    """
    Spec kavram çıkarma pipeline'ı.
    """
    
    def __init__(self, spec_dir: str | Path, output_dir: str | Path):
        self.spec_dir = Path(spec_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.extractor = HardwareConceptExtractor()
    
    def process_all_specs(self) -> Dict[str, Any]:
        """
        Tüm spec dosyalarını işle.
        """
        results = {
            "total_files": 0,
            "processed_files": 0,
            "total_concepts": 0,
            "concept_summary": {},
        }
        
        # JSON dosyalarını bul
        spec_files = list(self.spec_dir.glob("*.json"))
        results["total_files"] = len(spec_files)
        
        all_concepts = []
        
        for i, spec_file in enumerate(spec_files):
            print(f"[{i+1}/{len(spec_files)}] İşleniyor: {spec_file.name}")
            
            try:
                # Spec dosyasını oku
                with open(spec_file, "r", encoding="utf-8") as f:
                    spec_data = json.load(f)
                
                # İçeriği çıkar
                content = spec_data.get("content", "")
                sections = spec_data.get("sections", [])
                
                # Tüm içeriği birleştir
                full_content = content + "\n"
                for section in sections:
                    full_content += section.get("content", "") + "\n"
                
                # Kavramları çıkar
                concepts = self.extractor.extract_from_text(
                    full_content,
                    spec_file.stem
                )
                
                # Register haritasını çıkar
                register_map = self.extractor.extract_register_map(full_content)
                
                # Bit alanlarını çıkar
                bit_fields = self.extractor.extract_bit_fields(full_content)
                
                # Sonucu kaydet
                result = {
                    "source_file": spec_file.name,
                    "spec_type": spec_data.get("spec_type", "unknown"),
                    "concepts": concepts,
                    "register_map": register_map[:50],  # Max 50
                    "bit_fields": bit_fields[:100],  # Max 100
                }
                
                all_concepts.append(result)
                
                results["processed_files"] += 1
                results["total_concepts"] += (
                    len(concepts["registers"]) +
                    len(concepts["hardware_terms"]) +
                    len(register_map)
                )
                
                print(f"  Kavram: {len(concepts['hardware_terms'])}")
                print(f"  Register: {len(register_map)}")
                print(f"  Bit alanı: {len(bit_fields)}")
                
            except Exception as e:
                print(f"  Hata: {e}")
        
        # Tüm sonuçları kaydet
        output_file = self.output_dir / "all_concepts.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_concepts, f, indent=2, ensure_ascii=False)
        
        # Özet istatistikleri
        results["concept_summary"] = self._summarize_concepts(all_concepts)
        
        # Özet dosyasını kaydet
        summary_file = self.output_dir / "concept_summary.json"
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    def _summarize_concepts(self, all_concepts: List[Dict]) -> Dict[str, Any]:
        """
        Kavram özetini oluştur.
        """
        summary = {
            "by_category": {},
            "top_registers": {},
            "top_hardware_terms": {},
        }
        
        for concept in all_concepts:
            spec_type = concept.get("spec_type", "unknown")
            
            if spec_type not in summary["by_category"]:
                summary["by_category"][spec_type] = 0
            summary["by_category"][spec_type] += 1
            
            # Register'ları say
            for reg in concept.get("register_map", []):
                name = reg.get("name", "unknown")
                if name not in summary["top_registers"]:
                    summary["top_registers"][name] = 0
                summary["top_registers"][name] += 1
            
            # Donanım terimlerini say
            for term in concept.get("concepts", {}).get("hardware_terms", []):
                term_name = term.get("term", "unknown")
                if term_name not in summary["top_hardware_terms"]:
                    summary["top_hardware_terms"][term_name] = 0
                summary["top_hardware_terms"][term_name] += 1
        
        # En çok kullanılanları sırala
        summary["top_registers"] = dict(
            sorted(summary["top_registers"].items(), key=lambda x: x[1], reverse=True)[:20]
        )
        summary["top_hardware_terms"] = dict(
            sorted(summary["top_hardware_terms"].items(), key=lambda x: x[1], reverse=True)[:20]
        )
        
        return summary


def main():
    """Ana fonksiyon."""
    print("="*60)
    print("echOS Spec Kavram Çıkarıcı")
    print("="*60)
    
    # Pipeline'ı oluştur
    pipeline = SpecConceptPipeline(
        spec_dir=KRM_DATA_DIR,
        output_dir=KRM_CONCEPTS_DIR,
    )
    
    # Tüm spec'leri işle
    results = pipeline.process_all_specs()
    
    # Sonuçları göster
    print("\n" + "="*60)
    print("SONUÇLAR")
    print("="*60)
    
    print(f"İşlenen dosya: {results['processed_files']}")
    print(f"Toplam kavram: {results['total_concepts']}")
    
    print("\nKategori Dağılımı:")
    for cat, count in results["concept_summary"]["by_category"].items():
        print(f"  {cat}: {count}")
    
    print("\nEn Çok Kullanılan Donanım Terimleri:")
    for term, count in list(results["concept_summary"]["top_hardware_terms"].items())[:10]:
        print(f"  {term}: {count}")
    
    print("\nEn Çok Kullanılan Register'lar:")
    for reg, count in list(results["concept_summary"]["top_registers"].items())[:10]:
        print(f"  {reg}: {count}")
    
    print("\nÇıktı dizini:", KRM_CONCEPTS_DIR)


if __name__ == "__main__":
    main()
