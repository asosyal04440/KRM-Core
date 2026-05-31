"""
echOS Spec Ingester - Spec'leri KRM-Core'a ekler.

Bu script, echOS Kaynak Arşivi'ndeki spec dosyalarını okuyup
KRM-Core formatında kaydeder.

ÖNEMLI: Orijinal dosyaları DEĞİŞTİRMEZ, sadece KRM-Core kopyalar.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# KRM-Core path
KRM_ROOT = Path(__file__).parent.parent
KRM_DATA_DIR = KRM_ROOT / "data" / "echos_specs"


@dataclass
class SpecSection:
    """Spec içindeki bir bölüm."""
    title: str
    content: str
    level: int = 1
    subsections: List["SpecSection"] = field(default_factory=list)


@dataclass
class SpecDocument:
    """Bir spec dosyası."""
    name: str
    source_path: str
    spec_type: str  # hardware, kernel, filesystem, etc.
    content: str
    sections: List[SpecSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpecIngester:
    """
    Spec dosyalarını KRM-Core'a ekler.
    """
    
    def __init__(
        self,
        spec_archive_path: str = r"D:\echOS Kaynak Arşivi",
        output_path: str | None = None,
    ):
        """
        Ingester'ı başlat.
        
        Args:
            spec_archive_path: Spec arşivi dizini
            output_path: Çıktı dizini (varsayılan: KRM-Core/data/echos_specs)
        """
        self.spec_archive = Path(spec_archive_path)
        self.output_path = Path(output_path) if output_path else KRM_DATA_DIR
        
        # Çıktı dizinini oluştur
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Desteklenen formatlar
        self.supported_extensions = {".html", ".htm", ".md", ".txt", ".pdf"}
        
        # Spec kategorileri
        self.spec_categories = {
            "hardware": [
                "virtio", "nvme", "ahci", "sata", "pci", "usb", "xhci",
                "acpi", "uefi", "edk2", "e1000", "intel"
            ],
            "kernel": [
                "kernel", "linux", "freebsd", "driver"
            ],
            "filesystem": [
                "ext4", "f2fs", "fat32", "exfat", "btrfs", "xfs", "ntfs"
            ],
            "network": [
                "tcp", "udp", "ip", "dns", "dhcp", "arp", "rfc"
            ],
            "development": [
                "gcc", "binutils", "rust", "llvm"
            ]
        }
    
    def scan_specs(self) -> List[Path]:
        """
        Spec arşivini tara ve dosya listesi döndür.
        """
        specs = []
        
        for ext in self.supported_extensions:
            specs.extend(self.spec_archive.rglob(f"*{ext}"))
        
        print(f"Toplam {len(specs)} spec dosyası bulundu")
        
        return specs
    
    def categorize_spec(self, file_path: Path) -> str:
        """
        Spec dosyasının kategorisini belirle.
        """
        path_str = str(file_path).lower()
        
        for category, keywords in self.spec_categories.items():
            for keyword in keywords:
                if keyword in path_str:
                    return category
        
        return "other"
    
    def read_html_spec(self, file_path: Path) -> Optional[str]:
        """
        HTML spec dosyasını oku ve temizle.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            # HTML tag'lerini temizle
            content = re.sub(r'<[^>]+>', ' ', content)
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()
            
            return content
            
        except Exception as e:
            print(f"Hata ({file_path}): {e}")
            return None
    
    def read_text_spec(self, file_path: Path) -> Optional[str]:
        """
        Metin spec dosyasını oku.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            print(f"Hata ({file_path}): {e}")
            return None
    
    def extract_sections(self, content: str, spec_type: str) -> List[SpecSection]:
        """
        İçerikten bölümleri çıkar.
        """
        sections = []
        
        # Basit bölüm çıkarma (header bazlı)
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            # Header tespiti
            if re.match(r'^#{1,6}\s+', line) or re.match(r'^[A-Z][A-Z\s]{2,}$', line):
                # Önceki bölümü kaydet
                if current_section and current_content:
                    current_section.content = '\n'.join(current_content)
                    sections.append(current_section)
                
                # Yeni bölüm başlat
                title = re.sub(r'^#{1,6}\s+', '', line).strip()
                current_section = SpecSection(
                    title=title,
                    content="",
                    level=1
                )
                current_content = []
            else:
                current_content.append(line)
        
        # Son bölümü kaydet
        if current_section and current_content:
            current_section.content = '\n'.join(current_content)
            sections.append(current_section)
        
        return sections
    
    def create_spec_document(
        self,
        file_path: Path,
        content: str,
        category: str,
    ) -> SpecDocument:
        """
        Spec belgesi oluştur.
        """
        sections = self.extract_sections(content, category)
        
        return SpecDocument(
            name=file_path.stem,
            source_path=str(file_path),
            spec_type=category,
            content=content,
            sections=sections,
            metadata={
                "file_size": file_path.stat().st_size,
                "extension": file_path.suffix,
                "category": category,
            }
        )
    
    def save_spec_document(self, doc: SpecDocument) -> str:
        """
        Spec belgesini KRM-Core formatında kaydet.
        """
        # Çıktı dosya yolu
        output_file = self.output_path / f"{doc.name}.json"
        
        # Belge verisi
        doc_data = {
            "name": doc.name,
            "source_path": doc.source_path,
            "spec_type": doc.spec_type,
            "content": doc.content[:10000],  # İlk 10K karakter
            "sections": [
                {
                    "title": s.title,
                    "content": s.content[:2000],  # Her bölüm için 2K
                    "level": s.level
                }
                for s in doc.sections[:50]  # Max 50 bölüm
            ],
            "metadata": doc.metadata,
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(doc_data, f, indent=2, ensure_ascii=False)
        
        return str(output_file)
    
    def ingest_all(self, max_files: int = 100) -> List[str]:
        """
        Tüm spec dosyalarını KRM-Core'a ekle.
        """
        print(f"Spec arşivi taranıyor: {self.spec_archive}")
        
        specs = self.scan_specs()
        
        # Kategorilere göre filtrele
        priority_specs = []
        for spec in specs:
            category = self.categorize_spec(spec)
            if category in ["hardware", "filesystem"]:
                priority_specs.append((spec, category))
        
        # Sırala (öncelikli olanlar önce)
        priority_specs.sort(key=lambda x: x[0].stat().st_size)
        
        print(f"Öncelikli spec sayısı: {len(priority_specs)}")
        
        # İşle
        ingested = []
        
        for i, (spec_path, category) in enumerate(priority_specs[:max_files]):
            print(f"\n[{i+1}/{min(len(priority_specs), max_files)}] İşleniyor: {spec_path.name}")
            
            # Oku
            if spec_path.suffix == ".html":
                content = self.read_html_spec(spec_path)
            else:
                content = self.read_text_spec(spec_path)
            
            if not content:
                continue
            
            # Belge oluştur
            doc = self.create_spec_document(spec_path, content, category)
            
            # Kaydet
            output_path = self.save_spec_document(doc)
            
            ingested.append(output_path)
            
            print(f"  Kategori: {category}")
            print(f"  Boyut: {len(content)} karakter")
            print(f"  Bölüm: {len(doc.sections)}")
            print(f"  Kaydedildi: {output_path}")
        
        print(f"\n{'='*60}")
        print(f"Toplam {len(ingested)} spec eklendi")
        print(f"Çıktı dizini: {self.output_path}")
        
        return ingested
    
    def create_ingest_summary(self, ingested: List[str]) -> None:
        """
        Ingest özeti oluştur.
        """
        summary = {
            "total_files": len(ingested),
            "output_directory": str(self.output_path),
            "files": ingested,
        }
        
        summary_path = self.output_path / "ingest_summary.json"
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nÖzet kaydedildi: {summary_path}")


def main():
    """Ana fonksiyon."""
    print("="*60)
    print("echOS Spec Ingester")
    print("="*60)
    
    # Ingester oluştur
    ingester = SpecIngester()
    
    # Tüm spec'leri ekle
    ingested = ingester.ingest_all(max_files=50)
    
    # Özet oluştur
    ingester.create_ingest_summary(ingested)
    
    print("\nTamamlandı!")


if __name__ == "__main__":
    main()
