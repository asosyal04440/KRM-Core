"""
Büyük Spec Dosyalarını İşleme Scripti

HTML spec'lerini temizleyip KRM-Core formatında kaydeder.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class LargeSpecProcessor:
    """
    Büyük spec dosyalarını işleyen sınıf.
    """
    
    def __init__(self, output_dir: str | Path):
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def clean_html(self, html_content: str) -> str:
        """
        HTML'i temizle ve saf metin çıkar.
        """
        # Script ve style tag'lerini kaldır
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL)
        
        # Yorumları kaldır
        html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)
        
        # Tabloları koru (önemli bilgi içerebilir)
        # Önce tabloları bul ve işaretle
        tables = re.findall(r'<table[^>]*>.*?</table>', html_content, flags=re.DOTALL)
        
        # HTML tag'lerini kaldır
        html_content = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Özel karakterleri temizle
        html_content = html_content.replace('&nbsp;', ' ')
        html_content = html_content.replace('&amp;', '&')
        html_content = html_content.replace('&lt;', '<')
        html_content = html_content.replace('&gt;', '>')
        html_content = html_content.replace('&quot;', '"')
        
        # Boşlukları temizle
        html_content = re.sub(r'\s+', ' ', html_content)
        
        return html_content.strip()
    
    def extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        İçerikten bölümleri çıkar.
        """
        sections = []
        
        # Header desenleri
        header_patterns = [
            (r'^(#{1,6})\s+(.+)$', 'markdown'),
            (r'^([A-Z][A-Z\s]{2,})$', 'uppercase'),
            (r'^(\d+\.[\d\.]*)\s+(.+)$', 'numbered'),
        ]
        
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            
            if not line_stripped:
                if current_content:
                    current_content.append('')
                continue
            
            # Header tespiti
            is_header = False
            for pattern, header_type in header_patterns:
                match = re.match(pattern, line_stripped, re.MULTILINE)
                if match:
                    # Önceki bölümü kaydet
                    if current_section:
                        current_section['content'] = '\n'.join(current_content).strip()
                        if current_section['content']:
                            sections.append(current_section)
                    
                    # Yeni bölüm
                    if header_type == 'markdown':
                        title = match.group(2).strip()
                        level = len(match.group(1))
                    elif header_type == 'uppercase':
                        title = line_stripped.strip()
                        level = 1
                    else:
                        title = match.group(2).strip()
                        level = 1
                    
                    current_section = {
                        'title': title,
                        'level': level,
                        'content': ''
                    }
                    current_content = []
                    is_header = True
                    break
            
            if not is_header:
                current_content.append(line_stripped)
        
        # Son bölümü kaydet
        if current_section:
            current_section['content'] = '\n'.join(current_content).strip()
            if current_section['content']:
                sections.append(current_section)
        
        return sections
    
    def process_html_file(
        self,
        file_path: Path,
        spec_type: str,
        max_content_length: int = 500000,
    ) -> Dict[str, Any]:
        """
        HTML dosyasını işle.
        """
        print(f"  Okunuyor: {file_path.name}")
        
        # Dosyayı oku
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        print(f"  Ham boyut: {len(content)} karakter")
        
        # İçeriği kısalt (çok uzunsa)
        if len(content) > max_content_length:
            content = content[:max_content_length]
            print(f"  Kısaltıldı: {max_content_length} karakter")
        
        # HTML'i temizle
        clean_content = self.clean_html(content)
        print(f"  Temizlenmiş boyut: {len(clean_content)} karakter")
        
        # Bölümleri çıkar
        sections = self.extract_sections(clean_content)
        print(f"  Bölüm sayısı: {len(sections)}")
        
        return {
            "name": file_path.stem,
            "source_path": str(file_path),
            "spec_type": spec_type,
            "raw_size": len(content),
            "clean_size": len(clean_content),
            "content": clean_content[:100000],  # Max 100K
            "sections": sections[:200],  # Max 200 bölüm
            "metadata": {
                "file_size": file_path.stat().st_size,
                "extension": file_path.suffix,
            }
        }
    
    def save_spec(self, spec_data: Dict[str, Any]) -> str:
        """
        İşlenmiş spec'ı kaydet.
        """
        output_file = self.output_path / f"{spec_data['name']}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(spec_data, f, indent=2, ensure_ascii=False)
        
        return str(output_file)
    
    def process_priority_specs(self) -> List[str]:
        """
        Öncelikli spec dosyalarını işle.
        """
        # Öncelikli spec'ler
        priority_specs = [
            # VirtIO
            {
                "path": r"D:\echOS Kaynak Arşivi\echos_virtio_specs\specs\virtio-v1.2.html",
                "type": "hardware",
                "name": "virtio-v1.2"
            },
            # AHCI
            {
                "path": r"D:\echOS Kaynak Arşivi\specs\serial-ata-ahci-spec-rev1-3-1.pdf",
                "type": "hardware",
                "name": "ahci-spec"
            },
            # NVMe (kernel docs)
            {
                "path": r"D:\echOS Kaynak Arşivi\02_FileSystems_and_Storage\nvme-kernel-docs.html",
                "type": "hardware",
                "name": "nvme-kernel-docs"
            },
            # PCI
            {
                "path": r"D:\echOS Kaynak Arşivi\03_Hardware_Specs\pci-kernel-docs.html",
                "type": "hardware",
                "name": "pci-kernel-docs"
            },
            # UEFI
            {
                "path": r"D:\echOS Kaynak Arşivi\03_Hardware_Specs\uefi-spec-download.html",
                "type": "hardware",
                "name": "uefi-spec"
            },
            # x86_64 ABI
            {
                "path": r"D:\echOS Kaynak Arşivi\03_Hardware_Specs\x86_64-abi-reference.html",
                "type": "hardware",
                "name": "x86-64-abi"
            },
            # Intel SDM
            {
                "path": r"D:\echOS Kaynak Arşivi\03_Hardware_Specs\intel-sdm-overview.html",
                "type": "hardware",
                "name": "intel-sdm"
            },
            # GPT
            {
                "path": r"D:\echOS Kaynak Arşivi\03_Hardware_Specs\gpt-spec.html",
                "type": "hardware",
                "name": "gpt-spec"
            },
        ]
        
        saved_files = []
        
        for spec_info in priority_specs:
            file_path = Path(spec_info["path"])
            
            if not file_path.exists():
                print(f"  Dosya bulunamadı: {file_path}")
                continue
            
            if file_path.suffix.lower() == '.pdf':
                print(f"  PDF dosyası atlanıyor: {file_path.name}")
                continue
            
            print(f"\n{'='*60}")
            print(f"İşleniyor: {spec_info['name']}")
            print(f"{'='*60}")
            
            try:
                spec_data = self.process_html_file(
                    file_path,
                    spec_info["type"]
                )
                
                output_path = self.save_spec(spec_data)
                saved_files.append(output_path)
                
                print(f"  Kaydedildi: {output_path}")
                
            except Exception as e:
                print(f"  Hata: {e}")
        
        return saved_files


def main():
    """Ana fonksiyon."""
    print("="*60)
    print("Büyük Spec Dosyalarını İşleme")
    print("="*60)
    
    output_dir = Path(__file__).parent.parent / "data" / "echos_specs"
    
    processor = LargeSpecProcessor(output_dir)
    
    saved_files = processor.process_priority_specs()
    
    print(f"\n{'='*60}")
    print(f"Toplam {len(saved_files)} spec işlendi")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
