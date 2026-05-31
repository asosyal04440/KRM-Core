"""
Kavramsal Parçalı Eğitim - Test Dosyası
"""

import sys
from pathlib import Path

# Proje dizinini sys.path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from krm.training.conceptual_sharding import (
    ConceptBlock,
    ConceptMap,
    ConceptualSharder,
    ConceptualTrainingPipeline,
    create_training_pipeline,
)


def test_concept_block():
    """ConceptBlock testi."""
    print("=== ConceptBlock Testi ===")
    
    block = ConceptBlock(
        block_id="block_0001",
        concept_name="algorithms",
        domain_id=1,
        parameter_count=400_000_000,
        layer_indices=[0, 1, 2, 3],
    )
    
    print(f"Blok ID: {block.block_id}")
    print(f"Konsept: {block.concept_name}")
    print(f"Alan: {block.domain_id}")
    print(f"Parametreler: {block.parameter_count:,}")
    print(f"Boyut: {block.estimate_size_bytes() / 1024 / 1024:.2f} MB")
    
    # Dict'e dönüştür
    d = block.to_dict()
    print(f"Dict: {d}")
    
    # Geri oluştur
    block2 = ConceptBlock.from_dict(d)
    assert block.block_id == block2.block_id
    print("[OK] ConceptBlock testi gecti\n")


def test_concept_map():
    """ConceptMap testi."""
    print("=== ConceptMap Testi ===")
    
    # Blok oluştur
    blocks = [
        ConceptBlock(
            block_id=f"block_{i:04d}",
            concept_name=f"concept_{i}",
            domain_id=i % 16,
            parameter_count=400_000_000,
        )
        for i in range(10)
    ]
    
    # Harita oluştur
    concept_map = ConceptMap(
        total_parameters=4_000_000_000,
        total_blocks=10,
        blocks=blocks,
        domains={i: f"domain_{i}" for i in range(16)},
        adjacency={
            "block_0000": ["block_0001"],
            "block_0001": ["block_0000", "block_0002"],
        },
    )
    
    print(f"Toplam Parametre: {concept_map.total_parameters:,}")
    print(f"Toplam Blok: {concept_map.total_blocks}")
    
    # Alan bazlı filtreleme
    domain_blocks = concept_map.get_blocks_by_domain(0)
    print(f"Alan 0'daki bloklar: {len(domain_blocks)}")
    
    # İlişkili bloklar
    related = concept_map.get_related_blocks("block_0001")
    print(f"block_0001 ilişkileri: {[b.block_id for b in related]}")
    
    # Kaydet ve yükle
    test_path = "test_concept_map.json"
    concept_map.save(test_path)
    loaded = ConceptMap.load(test_path)
    assert loaded.total_blocks == 10
    
    import os
    os.remove(test_path)
    
    print("[OK] ConceptMap testi gecti\n")


def test_conceptual_sharder():
    """ConceptualSharder testi."""
    print("=== ConceptualSharder Testi ===")
    
    sharder = ConceptualSharder(
        target_parameters=400_000_000_000,  # 400B
        num_blocks=1000,
    )
    
    print(f"Hedef Parametre: {sharder.target_parameters:,}")
    print(f"Blok Sayısı: {sharder.num_blocks}")
    
    # Kavram haritası oluştur
    concept_map = sharder.create_concept_map(
        model_layers=96,  # 96 katman
        embedding_dim=8192,  # 8192 boyut
    )
    
    print(f"Oluşturulan Blok Sayısı: {concept_map.total_blocks}")
    print(f"Toplam Parametre: {concept_map.total_parameters:,}")
    
    # İlk 5 bloğu göster
    print("\nİlk 5 blok:")
    for block in concept_map.blocks[:5]:
        print(f"  {block.block_id}: {block.concept_name} "
              f"({block.parameter_count:,} param)")
    
    print("[OK] ConceptualSharder testi gecti\n")


def test_training_pipeline():
    """ConceptualTrainingPipeline testi."""
    print("=== ConceptualTrainingPipeline Testi ===")
    
    # Pipeline oluştur
    pipeline = create_training_pipeline(
        target_parameters=1_000_000_000,  # 1B (test için küçük)
        num_blocks=10,  # 10 blok (test için az)
        output_dir="test_training_output",
        max_ram_mb=2000,
    )
    
    print(f"Çıktı Dizini: {pipeline.output_dir}")
    print(f"Maks RAM: {pipeline.max_ram_mb} MB")
    
    # Kavram haritası oluştur
    concept_map = pipeline.sharder.create_concept_map(
        model_layers=12,
        embedding_dim=768,
    )
    
    print(f"Oluşturulan Blok: {concept_map.total_blocks}")
    
    # Kavram haritasını kaydet
    pipeline.save_concept_map(concept_map)
    
    print("[OK] ConceptualTrainingPipeline testi gecti\n")


def main():
    """Tüm testleri çalıştır."""
    print("Kavramsal Parçalı Eğitim Testleri\n")
    print("=" * 50)
    
    test_concept_block()
    test_concept_map()
    test_conceptual_sharder()
    test_training_pipeline()
    
    print("=" * 50)
    print("Tüm testler başarılı!")
    print("\nSonraki adımlar:")
    print("1. Gerçek PyTorch model entegrasyonu")
    print("2. Eğitim döngüsü implementasyonu")
    print("3. Gradient yönetimi")
    print("4. Bellek haritalı eğitim")


if __name__ == "__main__":
    main()
