"""
echOS Mega Corpus ile Gercek RWKV Egitimi
RWKV model = train_conceptual_sharded.py'daki ConceptBlock
"""

import gc, json, os, time, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

# RWKV modelini train_conceptual_sharded.py'dan al
from train_conceptual_sharded import ConceptBlock, WKVMemory

# ============================================================
# BASIT KARAKTER TOKENIZER
# ============================================================

class CharTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.unk_token = 3

    def encode(self, text):
        tokens = [self.bos_token]
        for ch in text:
            b = ord(ch)
            tokens.append(b if b < 256 else self.unk_token)
        tokens.append(self.eos_token)
        return tokens

    def decode(self, tokens):
        chars = []
        for t in tokens:
            if t >= 4 and t < 256:
                chars.append(chr(t))
            elif t == self.eos_token:
                break
        return "".join(chars)


# ============================================================
# CORPUS VERI SETI
# ============================================================

class CorpusDataset(Dataset):
    def __init__(self, corpus_path, seq_len=256, max_samples=5000):
        self.seq_len = seq_len
        self.tokenizer = CharTokenizer()
        self.vocab_size = self.tokenizer.vocab_size

        print(f"  Reading: {corpus_path}")
        text = Path(corpus_path).read_text("utf-8", errors="replace")
        print(f"  Size: {len(text):,} chars")

        self.tokens = self.tokenizer.encode(text)
        print(f"  Tokens: {len(self.tokens):,}")

        stride = max(1, (len(self.tokens) - seq_len) // max_samples) if max_samples > 0 else 1
        self.num_samples = min(max_samples, max(1, (len(self.tokens) - seq_len) // stride))
        self.stride = stride
        print(f"  Samples: {self.num_samples:,}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.tokens[start:start + self.seq_len]
        if len(chunk) < self.seq_len:
            chunk = chunk + [self.tokenizer.pad_token] * (self.seq_len - len(chunk))
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ============================================================
# EGITIM
# ============================================================

def main():
    print("=" * 60)
    print("ECHOS REAL TRAINING - RWKV KAVRAMSAL PARCALI")
    print("=" * 60)

    # Config
    config = {
        "corpus": r"D:\yeni_ai_hiyerarsisi\krm_core\data\rwkv_training\echos_mega_corpus.txt",
        "output_dir": "echos_training_output",
        "d_model": 128,
        "num_blocks": 1,
        "batch_size": 8,
        "seq_len": 128,
        "num_epochs": 5,
        "lr": 5e-4,
        "grad_clip": 1.0,
        "max_samples": 2000,
        "log_interval": 20,
    }

    print(f"\nConfig: {json.dumps(config, indent=2)}")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    print("\n--- Loading Dataset ---")
    dataset = CorpusDataset(
        config["corpus"],
        seq_len=config["seq_len"],
        max_samples=config["max_samples"]
    )

    val_size = int(len(dataset) * 0.05)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    # Model
    print("\n--- Creating Model ---")
    block = ConceptBlock(
        block_id="echos_block_00",
        vocab_size=dataset.vocab_size,
        d_model=config["d_model"],
        n_layers=2,
        d_ff=config["d_model"] * 4,
    )
    params = block.get_param_count()
    print(f"Params: {params:,}")
    print(f"Memory: {params * 4 / 1024 / 1024:.1f} MB")

    optim = torch.optim.AdamW(block.parameters(), lr=config["lr"], weight_decay=0.01)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Training
    print("\n--- Training ---")
    history = {"train_loss": [], "val_loss": [], "val_ppl": [], "epoch_times": []}
    best_val_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        t0 = time.time()

        # Train
        block.train().to(device)
        total_loss = 0
        total_tokens = 0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits, loss = block(x, y)

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(block.parameters(), config["grad_clip"])
            optim.step()

            total_loss += loss.item()
            total_tokens += y.numel()

            if i % config["log_interval"] == 0:
                print(f"  Ep {epoch+1} | B {i}/{len(train_loader)} | Loss: {total_loss/total_tokens:.4f}")

        avg_train_loss = total_loss / total_tokens

        # Validation
        block.eval()
        val_loss = 0
        val_tokens = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, loss = block(x, y)
                val_loss += loss.item()
                val_tokens += y.numel()

        avg_val_loss = val_loss / val_tokens
        val_ppl = torch.exp(torch.tensor(avg_val_loss)).item()
        epoch_time = time.time() - t0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_ppl"].append(val_ppl)
        history["epoch_times"].append(epoch_time)

        print(f"\n--- Epoch {epoch+1} ---")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val PPL:    {val_ppl:.2f}")
        print(f"  Time:       {epoch_time:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "block_state": block.state_dict(),
                "config": config,
                "val_loss": avg_val_loss,
            }, output_dir / "best_model.pt")
            print(f"  [*] Saved best!")

        block.cpu()
        gc.collect()

    # Final save
    torch.save({
        "block_state": block.state_dict(),
        "config": config,
        "history": history,
        "val_loss": avg_val_loss,
    }, output_dir / "final_model.pt")

    # Summary
    print(f"\n{'=' * 60}")
    print("DONE!")
    print(f"Best val loss: {best_loss_info['val']:.4f}" if 'best_loss_info' in locals() else f"Best val loss: {best_val_loss:.4f}")
    print(f"Total params:  {params:,}")
    print(f"Output:        {output_dir}/")

    # Generate sample
    print("\n--- Generation ---")
    block.to(device).eval()
    seed = torch.randint(4, dataset.vocab_size, (1, 1), device=device)
    generated = seed.tolist()[0]
    with torch.no_grad():
        for _ in range(200):
            logits, _ = block(seed, None)
            probs = F.softmax(logits[0, -1] / 1.0, dim=0)
            t = torch.multinomial(probs, 1).item()
            generated.append(t)
            seed = torch.cat([seed, torch.tensor([[t]], device=device)], dim=1)[:, -config["seq_len"]:]
    print(dataset.tokenizer.decode(generated))


if __name__ == "__main__":
    main()
