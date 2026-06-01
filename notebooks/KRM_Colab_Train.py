"""
KRM-Core RWKV Training - Colab Script
=====================================
Tüm GPU'yu kullanarak RWKV modelini eğitir.
Drive'dan veri okur, BPE tokenizer eğitir, model eğitir.

Kullanim:
  1. Colab'da Runtime > Change runtime type > GPU sec
  2. Bu scripti calistir
  3. Drive mount et
  4. Dataset sec ve egit
"""

# ============================================================
# HUCRE 1: KURULUM
# ============================================================
!pip install -q torch numpy tqdm datasets tokenizers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json, math, time, gc, os

# GPU kontrol
assert torch.cuda.is_available(), "GPU ac! Runtime > Change runtime type > T4 GPU"
device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")

# ============================================================
# HUCRE 2: DRIVE MOUNT
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

DRIVE_DATA = Path('/content/drive/MyDrive/KRM-Core/Datasets')
DRIVE_OUT = Path('/content/drive/MyDrive/KRM-Core/trained')
DRIVE_OUT.mkdir(parents=True, exist_ok=True)

# Drive'da ne var?
if DRIVE_DATA.exists():
    files = list(DRIVE_DATA.glob('*.jsonl'))
    print(f"\nDrive'da {len(files)} veri dosyasi bulundu:")
    for f in sorted(files):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}: {size_mb:.1f} MB")
else:
    print(f"\nDrive'da veri yok: {DRIVE_DATA}")
    print("Once download_to_drive.py ile veri indir!")

# ============================================================
# HUCRE 3: MODEL AYARLARI
# ============================================================

# === SECIMLER ===
MODEL = "rwkv_1b"        # rwkv_10m, rwkv_50m, rwkv_200m, rwkv_1b
BATCH_SIZE = 8           # T4 icin 8, A100 icin 32
SEQ_LEN = 512            # Dizi uzunlugu
MAX_STEPS = 10000        # Adim sayisi
LR = 3e-4                # Ogrenme hizi
SAVE_EVERY = 1000        # Her 1000 adimda kaydet
VOCAB_SIZE = 32768       # BPE vocab

MODEL_CONFIGS = {
    "rwkv_10m":  {"d_model": 384,  "n_layers": 6,  "d_ff": 1536,  "desc": "36M param"},
    "rwkv_50m":  {"d_model": 640,  "n_layers": 12, "d_ff": 2560,  "desc": "74M param"},
    "rwkv_200m": {"d_model": 1024, "n_layers": 16, "d_ff": 4096,  "desc": "252M param"},
    "rwkv_1b":   {"d_model": 1536, "n_layers": 24, "d_ff": 6144,  "desc": "780M param"},
}

cfg = MODEL_CONFIGS[MODEL]
print(f"\nModel: {MODEL} ({cfg['desc']})")
print(f"Batch: {BATCH_SIZE} | Seq: {SEQ_LEN} | Steps: {MAX_STEPS} | LR: {LR}")

# ============================================================
# HUCRE 4: RWKV MODEL
# ============================================================

class WKVMemory(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.log_gain = nn.Parameter(torch.zeros(d))
        self.log_decay = nn.Parameter(torch.zeros(d))

    def forward(self, v, k):
        B, S, D = v.shape
        g = torch.exp(self.log_gain)
        d = torch.exp(-torch.exp(self.log_decay))
        o = torch.zeros_like(v)
        s = torch.zeros(B, D, device=v.device)
        n = torch.zeros(B, D, device=v.device)
        for t in range(S):
            kt = k[:, t] * g
            s = s * d + kt * v[:, t]
            n = n * d + kt
            o[:, t] = s / (n + 1e-8)
        return o


class RWKVBlock(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.time_k = nn.Linear(d, d, bias=False)
        self.time_v = nn.Linear(d, d, bias=False)
        self.time_r = nn.Linear(d, d, bias=False)
        self.time_o = nn.Linear(d, d, bias=False)
        self.wkv = WKVMemory(d)
        self.chan_k = nn.Linear(d, d_ff, bias=False)
        self.chan_v = nn.Linear(d_ff, d, bias=False)
        self.chan_r = nn.Linear(d, d, bias=False)

    def forward(self, x):
        # Time-mixing
        h = self.ln1(x)
        k = self.time_k(h)
        v = self.time_v(h)
        r = torch.sigmoid(self.time_r(h))
        wkv_out = self.wkv(v, k)
        x = x + r * self.time_o(wkv_out)
        # Channel-mixing
        h = self.ln2(x)
        k = torch.relu(self.chan_k(h)) ** 2
        r = torch.sigmoid(self.chan_r(h))
        x = x + r * self.chan_v(k)
        return x


class RWKVLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, d_ff):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([RWKVBlock(d_model, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


# Model olustur ve GPU'ya gonder
model = RWKVLM(VOCAB_SIZE, cfg["d_model"], cfg["n_layers"], cfg["d_ff"]).to(device)
params = sum(p.numel() for p in model.parameters())
mem_mb = sum(p.numel() * 4 for p in model.parameters()) / 1e6
print(f"\nModel: {params:,} param ({mem_mb:.0f} MB FP32)")
print(f"GPU bellek: {torch.cuda.memory_allocated()/1e6:.1f} MB")

# ============================================================
# HUCRE 5: BPE TOKENIZER
# ============================================================

from tokenizers import Tokenizer, trainers, pre_tokenizers, normalizers, decoders, processors
from tokenizers.models import BPE
from tokenizers import AddedToken

SPECIAL_TOKENS = [
    "<|SOURCE|>", "<|QUERY|>", "<|CONTEXT|>", "<|CONCEPTS|>", "<|EDGES|>",
    "<|HOT_GRAPH|>", "<|POLICY|>", "<|ANSWER_PLAN|>", "<|ANSWER|>", "<|INTENT|>",
    "<|DOMAIN|>", "<|SHARD|>", "<|UNCERTAIN|>", "<|SPECULATIVE|>", "<|DO_NOT_CLAIM|>",
    "<|SUPPORTED_BY_SOURCE|>", "<|MISSING_INFO|>", "<|END|>",
    "<|PAD|>", "<|UNK|>", "<|BOS|>", "<|EOS|>",
]

class BPETokenizer:
    def __init__(self, tok):
        self._tok = tok
        self.vocab_size = tok.get_vocab_size()
    def encode(self, text):
        return self._tok.encode(text).ids
    def decode(self, ids):
        return self._tok.decode(ids)

tok_path = DRIVE_OUT.parent / "tokenizer.json"
if tok_path.exists():
    tok = Tokenizer.from_file(str(tok_path))
    print(f"Tokenizer yuklendi: {tok_path}")
else:
    print("BPE tokenizer egitiliyor (vocab=32K)...")
    corpus_files = [str(f) for f in DRIVE_DATA.glob("*.jsonl")]
    if not corpus_files:
        print("HATA: Drive'da veri dosyasi yok!")
        print(f"Beklenen: {DRIVE_DATA}/*.jsonl")
    else:
        print(f"{len(corpus_files)} dosya bulundu")
        tok = Tokenizer(BPE(unk_token="<|UNK|>"))
        tok.normalizer = normalizers.NFKC()
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.post_processor = processors.ByteLevel(trim_offsets=False)
        special = [AddedToken(t, single_word=False, normalized=False) for t in SPECIAL_TOKENS]
        tok.add_special_tokens(special)
        trainer = trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            min_frequency=2,
            special_tokens=[str(t) for t in special],
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train(corpus_files, trainer)
        tok.save(str(tok_path))
        print(f"Tokenizer kaydedildi: {tok_path}")

tokenizer = BPETokenizer(tok)
print(f"Vocab: {tokenizer.vocab_size:,}")

# ============================================================
# HUCRE 6: VERIYI OKU + TOKENIZE ET
# ============================================================

print("\nDrive'dan veriler okunuyor...")
all_ids = []

for ds_file in sorted(DRIVE_DATA.glob("*.jsonl")):
    print(f"  {ds_file.stem}...", end=" ", flush=True)
    count = 0
    with open(ds_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text", "")
            if not text or len(text) < 50:
                continue
            ids = tokenizer.encode(text)
            all_ids.extend(ids)
            all_ids.append(tokenizer.encode("<|END|>")[0])
            count += 1
    print(f"{count:,} ornek | toplam {len(all_ids):,} token")

total_tokens = len(all_ids)
print(f"\nToplam: {total_tokens:,} token (~{total_tokens/1e6:.1f}M)")
del tok
gc.collect()

# ============================================================
# HUCRE 7: DATASET
# ============================================================

class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.seq_len = seq_len
        n = (len(tokens) - 1) // seq_len
        self.data = torch.tensor(tokens[:n * seq_len + 1], dtype=torch.long)
        self.n_samples = n
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        s = idx * self.seq_len
        return self.data[s:s + self.seq_len], self.data[s + 1:s + self.seq_len + 1]

print("\nDataset olusturuluyor...")
dataset = TokenDataset(all_ids, SEQ_LEN)
del all_ids
gc.collect()

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)
print(f"Chunks: {len(dataset):,} | Batch: {len(loader):,}")

# ============================================================
# HUCRE 8: EGITIM
# ============================================================

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.99))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_STEPS, eta_min=LR/10)

# Mixed precision icin
scaler = torch.amp.GradScaler("cuda")

# Checkpoint'den devam
step = 0
best_loss = float("inf")
history = {"step": [], "loss": [], "ppl": [], "lr": []}
resume_path = DRIVE_OUT / f"{MODEL}_resume.pt"

if resume_path.exists():
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    step = ckpt.get("step", 0)
    best_loss = ckpt.get("best_loss", float("inf"))
    history = ckpt.get("history", history)
    print(f"\nDevam: step {step}, best_loss {best_loss:.4f}")

print(f"\n{'='*60}")
print(f"EGITIM: {MODEL} ({cfg['desc']})")
print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
print(f"Batch: {BATCH_SIZE} | Seq: {SEQ_LEN} | Steps: {MAX_STEPS}")
print(f"{'='*60}\n")

t0 = time.time()
model.train()

while step < MAX_STEPS:
    for xb, yb in loader:
        if step >= MAX_STEPS:
            break
        step += 1

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # Mixed precision forward
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

        # Backward
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        history["step"].append(step)
        history["loss"].append(loss.item())
        history["ppl"].append(math.exp(min(loss.item(), 20)))
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if step % 10 == 0:
            elapsed = time.time() - t0
            recent = history["loss"][-10:]
            avg_loss = sum(recent) / len(recent)
            avg_ppl = math.exp(min(avg_loss, 20))
            speed = step / elapsed if elapsed > 0 else 0
            print(
                f"  Step {step:5d}/{MAX_STEPS} | "
                f"Loss: {avg_loss:.4f} | PPL: {avg_ppl:.2f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"{speed:.1f} step/s | {elapsed:.0f}s"
            )

        if step % SAVE_EVERY == 0:
            # Resume kaydet
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
                "step": step,
                "best_loss": best_loss,
                "vocab_size": tokenizer.vocab_size,
                "history": history,
            }, resume_path)

            # En iyi modeli kaydet
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), DRIVE_OUT / f"{MODEL}_best.pt")

            # Adim kaydi
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg,
                "step": step,
                "best_loss": best_loss,
                "vocab_size": tokenizer.vocab_size,
                "history": history,
            }, DRIVE_OUT / f"{MODEL}_step{step}.pt")

            print(f"    [*] Kaydedildi: step {step}")

print(f"\n{'='*60}")
print(f"TAMAMLANDI!")
print(f"Adim: {step} | Sure: {time.time()-t0:.0f}s")
print(f"Best Loss: {best_loss:.4f} (PPL: {math.exp(min(best_loss, 20)):.2f})")
print(f"GPU bellek: {torch.cuda.max_memory_allocated()/1e6:.0f} MB max")
print(f"{'='*60}")

# ============================================================
# HUCRE 9: URETIM TESTI
# ============================================================

model.eval()
print(f"\n{'='*60}")
print("URETIM TESTI")
print(f"{'='*60}")

prompts = [
    "Turkiye'nin baskenti neresidir?",
    "Python'da fibonacci nasil yazilir?",
    "Bir dikdortgenin alani kac formul ile hesaplanir?",
    "Write a function to check prime numbers:",
    "What is the time complexity of merge sort?",
]

for p in prompts:
    print(f"\n{'-'*50}")
    print(f"Soru: {p}")
    ids = torch.tensor([tokenizer.encode(p)], device=device)
    with torch.no_grad():
        for _ in range(500):
            logits = model(ids[:, -SEQ_LEN:])
            probs = F.softmax(logits[0, -1] / 0.8, dim=0)
            nid = torch.multinomial(probs, 1).item()
            ids = torch.cat([ids, torch.tensor([[nid]], device=device)], dim=1)
            if nid == tokenizer.encode("<|END|>")[0]:
                break
    output = tokenizer.decode(ids[0].tolist())
    gen = output[len(p):]
    print(f"Cevap: {gen[:500]}")

# ============================================================
# HUCRE 10: SON KAYIT
# ============================================================

# Final model
torch.save({
    "model_state": model.state_dict(),
    "config": cfg,
    "step": step,
    "best_loss": best_loss,
    "vocab_size": tokenizer.vocab_size,
    "history": history,
}, DRIVE_OUT / f"{MODEL}_final.pt")

# Config
with open(DRIVE_OUT / "config.json", "w") as f:
    json.dump({
        "model": MODEL,
        **cfg,
        "vocab_size": tokenizer.vocab_size,
        "best_loss": best_loss,
        "final_step": step,
        "gpu": gpu_name,
        "gpu_mem_gb": gpu_mem,
    }, f, indent=2)

# History
with open(DRIVE_OUT / "history.json", "w") as f:
    json.dump(history, f)

print(f"\nDrive'a kaydedildi: {DRIVE_OUT}/")
for f in sorted(DRIVE_OUT.glob("*")):
    size = f.stat().st_size / 1e6
    print(f"  {f.name}: {size:.1f} MB")

print(f"\nBitis! GPU max bellek: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")
