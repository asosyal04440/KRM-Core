"""
KRM-Core: Tek Hücrelik Colab Script
====================================
Bu dosyayı Colab'a kopyala-yapıştır yap, tek hücrede çalışsın.
Drive'a veri indirir, tokenizer eğitir, RWKV modelini eğitir.
"""

# ═══════════════════════════════════════════════════════════
# HÜCRE 1: KURULUM + DRIVE MOUNT
# ═══════════════════════════════════════════════════════════

!pip install torch numpy tqdm datasets tokenizers huggingface_hub

import torch, gc, json, time, os, math
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from google.colab import drive
drive.mount('/content/drive')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ═══════════════════════════════════════════════════════════
# HÜCRE 2: AYARLAR
# ═══════════════════════════════════════════════════════════

DRIVE_OUT = Path('/content/drive/MyDrive/KRM-Core/Datasets')
DRIVE_OUT.mkdir(parents=True, exist_ok=True)

MODEL = 'rwkv_50m'  # rwkv_10m, rwkv_50m, rwkv_200m, rwkv_1b, rwkv_7b

MODEL_CONFIGS = {
    'rwkv_10m':  {'d_model': 384,  'n_layers': 6,  'd_ff': 1536},
    'rwkv_50m':  {'d_model': 640,  'n_layers': 12, 'd_ff': 2560},
    'rwkv_200m': {'d_model': 1024, 'n_layers': 16, 'd_ff': 4096},
    'rwkv_1b':   {'d_model': 1536, 'n_layers': 24, 'd_ff': 6144},
    'rwkv_7b':   {'d_model': 4096, 'n_layers': 32, 'd_ff': 16384},
}

cfg = MODEL_CONFIGS[MODEL]
VOCAB_SIZE = 32768
BATCH_SIZE = 8
SEQ_LEN = 512
MAX_STEPS = 2000
LR = 3e-4
SAVE_EVERY = 200

print(f"Model: {MODEL} (d={cfg['d_model']}, L={cfg['n_layers']}, ff={cfg['d_ff']})")

# ═══════════════════════════════════════════════════════════
# HÜCRE 3: DATASET İNDİR (sadece 1 kez çalıştır)
# ═══════════════════════════════════════════════════════════

from datasets import load_dataset

DATASETS_TO_DOWNLOAD = {
    'turkce_wikipedia': ('Wikipedia', 'tr', 'train', 'text', 'Türkçe Vikipedi'),
    'turkce_oscar':     ('oscar-corpus/OSCAR-2301', 'tr', 'train', 'text', 'OSCAR Türkçe'),
    'math_gsm8k':       ('openai/gsm8k', 'main', 'train', 'question', 'GSM8K Matematik'),
    'math_metamathqa':  ('meta-math/MetaMathQA', None, 'train', 'query', 'MetaMathQA'),
    'cs_the_stack':     ('bigcode/the-stack', 'Python', 'train', 'content', 'The Stack Python'),
    'fineweb_edu':      ('HuggingFaceFW/fineweb-edu', 'sample-10BT', 'train', 'text', 'FineWeb-Edu'),
}

MAX_TOKENS = 500_000_000  # 500M token (her dataset için)

for ds_name, (path, config, split, field, desc) in DATASETS_TO_DOWNLOAD.items():
    out_file = DRIVE_OUT / f'{ds_name}.jsonl'
    if out_file.exists():
        existing = sum(1 for _ in open(out_file, 'r', encoding='utf-8'))
        print(f"✓ {ds_name} zaten mevcut ({existing} satır)")
        continue

    print(f"\n{'─'*50}")
    print(f"İndiriliyor: {desc}")
    print(f"{'─'*50}")

    try:
        ds = load_dataset(path, config, split=split, streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  HATA: {e}")
        continue

    count = 0
    tokens = 0
    t0 = time.time()

    with open(out_file, 'w', encoding='utf-8') as f:
        for i, example in enumerate(ds):
            if tokens >= MAX_TOKENS:
                break
            text = example.get(field, '')
            if not text or len(text) < 10:
                continue
            tok_est = len(text) // 4
            record = json.dumps({'text': text, 'source': ds_name, 'tokens_est': tok_est}, ensure_ascii=False)
            f.write(record + '\n')
            count += 1
            tokens += tok_est
            if count % 5000 == 0:
                print(f"  {count:,} satır | {tokens:,} token | {time.time()-t0:.0f}s")

    print(f"✓ {count:,} satır, {tokens:,} token, {time.time()-t0:.0f}s")

print(f"\n{'='*50}")
print("Tüm indirmeler tamamlandı!")
print(f"{'='*50}")

# ═══════════════════════════════════════════════════════════
# HÜCRE 4: BPE TOKENIZER EĞİT
# ═══════════════════════════════════════════════════════════

from tokenizers import Tokenizer, trainers, pre_tokenizers, normalizers, decoders, processors
from tokenizers.models import BPE
from tokenizers import AddedToken

SPECIAL_TOKENS = [
    '<|SOURCE|>', '<|QUERY|>', '<|CONTEXT|>', '<|CONCEPTS|>', '<|EDGES|>',
    '<|HOT_GRAPH|>', '<|POLICY|>', '<|ANSWER_PLAN|>', '<|ANSWER|>', '<|INTENT|>',
    '<|DOMAIN|>', '<|SHARD|>', '<|UNCERTAIN|>', '<|SPECULATIVE|>', '<|DO_NOT_CLAIM|>',
    '<|SUPPORTED_BY_SOURCE|>', '<|MISSING_INFO|>', '<|END|>',
    '<|PAD|>', '<|UNK|>', '<|BOS|>', '<|EOS|>'
]

class BPETokenizer:
    def __init__(self, tok):
        self._tok = tok
        self.vocab_size = tok.get_vocab_size()
    def encode(self, text):
        return self._tok.encode(text).ids
    def decode(self, ids):
        return self._tok.decode(ids)

tok_path = Path('/content/drive/MyDrive/KRM-Core/tokenizer.json')
if tok_path.exists():
    tok = Tokenizer.from_file(str(tok_path))
    print(f"Tokenizer yüklendi: {tok_path}")
else:
    print("BPE tokenizer eğitiliyor (vocab=32K)...")
    corpus_files = [str(f) for f in DRIVE_OUT.glob('*.jsonl')]
    print(f"{len(corpus_files)} dosya bulundu")

    tok = Tokenizer(BPE(unk_token='<|UNK|>'))
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

# Test
test_enc = tokenizer.encode("Merhaba dünya, KRM-Core eğitim testi!")
test_dec = tokenizer.decode(test_enc)
print(f"Test: '{test_dec}' ({len(test_enc)} token)")

# ═══════════════════════════════════════════════════════════
# HÜCRE 5: VERİYİ OKU + TOKENIZE ET
# ═══════════════════════════════════════════════════════════

all_ids = []
print(f"\nDrive'dan veriler okunuyor...")
for ds_file in sorted(DRIVE_OUT.glob('*.jsonl')):
    print(f"  {ds_file.stem}...", end=" ")
    count = 0
    with open(ds_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get('text', '')
            if not text:
                continue
            ids = tokenizer.encode(text)
            all_ids.extend(ids)
            all_ids.append(tokenizer.encode('<|END|>')[0])
            count += 1
    print(f"{count:,} örnek, toplam {len(all_ids):,} token")

print(f"\nToplam token: {len(all_ids):,} (~{len(all_ids)/1e6:.1f}M)")
gc.collect()

# ═══════════════════════════════════════════════════════════
# HÜCRE 6: DATASET + MODEL
# ═══════════════════════════════════════════════════════════

class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.seq_len = seq_len
        n = (len(tokens) - 1) // seq_len
        self.data = torch.tensor(tokens[:n * seq_len + 1], dtype=torch.long)
        self.n_samples = n
        print(f"  {n:,} chunk")
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.data[start:start + self.seq_len], self.data[start + 1:start + self.seq_len + 1]

print("Dataset oluşturuluyor...")
dataset = TokenDataset(all_ids, SEQ_LEN)
del all_ids
gc.collect()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# RWKV Model
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
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.r = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.wkv = WKVMemory(d)
        self.ck = nn.Linear(d, d_ff, bias=False)
        self.cv = nn.Linear(d_ff, d, bias=False)
        self.cr = nn.Linear(d, d, bias=False)
    def forward(self, x):
        h = x + torch.sigmoid(self.r(self.ln1(x))) * self.o(self.wkv(self.v(self.ln1(x)), self.k(self.ln1(x))))
        return h + torch.sigmoid(self.cr(self.ln2(h))) * self.cv(F.relu(self.ck(self.ln2(h))) ** 2)

class RWKVLM(nn.Module):
    def __init__(self, vs, d, n, dff):
        super().__init__()
        self.embed = nn.Embedding(vs, d)
        self.blocks = nn.ModuleList([RWKVBlock(d, dff) for _ in range(n)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vs, bias=False)
    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))

model = RWKVLM(tokenizer.vocab_size, cfg['d_model'], cfg['n_layers'], cfg['d_ff']).to(device)
params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {MODEL} | {params:,} param ({params*4/1024/1024:.0f} MB)")

# ═══════════════════════════════════════════════════════════
# HÜCRE 7: EĞİTİM
# ═══════════════════════════════════════════════════════════

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_STEPS)

save_dir = Path(f'/content/drive/MyDrive/KRM-Core/trained_{MODEL}')
save_dir.mkdir(parents=True, exist_ok=True)

step = 0
best_loss = float('inf')
history = {'step': [], 'loss': [], 'ppl': []}

print(f"\n{'='*60}")
print(f"EGİTİM: {MODEL} | {MAX_STEPS} adım | {device}")
print(f"{'='*60}")

t0 = time.time()
model.train()

while step < MAX_STEPS:
    for xb, yb in loader:
        if step >= MAX_STEPS:
            break
        step += 1
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history['step'].append(step)
        history['loss'].append(loss.item())
        history['ppl'].append(math.exp(min(loss.item(), 20)))

        if step % 10 == 0:
            avg = sum(history['loss'][-10:]) / 10
            print(f"  Step {step:5d}/{MAX_STEPS} | Loss: {avg:.4f} | PPL: {math.exp(min(avg, 20)):.2f} | {time.time()-t0:.0f}s")

        if step % SAVE_EVERY == 0:
            torch.save({
                'model_state': model.state_dict(),
                'config': cfg,
                'step': step,
                'best_loss': best_loss,
                'vocab_size': tokenizer.vocab_size,
                'history': history,
            }, save_dir / f'{MODEL}_step{step}.pt')
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), save_dir / f'{MODEL}_best.pt')
            print(f"    [*] Kaydedildi: step {step}")

print(f"\n{'='*60}")
print(f"TAMAMLANDI! {step} adım, {time.time()-t0:.0f}s")
print(f"Best Loss: {best_loss:.4f} (PPL: {math.exp(min(best_loss, 20)):.2f})")

# Son modeli kaydet
torch.save({
    'model_state': model.state_dict(),
    'config': cfg,
    'step': step,
    'best_loss': best_loss,
    'vocab_size': tokenizer.vocab_size,
    'history': history,
}, save_dir / f'{MODEL}_final.pt')
print(f"Kayıtlar: {save_dir}/")

# ═══════════════════════════════════════════════════════════
# HÜCRE 8: ÜRETIM TESTİ
# ═══════════════════════════════════════════════════════════

model.eval()
print(f"\n{'='*60}")
print("ÜRETIM TESTİ")
print(f"{'='*60}")

prompts = [
    "Bir dikdörtgenin alan formülü nedir?",
    "Python'da fibonacci serisi nasıl yazılır?",
    "Türkiye'nin başkenti neresidir?",
    "What is binary search?",
    "Write a prime number checker:",
]

for p in prompts:
    print(f"\n{'─'*50}")
    print(f"Soru: {p}")
    ids = torch.tensor([tokenizer.encode(p)], device=device)
    with torch.no_grad():
        for _ in range(300):
            logits = model(ids[:, -SEQ_LEN:])
            probs = F.softmax(logits[0, -1] / 0.8, dim=0)
            nid = torch.multinomial(probs, 1).item()
            ids = torch.cat([ids, torch.tensor([[nid]], device=device)], dim=1)
            if nid == tokenizer.encode('<|END|>')[0]:
                break
    gen = tokenizer.decode(ids[0].tolist())[len(p):]
    print(f"Cevap: {gen[:500]}")
