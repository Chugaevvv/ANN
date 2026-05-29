#!/usr/bin/env python3
"""
九章算術語言模型 — 最終版
Strategy: use the prompt-recommended hyperparams, full-text training,
heavier regularization, and produce all required analysis outputs.
"""

import re, os, json, math, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Disable font warnings by using a safe backend
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

rcParams['font.family'] = 'sans-serif'
rcParams['axes.unicode_minus'] = False

from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == "cuda": torch.cuda.manual_seed_all(SEED)
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════

def load_text(path="九章算经.txt"):
    with open(path, "rb") as f: return f.read().decode("gb18030")

def extract_main_chapters(text):
    """Get the 9 main chapters' text ranges."""
    pattern = r'九章算[術术]卷[第第]([一二三四五六七八九])'
    matches = list(re.finditer(pattern, text))
    pos = [m.start() for m in matches]
    nums = [m.group(1) for m in matches]
    # Filter TOC clusters
    is_toc = [False] * len(pos)
    for i in range(1, len(pos)):
        if pos[i] - pos[i-1] < 200:
            is_toc[i] = is_toc[i-1] = True
    real = [(nums[i], pos[i]) for i in range(len(pos)) if not is_toc[i]]
    # Take first 9 unique
    seen, chapters = set(), []
    for n, p in real:
        if n not in seen:
            seen.add(n); chapters.append((n, p))
    chapters.sort(key=lambda x: x[1])
    return chapters

def build_tokenizer(text):
    chars = sorted(set(text))
    specials = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
    vocab = specials + chars
    c2i = {c: i for i, c in enumerate(vocab)}
    i2c = {i: c for c, i in c2i.items()}
    return vocab, c2i, i2c

def extract_qa(text):
    pattern = r'〔[一二三四五六七八九十百０-９0-9]+〕(.*?)荅曰[：:](.*?)(?=〔|　　　　　|術曰|$)'
    pairs = []
    for q, a in re.findall(pattern, text, re.DOTALL):
        qc = re.sub(r'[\r\n\s　]+', '', q).strip()
        ac = re.sub(r'[\r\n\s　]+', '', a).strip()
        if qc and ac: pairs.append({"question": qc, "answer": ac})
    return pairs

class CharDataset(Dataset):
    def __init__(self, text, c2i, max_len=128, stride=64):
        self.max_len = max_len
        tokens = [c2i.get(c, c2i['<UNK>']) for c in text]
        pad = c2i['<PAD>']
        self.data = []
        for start in range(0, len(tokens) - 1, stride):
            chunk = tokens[start:start + max_len + 1]
            if len(chunk) < 8: continue
            if len(chunk) < max_len + 1:
                chunk = chunk + [pad] * (max_len + 1 - len(chunk))
            self.data.append(torch.tensor(chunk, dtype=torch.long))
        print(f"  Dataset: {len(self.data)} windows")

    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        c = self.data[i]; return c[:-1], c[1:]

# ═══════════════════════════════════════════════════════════════
# 2. WORD2VEC
# ═══════════════════════════════════════════════════════════════

def train_w2v(text):
    sents = []
    for line in text.replace('。', '。\n').split('\n'):
        chars = [c for c in line if c.strip()]
        if len(chars) >= 2: sents.append(chars)
    print(f"Word2Vec: {len(sents)} sentences")
    return Word2Vec(sents, vector_size=128, window=5, min_count=2, sg=1, epochs=100, workers=4)

def analyze_w2v(w2v, vocab_set, res_dir="results"):
    os.makedirs(res_dir, exist_ok=True)
    queries = ['步','乘','句','田','尺','畝','數','分']
    with open(f"{res_dir}/w2v_neighbors.txt", "w", encoding="utf-8") as f:
        for q in queries:
            if q not in w2v.wv: continue
            nb = w2v.wv.most_similar(q, topn=10)
            f.write(f"=== {q} ===\n")
            for w, s in nb: f.write(f"  {w}: {s:.3f}\n")
            f.write("\n")
            print(f"  '{q}' -> {[w for w,_ in nb[:5]]}")

    # t-SNE
    cats = {
        'Numerals': list('一二三四五六七八九十百千萬兩'),
        'Measures': list('步畝頃里尺寸斗升斤兩斛秉'),
        'Arithmetic': list('乘除減益損加得為'),
        'Geometry': list('句股弦方面圓徑周'),
    }
    words, labels, cats_ = [], [], []
    for cat, ws in cats.items():
        for w in ws:
            if w in w2v.wv:
                words.append(w); labels.append(w); cats_.append(cat)
    if len(words) >= 10:
        vec = np.array([w2v.wv[w] for w in words])
        tsne = TSNE(2, random_state=42, perplexity=min(5, len(words)-1))
        v2d = tsne.fit_transform(vec)
        fig, ax = plt.subplots(figsize=(10, 8))
        for cat in sorted(set(cats_)):
            m = [c == cat for c in cats_]
            if not any(m): continue
            ax.scatter(v2d[m, 0], v2d[m, 1], label=cat, s=80)
            for i, ok in enumerate(m):
                if ok: ax.annotate(labels[i], (v2d[i,0], v2d[i,1]), fontsize=9)
        ax.set_title("t-SNE — Word2Vec Character Embeddings")
        ax.legend()
        fig.savefig(f"{res_dir}/w2v_tsne.png", dpi=150)
        plt.close(fig); print(f"Saved w2v_tsne.png")

    # Build weight matrix
    specials = ['<PAD>','<BOS>','<EOS>','<UNK>']
    full_vocab = specials + sorted(vocab_set)
    weights = np.zeros((len(full_vocab), 128), dtype=np.float32)
    for i, ch in enumerate(full_vocab):
        if ch in w2v.wv: weights[i] = w2v.wv[ch]
        elif ch not in specials: weights[i] = np.random.randn(128) * 0.02
    np.save(f"{res_dir}/w2v_weights.npy", weights)
    return weights

# ═══════════════════════════════════════════════════════════════
# 3. MINIGPT
# ═══════════════════════════════════════════════════════════════

class GPTBlock(nn.Module):
    def __init__(self, d, heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout),
                               nn.Linear(d_ff, d), nn.Dropout(dropout))
    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)[0]
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, V, d=128, heads=4, layers=4, d_ff=512, max_len=256, dropout=0.15):
        super().__init__()
        self.d = d; self.max_len = max_len
        self.tok = nn.Embedding(V, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([GPTBlock(d, heads, d_ff, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.tok.weight  # tie
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=DEVICE).unsqueeze(0)
        h = self.tok(x) * math.sqrt(self.d) + self.pos(pos)
        h = self.drop(h)
        mask = torch.triu(torch.ones(T, T, device=DEVICE) * float('-inf'), 1)
        for blk in self.blocks: h = blk(h, mask)
        return self.head(self.ln(h))
    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

class LSTM_LM(nn.Module):
    def __init__(self, V, emb=256, hid=256, layers=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, layers, batch_first=True, dropout=dropout if layers>1 else 0)
        self.head = nn.Linear(hid, V)
        self.head.weight = self.emb.weight
    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.head(h)
    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ═══════════════════════════════════════════════════════════════
# 4. TRAINING
# ═══════════════════════════════════════════════════════════════

def train_epoch(m, dl, opt, scaler, V):
    m.train(); total, n = 0, 0
    for x, y in (pbar := tqdm(dl, desc="Train", leave=False)):
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        with autocast():
            logits = m(x)
            loss = F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=0)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        nt = (y != 0).sum().item()
        total += loss.item() * nt; n += nt
        pbar.set_postfix({'loss': f'{loss.item():.3f}'})
    return total / max(n, 1)

@torch.no_grad()
def evaluate(m, dl, V):
    m.eval(); total, n = 0, 0
    for x, y in tqdm(dl, desc="Eval", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with autocast():
            loss = F.cross_entropy(m(x).view(-1, V), y.view(-1), ignore_index=0)
        nt = (y != 0).sum().item()
        total += loss.item() * nt; n += nt
    avg = total / max(n, 1)
    return avg, math.exp(avg)

def train_model(m, tr, va, epochs, lr, V, label, ckpt="checkpoints"):
    os.makedirs(ckpt, exist_ok=True)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05)
    # Linear warmup + cosine decay
    warmup = max(1, epochs // 10)
    def get_lr(e):
        if e < warmup: return lr * (e + 1) / warmup
        return lr * 0.5 * (1 + math.cos(math.pi * (e - warmup) / (epochs - warmup)))
    scaler = GradScaler()
    hist = {'train': [], 'val': [], 'ppl': []}
    best_ppl = float('inf')
    for ep in range(epochs):
        for g in opt.param_groups: g['lr'] = get_lr(ep)
        tl = train_epoch(m, tr, opt, scaler, V)
        vl, vp = evaluate(m, va, V)
        hist['train'].append(tl); hist['val'].append(vl); hist['ppl'].append(vp)
        if (ep+1) % 20 == 0 or ep == 0:
            print(f"  Ep {ep+1}/{epochs} | Train loss: {tl:.4f} | Val loss: {vl:.4f} | Val PPL: {vp:.1f}")
        if vp < best_ppl:
            best_ppl = vp
            torch.save({'ep': ep, 'sd': m.state_dict(), 'ppl': vp}, f"{ckpt}/best_{label}.pt")
    ck = torch.load(f"{ckpt}/best_{label}.pt", weights_only=False)
    m.load_state_dict(ck['sd'])
    print(f"  Best {label} PPL: {ck['ppl']:.1f} (epoch {ck['ep']+1})")
    return m, hist

# ═══════════════════════════════════════════════════════════════
# 5. GENERATION
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def gen(model, prompt, c2i, i2c, max_new=30, temp=0.7):
    model.eval()
    tokens = [c2i.get(c, c2i['<UNK>']) for c in prompt]
    ids = torch.tensor([tokens], device=DEVICE, dtype=torch.long)
    for _ in range(max_new):
        if ids.size(1) > model.max_len: ids = ids[:, -model.max_len:]
        logits = model(ids)[:, -1, :] / max(temp, 1e-8)
        v, _ = torch.topk(logits, min(50, logits.size(-1)))
        logits[logits < v[:, -1:]] = float('-inf')
        probs = F.softmax(logits, -1)
        nxt = torch.multinomial(probs, 1).item()
        if nxt in (c2i['<PAD>'], c2i['<EOS>']): break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEVICE)], 1)
    return ''.join(i2c.get(t, '') for t in ids[0].tolist())

# ═══════════════════════════════════════════════════════════════
# 6. PLOTS
# ═══════════════════════════════════════════════════════════════

def plot_curves(histories, labels, path="results/curves.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['tab:blue', 'tab:red']
    for (h, lb, c) in zip(histories, labels, colors):
        eps = range(1, len(h['train'])+1)
        a1.plot(eps, h['train'], color=c, alpha=0.6, lw=1, label=f'{lb} train')
        a1.plot(eps, h['val'], color=c, lw=2, label=f'{lb} val')
        a2.plot(eps, h['ppl'], color=c, lw=2, label=lb)
    a1.set_xlabel('Epoch'); a1.set_ylabel('Loss'); a1.set_title('Loss'); a1.legend(); a1.grid(alpha=0.3)
    a2.set_xlabel('Epoch'); a2.set_ylabel('PPL'); a2.set_title('Validation Perplexity'); a2.legend(); a2.grid(alpha=0.3)
    a2.set_yscale('log')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved {path}")

# ═══════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    BASE = os.path.dirname(os.path.abspath(__file__))
    RES = os.path.join(BASE, "results"); CKPT = os.path.join(BASE, "checkpoints")
    os.makedirs(RES, exist_ok=True); os.makedirs(CKPT, exist_ok=True)

    print("=" * 60)
    print("  九章算術語言模型 — 最終版")
    print("=" * 60)

    # -- Load --
    print("\n[1] Data Loading")
    text = load_text(os.path.join(BASE, "九章算经.txt"))
    chapters = extract_main_chapters(text)

    # Train: ch1-7 + annotations for more data
    if len(chapters) >= 9:
        train_raw = text[chapters[0][1]:chapters[7][1]]
        val_raw = text[chapters[7][1]:chapters[8][1]]
        test_raw = text[chapters[8][1]:]
        # Cut test_raw before annotated section
        next_ch = [p for _, p in chapters if p > chapters[8][1]]
        if next_ch: test_raw = text[chapters[8][1]:next_ch[0]]
    else:
        n = len(text); train_raw = text[:int(.8*n)]; val_raw = text[int(.8*n):int(.9*n)]; test_raw = text[int(.9*n):]

    print(f"  Train: {len(train_raw)} chars | Val: {len(val_raw)} | Test: {len(test_raw)}")

    vocab, c2i, i2c = build_tokenizer(text)
    V = len(vocab)
    MAX_LEN, STRIDE, BS = 128, 48, 32

    tr_ds = CharDataset(train_raw, c2i, MAX_LEN, STRIDE)
    va_ds = CharDataset(val_raw, c2i, MAX_LEN, MAX_LEN//2)
    te_ds = CharDataset(test_raw, c2i, MAX_LEN, MAX_LEN//2)

    tr_dl = DataLoader(tr_ds, BS, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, BS, shuffle=False, num_workers=2, pin_memory=True)
    te_dl = DataLoader(te_ds, BS, shuffle=False, num_workers=2, pin_memory=True)

    # QA pairs for evaluation
    train_qa = extract_qa(train_raw)
    test_qa = extract_qa(test_raw)
    print(f"  QA pairs: train={len(train_qa)}, test={len(test_qa)}")

    # -- Word2Vec --
    print("\n[2] Word2Vec")
    w2v = train_w2v(text)
    w2v_w = analyze_w2v(w2v, set(text), RES)

    # -- MiniGPT --
    print("\n[3] MiniGPT")
    gpt = MiniGPT(V, d=128, heads=4, layers=4, d_ff=512, max_len=MAX_LEN, dropout=0.15).to(DEVICE)
    # Init from Word2Vec
    with torch.no_grad():
        gpt.tok.weight.data[:len(w2v_w), :] = torch.from_numpy(w2v_w)
    print(f"  Params: {gpt.n_params():,}")

    gpt, gpt_h = train_model(gpt, tr_dl, va_dl, 120, 3e-4, V, "gpt", CKPT)
    gpt_tl, gpt_ppl = evaluate(gpt, te_dl, V)
    print(f"  GPT Test PPL: {gpt_ppl:.1f}")

    # -- LSTM --
    print("\n[4] LSTM Baseline")
    lstm = LSTM_LM(V, emb=256, hid=256, layers=2, dropout=0.3).to(DEVICE)
    print(f"  Params: {lstm.n_params():,}")
    lstm, lstm_h = train_model(lstm, tr_dl, va_dl, 80, 3e-4, V, "lstm", CKPT)
    lstm_tl, lstm_ppl = evaluate(lstm, te_dl, V)
    print(f"  LSTM Test PPL: {lstm_ppl:.1f}")

    # -- Experiments --
    print("\n[5] Qualitative Evaluation")

    print("\n  [A] Problem Completion")
    prompts = [
        "今有田廣十五步，從十六步。問為田幾何？荅曰：",
        "今有句三尺，股四尺，問為弦幾何？荅曰：",
        "今有句五尺，股十二尺，問為弦幾何？荅曰：",
        "今有三分之一，五分之二。問合之得幾何？荅曰：",
        "今有上禾三秉，中禾二秉，下禾一秉，實三十九斗；上禾二秉，中禾三秉，下禾一秉，實三十四斗。問上禾一秉幾何？荅曰：",
    ]
    comps = []
    for p in prompts:
        g = gen(gpt, p, c2i, i2c, max_new=25, temp=0.5)
        ans = g[g.find('荅曰：')+4:] if '荅曰：' in g else g
        comps.append({'prompt': p, 'gen': g})
        print(f"    Q: {p[:40]}...")
        print(f"    A (gen): {ans[:50]}")

    print("\n  [B] Cross-Chapter Accuracy")
    if test_qa:
        correct = 0
        for pair in test_qa:
            prompt = pair['question'] + '荅曰：'
            g = gen(gpt, prompt, c2i, i2c, max_new=25, temp=0.3)
            pred = g[g.find('荅曰：')+4:] if '荅曰：' in g else g[len(prompt):]
            pred = pred.rstrip('。；，、 \t\n\r')
            true = pair['answer'].rstrip('。；，、 \t')
            if pred == true: correct += 1
        acc = correct / len(test_qa)
        print(f"    Accuracy: {correct}/{len(test_qa)} = {acc:.1%}")
        # Show first 3
        for pair in test_qa[:3]:
            prompt = pair['question'] + '荅曰：'
            g = gen(gpt, prompt, c2i, i2c, max_new=20, temp=0.3)
            pred = g[g.find('荅曰：')+4:] if '荅曰：' in g else g
            print(f"    Q: {pair['question'][:40]}")
            print(f"    True: {pair['answer'][:40]}")
            print(f"    Pred: {pred[:40]}")
    else:
        acc = 0.0

    print("\n  [C] Temperature Comparison")
    sample = (test_qa[0]['question'] if test_qa else "今有句三尺股四尺問為弦幾何") + '荅曰：'
    for t in [0.3, 0.7, 1.0, 1.5]:
        g = gen(gpt, sample, c2i, i2c, max_new=25, temp=t)
        ans = g[g.find('荅曰：')+4:] if '荅曰：' in g else g
        print(f"    T={t:.1f}: {ans[:50]}")

    # -- Plots --
    print("\n[6] Generating Plots")
    plot_curves([gpt_h, lstm_h], ['MiniGPT', 'LSTM'], f"{RES}/curves.png")

    # -- Summary --
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    summary = {
        'Vocab': V, 'GPT Params': gpt.n_params(), 'LSTM Params': lstm.n_params(),
        'Random PPL': V, 'GPT Test PPL': f"{gpt_ppl:.1f}", 'LSTM Test PPL': f"{lstm_ppl:.1f}",
        'Test Acc': f"{acc:.1%}", 'Train QA': len(train_qa), 'Test QA': len(test_qa),
        'Train Chars': len(train_raw), 'Train Windows': len(tr_ds),
    }
    for k, v in summary.items(): print(f"  {k}: {v}")
    with open(f"{RES}/summary.json", "w", encoding="utf-8") as f:
        json.dump({k: str(v) for k, v in summary.items()}, f, ensure_ascii=False, indent=2)
    with open(f"{RES}/completions.txt", "w", encoding="utf-8") as f:
        for c in comps: f.write(f"Prompt: {c['prompt']}\nGen: {c['gen']}\n\n")
    print(f"\nAll results in {RES}/"); print("Done!")

if __name__ == "__main__":
    main()
