"""Auric's OWN image AI, built and trained from zero: a small diffusion model (64x64, black & white) that draws one of
~200 kinds of things from a prompt ("a cat", "a red car", "mountains").

How it works: in training, a photo gets noise added and the network learns to remove the noise, knowing WHAT the photo
shows. To create, it starts from pure noise and removes it step by step toward what was asked ("guidance" makes it
follow the prompt more strongly). A separate small JUDGE (also from zero, trained on the real photos) checks that the
pictures really look like what was asked: that's the quality score, no human has to look at test pictures.

    python artgen.py train          (days, pausable: auric-stop or close the window; run again to resume)
    python artgen.py judge          (trains the judge, ~10 min)
"""
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
ART = ROOT / "data" / "art"
CKPT, JUDGE = ROOT / "checkpoints" / "art_brain.pt", ROOT / "checkpoints" / "art_judge.pt"
LOG, STATUS = ART / "train_log.txt", ART / "train_status.json"
STOP = ROOT / "STOP"
SIZE, T = 64, 1000


# ---------------------------------------------------------------- data
def load_photos():
    kinds = json.loads((ART / "kinds.json").read_text(encoding="utf-8"))
    names = sorted(k for k, v in kinds.items() if v["photos"] >= 40 and (ART / f"photos_{k.replace(' ', '_')}.npy").exists())
    xs, ys = [], []
    for i, k in enumerate(names):
        a = np.load(ART / f"photos_{k.replace(' ', '_')}.npy")
        xs.append(a)
        ys.append(np.full(len(a), i, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys), names


# ---------------------------------------------------------------- the network (U-Net)
def time_embed(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class Res(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(emb, cout * 2)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.e(e)[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift                   # what to draw + how noisy: steers every block
        return self.skip(x) + self.c2(F.silu(h))


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n, self.qkv, self.out = nn.GroupNorm(8, c), nn.Conv2d(c, c * 3, 1), nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, 4, C // 4, H * W).unbind(1)
        y = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.out(y.transpose(-1, -2).reshape(B, C, H, W))


class UNet(nn.Module):
    def __init__(self, n_kinds, ch=(64, 128, 256, 256), emb=256):
        super().__init__()
        self.n_kinds = n_kinds
        self.temb = nn.Sequential(nn.Linear(128, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.kemb = nn.Embedding(n_kinds + 1, emb)              # +1: "nothing in particular" (for guidance)
        self.inp = nn.Conv2d(1, ch[0], 3, padding=1)
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        c, skips = ch[0], [ch[0]]
        for i, co in enumerate(ch):
            for _ in range(2):
                self.down.append(nn.ModuleList([Res(c, co, emb), Attn(co) if i >= 2 else nn.Identity()]))
                c = co
                skips.append(c)
            if i < len(ch) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1)]))
                skips.append(c)
        self.mid = nn.ModuleList([Res(c, c, emb), Attn(c), Res(c, c, emb)])
        for i, co in reversed(list(enumerate(ch))):
            for _ in range(3):
                self.up.append(nn.ModuleList([Res(c + skips.pop(), co, emb), Attn(co) if i >= 2 else nn.Identity()]))
                c = co
            if i > 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1)]))
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t, kind):
        e = self.temb(time_embed(t, 128)) + self.kemb(kind)
        h = self.inp(x)
        hs = [h]
        for blk in self.down:
            if len(blk) == 1:
                h = blk[0](h)
            else:
                h = blk[1](blk[0](h, e))
            hs.append(h)
        h = self.mid[2](self.mid[1](self.mid[0](h, e)), e)
        for blk in self.up:
            if isinstance(blk[0], nn.Upsample):
                h = blk[1](blk[0](h))
            else:
                h = blk[1](blk[0](torch.cat([h, hs.pop()], 1), e))
        return self.out(h)


# ---------------------------------------------------------------- noise schedule (cosine), v-prediction
def alphas_bar(device):
    s = 0.008
    t = torch.linspace(0, T, T + 1, device=device) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return (f / f[0]).clamp(1e-5, 1)[1:]


@torch.no_grad()
def generate(model, kinds, device, steps=50, guidance=3.0, seed=None):
    """Pictures from noise (DDIM, deterministic), following `kinds` (list of kind indices). Returns uint8 Nx64x64."""
    g = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    ab = alphas_bar(device)
    n = len(kinds)
    x = torch.randn(n, 1, SIZE, SIZE, device=device, generator=g)
    k = torch.tensor(kinds, device=device)
    none = torch.full_like(k, model.n_kinds)
    ts = torch.linspace(T - 1, 0, steps, device=device).long()
    for i, t in enumerate(ts):
        a = ab[t]
        tt = torch.full((n,), int(t), device=device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            v = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([k, none])).float()
        v_k, v_0 = v.chunk(2)
        v = v_0 + guidance * (v_k - v_0)                      # follow the prompt more strongly
        x0 = (a.sqrt() * x - (1 - a).sqrt() * v).clamp(-1, 1)
        eps = ((1 - a).sqrt() * x + a.sqrt() * v)
        a_next = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
        x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().squeeze(1).cpu().numpy()


# ---------------------------------------------------------------- the judge: does this picture show what was asked?
class Judge(nn.Module):
    def __init__(self, n_kinds):
        super().__init__()
        c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.SiLU())
        self.f = nn.Sequential(c(1, 48), c(48, 48), nn.MaxPool2d(2), c(48, 96), c(96, 96), nn.MaxPool2d(2),
                               c(96, 192), c(192, 192), nn.MaxPool2d(2), c(192, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                               nn.Dropout(0.3), nn.Linear(256, n_kinds))

    def forward(self, x):
        return self.f(x)


def to_tensor(a, device):
    return torch.from_numpy(a).float().div(127.5).sub(1).unsqueeze(1).to(device)


def train_judge(minutes=12):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from brain import running_training
    while running_training():                                   # one training at a time on the graphics card
        print("waiting: Auric's other training is using the graphics card...", flush=True)
        time.sleep(120)
    X, Y, names = load_photos()
    idx = np.random.default_rng(0).permutation(len(X))
    test, tr = idx[:2000], idx[2000:]
    m = Judge(len(names)).to(device)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=0.05)
    t0, step = time.time(), 0
    while time.time() - t0 < minutes * 60:
        b = np.random.choice(tr, 128)
        x = to_tensor(X[b], device)
        if random.random() < 0.5:
            x = x.flip(3)
        x = x + 0.05 * torch.randn_like(x)                      # a little noise: generated pictures aren't perfect
        loss = F.cross_entropy(m(x), torch.from_numpy(Y[b]).to(device), label_smoothing=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
    m.eval()
    with torch.no_grad():
        acc = sum((m(to_tensor(X[test[i:i + 256]], device)).argmax(1).cpu().numpy() == Y[test[i:i + 256]]).sum()
                  for i in range(0, len(test), 256)) / len(test)
    torch.save({"model": m.state_dict(), "names": names, "acc": float(acc)}, JUDGE)
    print(f"judge: {step} steps, recognises real photos {acc:.0%} of the time ({len(names)} kinds; chance = {1 / len(names):.1%})")
    return acc


@torch.no_grad()
def quality(model, names, device, per_kind=2, n_kinds=60):
    """The judge looks at freshly generated pictures: how often is it what was asked? (+ top-5)"""
    if not JUDGE.exists():
        return None
    j = torch.load(JUDGE, map_location="cpu", weights_only=False)
    if j["names"] != names:
        return None
    judge = Judge(len(names)).to(device)
    judge.load_state_dict(j["model"])
    judge.eval()
    kinds = [k for k in random.Random(1).sample(range(len(names)), min(n_kinds, len(names))) for _ in range(per_kind)]
    imgs = generate(model, kinds, device, steps=30, seed=3)
    logits = judge(to_tensor(imgs, device))
    top1 = (logits.argmax(1).cpu().numpy() == np.array(kinds)).mean()
    top5 = np.mean([k in row for k, row in zip(kinds, logits.topk(5, 1).indices.cpu().numpy())])
    return float(top1), float(top5)


# ---------------------------------------------------------------- training (days, resumable)
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def train(total_steps=150_000, batch=48, lr=2e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from brain import running_training
    while running_training():                                   # one training at a time on the graphics card
        STATUS.write_text(json.dumps({"step": 0, "total": total_steps, "loss": None, "hours_left": 0, "best": -1,
                                      "waiting": True, "updated": datetime.now().strftime("%H:%M")}), encoding="utf-8")
        print("waiting: Auric's other training is using the graphics card...", flush=True)
        time.sleep(120)
    torch.backends.cudnn.benchmark = True                       # 0.86 s -> 0.23 s per step on the RTX 5060
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    X, Y, names = load_photos()
    model = UNet(len(names)).to(device)
    ema = UNet(len(names)).to(device)
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.01)
    step, best = 0, -1.0
    if CKPT.exists():
        s = torch.load(CKPT, map_location="cpu", weights_only=False)
        if s["names"] == names:
            model.load_state_dict(s["model"])
            ema.load_state_dict(s["ema"])
            opt.load_state_dict(s["opt"])
            step, best = s["step"], s.get("best", -1.0)
            log(f"resuming at step {step:,}")
        else:
            log("the photo collection changed: starting a new image brain")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    total_steps = max(total_steps, s.get("total", total_steps)) if CKPT.exists() and s["names"] == names else total_steps
    best_step = s.get("best_step", step) if CKPT.exists() and s["names"] == names else step
    log(f"Auric's image AI: {n_params:.1f}M parameters, {len(X):,} photos of {len(names)} kinds, step {step:,}/{total_steps:,}")
    ab = alphas_bar(device)
    t0, last_save, start = time.time(), time.time(), step
    yielded = False
    model.train()
    while not STOP.exists():
        if step >= total_steps:                                  # the planned end: go on while it still improves
            if step - best_step <= 30_000:
                total_steps += 50_000
                log(f"still improving (best quality at step {best_step:,}): training on until step {total_steps:,}")
            else:
                log(f"no better pictures since step {best_step:,}: the image AI is as good as this training makes it")
                break
        if step % 500 == 0 and step > start and _should_yield():  # JARVIS opened / another training waits: step aside
            yielded = True
            break
        b = np.random.randint(0, len(X), batch)
        x0 = to_tensor(X[b], device)
        flip = torch.rand(batch, device=device) < 0.5
        x0 = torch.where(flip[:, None, None, None], x0.flip(3), x0)
        kind = torch.from_numpy(Y[b]).to(device)
        kind = torch.where(torch.rand(batch, device=device) < 0.12, torch.full_like(kind, len(names)), kind)  # guidance
        t = torch.randint(0, T, (batch,), device=device)
        a = ab[t][:, None, None, None]
        noise = torch.randn_like(x0)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        v = a.sqrt() * noise - (1 - a).sqrt() * x0
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = F.mse_loss(model(xt, t, kind).float(), v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:                               # warm up, then a slow decay
            g["lr"] = lr * min(1.0, (step + 1) / 2000) * (0.3 + 0.7 * 0.5 * (1 + math.cos(math.pi * step / total_steps)))
        opt.step()
        with torch.no_grad():
            d = min(0.9995, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        step += 1
        if step % 200 == 0:
            el = time.time() - t0
            left = (total_steps - step) * el / max(1, step - start) / 3600
            STATUS.write_text(json.dumps({"step": step, "total": total_steps, "loss": round(loss.item(), 4),
                                          "hours_left": round(left, 1), "best": best,
                                          "updated": datetime.now().strftime("%H:%M")}), encoding="utf-8")
        if step % 1000 == 0:
            log(f"step {step:,}/{total_steps:,} | error {loss.item():.4f} | {(time.time() - t0) / max(1, step - start):.2f}s/step")
        if step % 2000 == 0:                                      # what it draws now (the live training page shows it)
            ema.eval()
            save_samples(ema, names, device, step)
            model.train()
        if step % 10000 == 0:
            ema.eval()
            q = quality(ema, names, device)
            if q:
                log(f"QUALITY at step {step:,}: the judge recognises {q[0]:.0%} of my pictures (top-5: {q[1]:.0%})")
                if q[0] > best:
                    best, best_step = q[0], step
                    torch.save({"ema": ema.state_dict(), "names": names, "step": step, "quality": q},
                               CKPT.with_name("art_brain_best.pt"))
        if step % 2000 == 0 or time.time() - last_save > 1200:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                        "names": names, "best": best, "best_step": best_step, "total": total_steps}, CKPT)
            last_save = time.time()
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                "names": names, "best": best, "best_step": best_step, "total": total_steps}, CKPT)
    STOP.unlink(missing_ok=True)
    if yielded:
        log(f"stepped aside at step {step:,} (JARVIS mode or another training needs the graphics card): continues later")
        STATUS.write_text(json.dumps({"step": step, "total": total_steps, "loss": None, "hours_left": None, "best": best,
                                      "waiting": True, "updated": datetime.now().strftime("%H:%M")}), encoding="utf-8")
        sys.exit(3)
    log(f"paused / finished at step {step:,}")


def _should_yield():
    try:
        import switch
        return switch.jarvis_open() or bool(switch.gpu_waiters("art-train"))
    except Exception:
        return False


SAMPLES = ART / "samples_latest.png"
SHOW = ["cat", "dog", "car", "tree", "house", "flower", "bird", "horse", "mountain", "boat", "apple", "chair",
        "airplane", "fish", "bicycle", "cake"]


@torch.no_grad()
def save_samples(model, names, device, step):
    """16 pictures of well-known things, in a 4x4 sheet with their names (data/art/samples_latest.png + .json)."""
    try:
        from PIL import Image
        kinds = [names.index(k) for k in SHOW if k in names][:16]
        kinds += [i for i in random.Random(7).sample(range(len(names)), 16) if i not in kinds][:16 - len(kinds)]
        imgs = generate(model, kinds, device, steps=30, seed=11)
        sheet = Image.new("L", (4 * 66, 4 * 66), 20)
        for n, a in enumerate(imgs):
            sheet.paste(Image.fromarray(np.asarray(a, dtype=np.uint8)).resize((64, 64)), ((n % 4) * 66 + 1, (n // 4) * 66 + 1))
        sheet.resize((4 * 66 * 2, 4 * 66 * 2), Image.NEAREST).save(SAMPLES, optimize=True)
        SAMPLES.with_suffix(".json").write_text(json.dumps({"step": step, "kinds": [names[k] for k in kinds],
                                                            "time": datetime.now().strftime("%H:%M")}), encoding="utf-8")
    except Exception as e:
        log(f"(couldn't save sample pictures: {type(e).__name__}: {e})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "judge":
        train_judge()
    else:
        train()
