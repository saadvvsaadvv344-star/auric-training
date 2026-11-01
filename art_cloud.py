"""Auric's image AI, trained on GitHub's free machines (no graphics card): one ~5.6 h piece per run, then the next run
continues exactly where this one stopped. Same model and method as artgen.py on the PC (64x64 diffusion + judge).

    python art_cloud.py            (in the workflow; needs GITHUB_TOKEN, the photos in the release "data")
"""
import json
import math
import random
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common as C

torch.set_num_threads(int(__import__("os").environ.get("AURIC_THREADS", "4")))
sys.modules["brain"] = types.SimpleNamespace(running_training=lambda: False)     # (the PC's GPU queue doesn't exist here)
import artgen  # noqa: E402

BATCH = int(__import__("os").environ.get("AURIC_BATCH", "32"))
PLANNED = 150_000
CKPT, JUDGE = C.WORK / "art_brain.pt", artgen.JUDGE
artgen.SAMPLES = C.WORK / "samples.png"
artgen.LOG = C.WORK / "train_log.txt"
artgen.ART.mkdir(parents=True, exist_ok=True)
JUDGE.parent.mkdir(parents=True, exist_ok=True)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_photos():
    p = C.WORK / "art_photos.npz"
    if not p.exists() and not C.download("art_photos.npz", p, tag="data"):
        raise SystemExit("the photos aren't in the release 'data'")
    d = np.load(p, allow_pickle=False)
    return d["X"], d["Y"], [str(n) for n in d["names"]]


def main():
    old = C.read_status("art.json") or {}
    C.publish_status({"art.json": {**old, "phase": "starting a new run (loading the photos and the saved state)",
                                   "updated": now()}})
    X, Y, names = load_photos()
    C.download("art_judge.pt", JUDGE)
    if not JUDGE.exists():                                      # first run: the judge (it rates the pictures) learns
        artgen.log("training the judge first (it checks that pictures show what was asked)")
        artgen.load_photos = lambda: (X, Y, names)
        artgen.train_judge(minutes=25)
        C.upload(JUDGE)
    model, ema = artgen.UNet(len(names)), artgen.UNet(len(names))
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), 2e-4, weight_decay=0.01)
    step, best, best_step, total, run = 0, -1.0, 0, PLANNED, 1
    if C.download("art_brain.pt", CKPT):
        s = torch.load(CKPT, map_location="cpu", weights_only=False)
        if s["names"] == names:
            model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"])
            step, best, best_step = s["step"], s.get("best", -1.0), s.get("best_step", s["step"])
            total, run = s.get("total", PLANNED), s.get("run", 0) + 1
    params = sum(p.numel() for p in model.parameters()) / 1e6
    artgen.log(f"run {run}: {params:.1f}M parameters, {len(X):,} photos of {len(names)} kinds, step {step:,}")
    ab = artgen.alphas_bar("cpu")
    t0, start, last_pub, last_save, lr, loss_avg = time.time(), step, 0.0, time.time(), 2e-4, None
    finished = False

    def save():
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                    "names": names, "best": best, "best_step": best_step, "total": total, "run": run}, CKPT)
        C.upload(CKPT)

    def publish(phase):
        sps = (time.time() - t0) / max(1, step - start)
        files = {"art.json": {"phase": phase, "run": run, "step": step, "planned": total, "sec_per_step": round(sps, 2),
                              "loss": None if loss_avg is None else round(loss_avg, 4), "best_quality": best,
                              "best_step": best_step, "kinds": len(names), "photos": int(len(X)), "batch": BATCH,
                              "params_m": round(params, 1),
                              "cloud_hours_left": round((total - step) * sps / 3600, 1), "updated": now()}}
        if artgen.SAMPLES.exists():
            files["art_samples.png"] = artgen.SAMPLES.read_bytes()
            info = artgen.SAMPLES.with_suffix(".json")
            if info.exists():
                files["art_samples.json"] = json.loads(info.read_text(encoding="utf-8"))
        C.publish_status(files)

    model.train()
    while C.time_left() > 900:                                  # (15 min kept for the last save + upload)
        if step >= total:
            if step - best_step <= 30_000:
                total += 50_000
                artgen.log(f"still improving: training on until step {total:,}")
            else:
                finished = True
                break
        b = np.random.randint(0, len(X), BATCH)
        x0 = artgen.to_tensor(X[b], "cpu")
        x0 = torch.where((torch.rand(BATCH) < 0.5)[:, None, None, None], x0.flip(3), x0)
        kind = torch.from_numpy(Y[b])
        kind = torch.where(torch.rand(BATCH) < 0.12, torch.full_like(kind, len(names)), kind)
        t = torch.randint(0, artgen.T, (BATCH,))
        a = ab[t][:, None, None, None]
        noise = torch.randn_like(x0)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        v = a.sqrt() * noise - (1 - a).sqrt() * x0
        loss = F.mse_loss(model(xt, t, kind), v)                # (full precision: half precision is ~37x slower
                                                                #  on a processor - measured 232 s vs 6.3 s per step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 2000) * (0.3 + 0.7 * 0.5 * (1 + math.cos(math.pi * min(step, total) / total)))
        opt.step()
        with torch.no_grad():
            d = min(0.9995, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        step += 1
        loss_avg = loss.item() if loss_avg is None else 0.98 * loss_avg + 0.02 * loss.item()
        if step % 2000 == 0:
            ema.eval(); artgen.save_samples(ema, names, "cpu", step); model.train()
        if step % 10000 == 0:
            ema.eval()
            q = artgen.quality(ema, names, "cpu")
            model.train()
            if q:
                artgen.log(f"QUALITY at step {step:,}: the judge recognises {q[0]:.0%} of the pictures (top-5 {q[1]:.0%})")
                if q[0] > best:
                    best, best_step = q[0], step
                    torch.save({"ema": ema.state_dict(), "names": names, "step": step, "quality": q},
                               C.WORK / "art_brain_best.pt")
                    C.upload(C.WORK / "art_brain_best.pt")
        if time.time() - last_pub > 1200:                       # the website: every 20 minutes
            publish("training")
            last_pub = time.time()
        if time.time() - last_save > 2700:                      # a safety save every 45 minutes
            save()
            last_save = time.time()
    save()
    publish("finished: no better pictures any more" if finished else "between runs (the next one continues)")
    C.history_add("art", {"run": run, "from": start, "to": step, "hours": round((time.time() - C.STARTED) / 3600, 2),
                          "loss": None if loss_avg is None else round(loss_avg, 4), "best_quality": best, "ended": now()})
    artgen.log(f"run {run} ended at step {step:,}")


if __name__ == "__main__":
    main()
