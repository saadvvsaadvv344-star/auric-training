"""Auric v2: the 1.5B base (Qwen2.5-1.5B-Instruct, Apache 2.0) learns everything that is Auric (data/v2/train.jsonl:
identity, Saad's lessons, the teacher's 20,000 answers, answers written from sources, other languages) with LoRA
on this PC's graphics card, then the result is merged into ONE brain: models/auric-v2.

The exam: 300 examples never studied (data/v2/exam.jsonl). Auric v2 is kept only if it does better than the base.

    python switch.py start v2train      (hidden job; report: data/v2/train_report.txt)
"""
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "models" / "qwen2.5-1.5b-instruct"
OUT = ROOT / "models" / "auric-v2"
ADAPTER = ROOT / "models" / "auric-v2-adapter"      # the LoRA part alone (small): saved during training
DATA = ROOT / "data" / "v2"
STATUS, REPORT = DATA / "train_status.json", DATA / "train_report.txt"
MAX_LEN, MICRO, ACCUM, LR, EPOCHS = 1024, 2, 8, 2e-4, 1
SAMPLES = [("Saad", "hello"), ("Saad", "who are you?"), ("Amina", "what's my name?"),
           ("Saad", "quelle est la capitale du Maroc ?"), ("Saad", "¿por qué el cielo es azul?"),
           ("Saad", "what is the command to install wsl?"), ("Saad", "who was the first person to reach America?")]


def status(**kw):
    s = {}
    try:
        s = json.loads(STATUS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    s.update(kw, updated=time.strftime("%H:%M"))
    STATUS.write_text(json.dumps(s, indent=1), encoding="utf-8")


def allowed():
    try:
        import switch
        return switch.is_on() and not (ROOT / "STOP").exists()
    except Exception:
        return True


def encode(tok, ex):
    """Chat -> token ids, with the loss only on Auric's answer."""
    msgs = ex["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = prompt + msgs[-1]["content"] + "<|im_end|>\n"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    if len(f_ids) > MAX_LEN:                                # too long: keep the end (the question and the answer)
        cut = len(f_ids) - MAX_LEN
        if cut >= len(p_ids) - 8:
            return None
        f_ids, p_len = f_ids[cut:], len(p_ids) - cut
    else:
        p_len = len(p_ids)
    labels = [-100] * p_len + f_ids[p_len:]
    return f_ids, labels


def batches(items, rng):
    items = sorted(items, key=lambda x: len(x[0]))           # similar lengths together: less padding, faster
    groups = [items[i:i + MICRO] for i in range(0, len(items), MICRO)]
    rng.shuffle(groups)
    return groups


def collate(group, pad, device):
    import torch
    L = max(len(i) for i, _ in group)
    ids = torch.full((len(group), L), pad, dtype=torch.long)
    lab = torch.full((len(group), L), -100, dtype=torch.long)
    att = torch.zeros((len(group), L), dtype=torch.long)
    for k, (i, l) in enumerate(group):
        ids[k, :len(i)] = torch.tensor(i)
        lab[k, :len(l)] = torch.tensor(l)
        att[k, :len(i)] = 1
    return ids.to(device), lab.to(device), att.to(device)


def exam_loss(model, tok, exam, device):
    import torch
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(exam), 4):
            ids, lab, att = collate(exam[i:i + 4], tok.pad_token_id, device)
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            tot += out.loss.item()
            n += 1
    model.train()
    return tot / max(n, 1)


def samples(model, tok, device):
    import torch
    sys.path.insert(0, str(ROOT))
    from v2data import system
    model.eval()
    out = []
    for name, q in SAMPLES:
        text = tok.apply_chat_template([{"role": "system", "content": system(name)}, {"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
        out.append(f"[{name}] {q}\n  -> {tok.decode(g[0, ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()}")
    model.train()
    return out


def main():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os
    smoke = os.environ.get("V2_SMOKE") == "1"               # a quick check of this program (CPU, a few examples, nothing saved)
    device = "cpu" if smoke else "cuda"
    rng = random.Random(3)
    tok = AutoTokenizer.from_pretrained(BASE)
    if smoke:
        rows = [json.loads(l) for l in (DATA / "rag.jsonl").read_text(encoding="utf-8").splitlines()[:12]]
        train = [e for e in (encode(tok, r) for r in rows[:8]) if e]
        exam = [e for e in (encode(tok, r) for r in rows[8:]) if e]
        global SAMPLES, ACCUM, status
        SAMPLES, ACCUM = SAMPLES[:1], 2
        status = lambda **kw: print("status:", kw)          # noqa: E731 (the real status file isn't touched)
    else:
        status(phase="preparing the examples")
        train = [e for e in (encode(tok, json.loads(l)) for l in (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()) if e]
        exam = [e for e in (encode(tok, json.loads(l)) for l in (DATA / "exam.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()) if e]
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    before = exam_loss(model, tok, exam, device)
    report = [f"Auric v2 training: {len(train):,} examples, exam {len(exam)}", f"BEFORE (base): exam loss {before:.4f}"]
    report += samples(model, tok, device)
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    groups = batches(train, rng) * EPOCHS                    # (same seed: the same order after a restart)
    total = math.ceil(len(groups) / ACCUM)
    step, t0, run = 0, time.time(), 0.0
    progress = ADAPTER / "progress.json"
    if not smoke and progress.exists() and (ADAPTER / "adapter_model.safetensors").exists():
        try:                                                 # the PC restarted: continue from the last saved point
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            saved = json.loads(progress.read_text(encoding="utf-8"))
            if saved.get("total") == total:
                set_peft_model_state_dict(model, load_file(str(ADAPTER / "adapter_model.safetensors"), device=device))
                step = int(saved["step"])
                report.append(f"continued from step {step} (after a restart)")
        except Exception as e:
            report.append(f"couldn't continue from the saved point ({type(e).__name__}): starting over")
            step = 0
    first = step
    model.train()
    for gi, g in enumerate(groups):
        if gi < first * ACCUM:                               # already learned before the restart
            continue
        if gi % 200 == 0 and not allowed():
            report.append(f"paused at step {step}/{total} (switched off): nothing saved")
            REPORT.write_text("\n".join(report), encoding="utf-8")
            status(phase="paused")
            return
        ids, lab, att = collate(g, tok.pad_token_id, device)
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / ACCUM
        loss.backward()
        run = 0.98 * run + 0.02 * loss.item() * ACCUM if gi else loss.item() * ACCUM
        if (gi + 1) % ACCUM == 0:
            lr = LR * min(1, step / 30) * 0.5 * (1 + math.cos(math.pi * step / total))
            for pg in opt.param_groups:
                pg["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                el = time.time() - t0
                status(phase="learning", step=step, total=total, loss=round(run, 4),
                       hours_left=round(el / max(1, step - first) * (total - step) / 3600, 2))
            if step % 300 == 0 and not smoke:              # progress kept: a restart or crash can't lose the hours
                model.save_pretrained(ADAPTER)
                progress.write_text(json.dumps({"step": step, "total": total, "examples": len(train)}), encoding="utf-8")
    if not smoke:
        model.save_pretrained(ADAPTER)                     # the learned part, before the exam and the merge
    after = exam_loss(model, tok, exam, device)
    report.append(f"AFTER: exam loss {after:.4f} ({'better' if after < before else 'NOT better'} than the base {before:.4f})")
    report += samples(model, tok, device)
    current = None                                         # the Auric v2 in use now: the new one must beat IT, too
    if not smoke and (OUT / "config.json").exists():
        try:
            model.cpu()
            torch.cuda.empty_cache()
            cur = AutoModelForCausalLM.from_pretrained(OUT, dtype=torch.bfloat16).to(device)
            current = exam_loss(cur, tok, exam, device)
            del cur
            torch.cuda.empty_cache()
            model.to(device)
            report.append(f"the CURRENT Auric v2: exam loss {current:.4f} -> the new one is "
                          f"{'better' if after < current else 'NOT better'}")
        except Exception as e:
            report.append(f"couldn't test the current Auric v2 ({type(e).__name__}): keeping it to be safe")
            current = -1.0
    better = after < before and (current is None or after < current)
    if smoke:                                              # check merging + saving too, then throw it away
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            model.merge_and_unload().save_pretrained(d, safe_serialization=True)
            report.append(f"merge + save ok ({sum(f.stat().st_size for f in Path(d).glob('*.safetensors')) / 1e9:.2f} GB)")
        print("\n".join(report))
        return
    if better:
        import shutil
        if (OUT / "config.json").exists():                 # the old brain is kept: one step back is always possible
            prev = OUT.with_name("auric-v2-prev")
            shutil.rmtree(prev, ignore_errors=True)
            shutil.copytree(OUT, prev)
            report.append(f"the previous brain is kept in {prev}")
        merged = model.merge_and_unload()
        OUT.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(OUT, safe_serialization=True)
        tok.save_pretrained(OUT)
        report.append(f"saved: {OUT} (NOT published: Saad decides when it goes online - switch.py start v2publish)")
    REPORT.write_text("\n".join(report), encoding="utf-8")
    progress.unlink(missing_ok=True)                       # finished: a next training starts fresh
    status(phase="done", before=round(before, 4), after=round(after, 4), current=current, kept=better)


if __name__ == "__main__":
    main()
