"""Auric v2 (1.5B, LoRA) trained on GitHub's free machines, one ~5.6 h piece per run, continuing where the last run
stopped. Same examples, order, LoRA and schedule as v2train.py on the PC.

PRIVATE: the material and every saved state are encrypted (AURIC_KEY); nothing readable is ever published - the
website only gets numbers (step, error). At the end the trained part (encrypted) waits in the release for Saad's PC,
which checks it against the CURRENT Auric v2 before keeping it.
"""
import io
import json
import math
import os
import random
import time
import zipfile
from datetime import datetime, timezone

import torch

import common as C
import v2train as V

torch.set_num_threads(int(os.environ.get("AURIC_THREADS", "4")))
BASE = os.environ.get("AURIC_V2_BASE", "Qwen/Qwen2.5-1.5B-Instruct")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def material():
    enc, raw = C.WORK / "v2_data.enc", C.WORK / "v2_data.zip"
    if not enc.exists() and not C.download("v2_data.enc", enc, tag="data"):
        raise SystemExit("Auric v2's material isn't in the release 'data'")
    C.decrypt_file(enc, raw)
    with zipfile.ZipFile(raw) as z:
        rows = {n: [json.loads(l) for l in z.read(n).decode("utf-8").splitlines() if l.strip()] for n in z.namelist()}
    raw.unlink()                                                  # (the readable copy never stays on disk)
    return rows["train.jsonl"], rows["exam.jsonl"]


def main():
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer
    old = C.read_status("v2.json") or {}
    if not str(old.get("phase", "")).startswith("finished"):
        C.publish_status({"v2.json": {**old, "phase": "starting a new run (decrypting the material, loading the brain"
                                      + (", then the base brain's exam)" if not old else ")"), "updated": now()}})
    train_rows, exam_rows = material()
    tok = AutoTokenizer.from_pretrained(BASE)
    train = [e for e in (V.encode(tok, r) for r in train_rows) if e]
    exam = [e for e in (V.encode(tok, r) for r in exam_rows) if e]
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)   # (processors are slow in half precision)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=V.LR, weight_decay=0.0)
    groups = V.batches(train, random.Random(3)) * V.EPOCHS               # (the same order as on the PC, every run)
    total = math.ceil(len(groups) / V.ACCUM)
    step, run, before, loss_avg = 0, 1, None, None
    state_enc, state = C.WORK / "v2_state.enc", C.WORK / "v2_state.pt"
    if C.download("v2_state.enc", state_enc):
        C.decrypt_file(state_enc, state)
        s = torch.load(state, map_location="cpu", weights_only=False)
        state.unlink()
        if s.get("total") == total:
            if s.get("finished"):                                 # done already: the schedule mustn't redo the exam
                print("Auric v2's cloud training is finished; the result waits for Saad's PC", flush=True)
                return
            set_peft_model_state_dict(model, s["adapter"])
            opt.load_state_dict(s["opt"])
            step, run, before, loss_avg = s["step"], s.get("run", 0) + 1, s.get("before"), s.get("loss")
    print(f"run {run}: {len(train):,} examples, step {step}/{total}", flush=True)

    def save(phase, finished=False):
        torch.save({"adapter": get_peft_model_state_dict(model), "opt": opt.state_dict(), "step": step, "total": total,
                    "run": run, "before": before, "loss": loss_avg, "finished": finished}, state)
        C.encrypt_file(state, state_enc)
        state.unlink()
        C.upload(state_enc)
        publish(phase)

    t0, start, last_pub = time.time(), step, 0.0

    def publish(phase):
        sps = (time.time() - t0) / max(1, step - start)
        C.publish_status({"v2.json": {"phase": phase, "run": run, "step": step, "total": total, "examples": len(train),
                                      "sec_per_step": round(sps, 1), "loss": None if loss_avg is None else round(loss_avg, 4),
                                      "exam_base": before, "cloud_hours_left": round((total - step) * sps / 3600, 1),
                                      "updated": now()}})

    if before is None and step == 0:                              # the base brain's exam score, once, for comparison
        with model.disable_adapter():
            before = V.exam_loss(model, tok, exam, "cpu")
        print(f"base exam loss {before:.4f}", flush=True)
    model.train()
    gi = step * V.ACCUM
    while step < total and C.time_left() > 1800:                 # (30 min kept to save + upload the state)
        ids, lab, att = V.collate(groups[gi], tok.pad_token_id, "cpu")
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / V.ACCUM
        loss.backward()
        gi += 1
        if gi % V.ACCUM == 0:
            lr = V.LR * min(1, step / 30) * 0.5 * (1 + math.cos(math.pi * step / total))
            for pg in opt.param_groups:
                pg["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            loss_avg = loss.item() * V.ACCUM if loss_avg is None else 0.98 * loss_avg + 0.02 * loss.item() * V.ACCUM
            if time.time() - last_pub > 1200:
                publish("learning")
                last_pub = time.time()
    if step < total:
        save("between runs (the next one continues)")
        C.history_add("v2", {"run": run, "from": start, "to": step, "loss": None if loss_avg is None else round(loss_avg, 4),
                             "ended": now()})
        return
    after = V.exam_loss(model, tok, exam, "cpu")                  # finished: the exam (never studied)
    final = C.WORK / "v2_adapter_final.pt"
    torch.save({"adapter": get_peft_model_state_dict(model), "exam_after": after, "exam_base": before}, final)
    C.encrypt_file(final, final.with_suffix(".enc"))
    C.upload(final.with_suffix(".enc"))
    save(f"finished: exam {after:.4f} (base {before:.4f}) - waiting for Saad's PC to compare it with the current Auric v2",
         finished=True)
    C.history_add("v2", {"run": run, "from": start, "to": step, "exam_after": round(after, 4), "exam_base": before,
                         "ended": now()})


if __name__ == "__main__":
    main()
