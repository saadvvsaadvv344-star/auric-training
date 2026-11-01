# Auric · cloud training

Saad's AIs learn here, on GitHub's free computers, so his PC's graphics card stays free.

**Live progress:** https://saadvvsaadvv344-star.github.io/auric-training/

- **Image AI** (`art_cloud.py`, model in `artgen.py`): Auric's own picture generator, built from zero, 64x64, ~200 kinds of
  things, 49,074 photos. Trains for as long as its pictures keep getting better.
- **Auric v2** (`v2_cloud.py`, method in `v2train.py`): Auric's 1.5B language brain (LoRA). Its material contains personal
  data, so it is **encrypted** (secret `AURIC_KEY`); only numbers are published.

Each run is ~5.6 h (`.github/workflows/`), saves its state to the release `state`, and starts the next run.
On/off: `control.json`.
