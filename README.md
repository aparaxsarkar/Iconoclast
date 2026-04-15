# 🗡️ ICONOCLAST

**Discriminative representation editing for open-weight language models: remove refusal behavior without paying the utility tax.**

ICONOCLAST extends HERETIC-style directional abliteration with **benign-subspace preservation**: before editing, candidate refusal directions are projected out of a low-rank subspace estimated from harmless activations, so the edit removes refusals while leaving normal behavior intact.

---

## 💻 Quick Start

Use Python 3.10+ and a GPU-capable PyTorch installation.

```bash
git clone https://github.com/aparaxsarkar/Iconoclast.git
cd Iconoclast/Iconoclast

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[research,benchmark,quantized]"
```

ICONOCLAST reads `config.toml` by default. To run one of the benchmark configs locally:

```bash
cp config.llama3_1_8b.benchmark.rutgers.toml config.toml
iconoclast
```

*(For gated Hugging Face models, export `HF_TOKEN` first.)*

Run the tests with:

```bash
python -m pytest tests
```

---

## 🚀 Results: ICONOCLAST vs. HERETIC

We matched ICONOCLAST against a HERETIC-style directional abliteration baseline across **10 open-weight models** on the Rutgers iLabs cluster.

The selection rule is strict: fewer harmful refusals (20 JailbreakBench holdouts), then fewer benign overrefusals (64 Alpaca holdouts), then lower first-token KL divergence from the base model.

**ICONOCLAST wins 10 out of 10 matched rows.**

| Model | ICONOCLAST Refusals | ICONOCLAST Overrefusals | ICONOCLAST KL | HERETIC Refusals | HERETIC Overrefusals | HERETIC KL | Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Llama-3.1-8B-Instruct** | **0/20** | **0/64** | **0.0447** | 1/20 | 0/64 | 0.1854 | 🏆 **ICONOCLAST** |
| **Qwen3.5-9B base** | **10/20** | **2/64** | **0.0055** | 10/20 | 3/64 | 0.0160 | 🏆 **ICONOCLAST** |
| **Mistral-7B-Instruct-v0.3** | **1/20** | **0/64** | **0.0554** | 4/20 | 0/64 | 0.1317 | 🏆 **ICONOCLAST** |
| **Falcon3-7B-Instruct** | **0/20** | **0/64** | **6.1448** | 4/20 | 1/64 | 0.1648 | 🏆 **ICONOCLAST** (Behavior Win) |
| **Gemma-2-2B-IT** | **1/20** | **0/64** | **0.1849** | 1/20 | 2/64 | 0.6441 | 🏆 **ICONOCLAST** |
| **Phi-4-mini-instruct** | **2/20** | **1/64** | **0.0204** | 2/20 | 1/64 | 0.0978 | 🏆 **ICONOCLAST** |
| **Yi-1.5-9B-Chat** | **2/20** | **0/64** | **0.0511** | 3/20 | 0/64 | 0.0355 | 🏆 **ICONOCLAST** (Behavior Win) |
| **StableLM2-1.6B** | **2/20** | **0/64** | **0.0328** | 3/20 | 0/64 | 0.0670 | 🏆 **ICONOCLAST** |
| **SmolLM2-1.7B-Instruct** | **1/20** | **1/64** | **0.0087** | 2/20 | 2/64 | 0.2699 | 🏆 **ICONOCLAST** |
| **OLMo-2-1B-Instruct** | **2/20** | **0/64** | **0.0345** | 2/20 | 1/64 | 0.0944 | 🏆 **ICONOCLAST** |

### 🌟 Highlights
- **Strict behavior wins:** strictly fewer harmful refusals in 6/10 rows.
- **Utility preservation:** lower KL divergence in 8/10 rows.
- **Large KL reductions:** SmolLM2 (0.2699 → 0.0087) and Gemma-2-2B (0.6441 → 0.1849).
- **Llama-3.1-8B:** 0/20 harmful refusals and 0/64 benign overrefusals at 0.0447 KL.

Full breakdown: [Iconoclast/PUBLISHABLE_RESULTS.md](Iconoclast/PUBLISHABLE_RESULTS.md).

---

## 🧠 Method

Standard directional abliteration stops refusals, but by erasing a direction wholesale it often damages benign capabilities: the model over-refuses safe prompts or drifts from its original distribution (high KL). ICONOCLAST avoids this:

1. **Collect & contrast:** collect residual activations for harmless and harmful prompts during one-token generation, and build candidate directions with mean, median, variance-scaled and hybrid estimators.
2. **Preserve the benign subspace:** estimate a low-rank PCA subspace from the harmless residuals and project candidate refusal directions *out* of it before editing.
3. **Low-rank LoRA edits:** apply rank-one or low-rank LoRA edits to attention output and MLP down-projection modules, with Optuna searching over layer, component, strength and direction parameters.
4. **Multi-objective selection:** track harmful-prompt refusals, benign overrefusals, disclaimer-marker hits and KL divergence, and pick the Pareto-optimal edit.

---

## 📂 Repository Layout

| Path | Purpose |
| --- | --- |
| `Iconoclast/src/iconoclast/config.py` | Pydantic settings for models, datasets, objectives, checkpointing. |
| `Iconoclast/src/iconoclast/direction.py` | Direction estimators and benign-subspace projection math. |
| `Iconoclast/src/iconoclast/model.py` | Hugging Face loading, LoRA setup, generation, residual extraction. |
| `Iconoclast/src/iconoclast/evaluator.py` | Multi-objective evaluation (refusal, overrefusal, compliance, KL). |
| `Iconoclast/src/iconoclast/main.py` | Optuna optimization loop, Pareto selection, batch summaries. |
| `Iconoclast/scripts/` | Rutgers iLabs setup, Slurm runners, matched HERETIC sweeps. |
| `Iconoclast/tests/` | Unit tests for projection math and reproducible seeding. |
| `Iconoclast/config.*.toml` | Per-model benchmark configurations. |
| `pipelines/`, `benchmarks/`, `results/` | Earlier PII-unlearning experiments (MLX/PyTorch abliteration, DPO baseline, MMLU and generation evals). |
| `proposal/`, `MIDSEM_REPORT.md` | Project proposal and midsemester report. |
| `docs/` | Design notes, iLab experiment notes, and the [MLX PII Abliterator](docs/MLX_PII_ABLITERATOR.md) write-up. |

---

## 🎓 License

- **Copyright:** (C) 2025-2026 Varesh Patel, Urvi Desai, Aparajita Sarkar + contributors
- **License:** AGPL-3.0-or-later. See [Iconoclast/NOTICE.md](Iconoclast/NOTICE.md) for attribution details.

Thanks to Rutgers iLabs for the compute used in these evaluations.
