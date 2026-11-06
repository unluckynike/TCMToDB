# TCMToDB

[English](README.md) | [简体中文](README_zh-CN.md)

Applications and model experiments built on [TCMToxDB](https://www.sdu-idea.cn/TCMToxDB) — a Toxic Medicinal Herb Database of Traditional Chinese Medicine by Shandong University — and its open downloads.

> ⚠️ Research / experiment repo only. Nothing here is medical advice or a medical product.

## What's Inside

**Active subproject: [`tcmtox-llm/`](tcmtox-llm) — domain LLM fine-tuning for TCM toxicity QA**

| Pipeline Stage | What It Does | Key Numbers |
|---|---|---|
| Toxicity grade cleaning | Align 229 DB herbs against the 83 toxic herbs of *Chinese Pharmacopoeia (2020)*; unify label casing; resolve conflicts with the pharmacopoeia as authority (originals kept in a separate column) | 43 → **72** graded herbs; **17** conflicts corrected (e.g. *Chuan Wu* "slightly toxic" → "highly toxic") |
| Prescription parser | Pinyin dictionary matching + processing-prefix stripping (炒/制/醋/酒…) + dosage extraction over formula prescriptions | **4,522** formula–herb–dose records; **1,289 / 1,534** formulas covered; **0 false positives** (fuzzy matching deliberately disabled — it mislabeled non-toxic herbs as toxic in early trials) |
| SFT data construction | 7 task types: single-herb QA, pharmacopoeia safety, formula analysis, target/ingredient facts, conservative dose comparison, **refusal & boundary training**, reverse lookup | **6,234** samples (train 5,512 / val 329 / test 393); entity-grouped split — **0/3,640 groups leak across sets**; refusal tasks ≥ **22%** of test |

Design docs (Chinese): [instruction data design](tcmtox-llm/docs/指令数据设计.md) · [baseline & training assessment](tcmtox-llm/eval/基线评测与训练评估方案.md)

## Repo Layout

```
TCMToDB/
├── private/                   # personal memos (gitignored)
├── tcmtox_data/               # raw + derived data (gitignored, regenerable)
├── tcmtox_scripts/            # data profiling scripts
└── tcmtox-llm/                # subproject: TCM toxicity LLM QA (direction D)
    ├── scripts/               # cleaning → parsing → SFT builders
    ├── data/                  # derived artifacts (gitignored, regenerable)
    ├── train/                 # training configs (QLoRA)
    ├── eval/                  # evaluation plan & scripts
    └── demo/                  # inference demo
```

## Data Provenance & License

- Sources: TCMToxDB official download page (4 CSVs: formula / herb / ingredient / target) and the toxic-herb grading list of *Chinese Pharmacopoeia (2020 edition)* via public pharmacist references.
- This repo **does not distribute any data files** — only the scripts that generate them. Use of the underlying data follows the original database's terms.

## Reproduce

```bash
# 1. toxicity grade cleaning (229 herbs × Pharmacopoeia 2020)
python3 tcmtox-llm/scripts/clean_herb_toxicity.py
# 2. prescription parsing → formula_herbs.csv
python3 tcmtox-llm/scripts/parse_formula_prescriptions.py
# 3. SFT data construction → data/sft/*.jsonl
python3 tcmtox-llm/scripts/build_sft_data.py
```

Deterministic (SEED=42). Each script reads `tcmtox_data/` and writes regenerable artifacts.

## Status

- [x] Direction survey & decision (predict / review / DTI / LLM → chose LLM-first)
- [x] Grade cleaning, prescription parsing, SFT construction (6,234 samples)
- [x] Baseline & training assessment (Qwen3.5-9B, multi × L20 48GB)
- [x] Baseline evaluation (zero/few-shot) — grade accuracy 4.8% → 90.5% after SFT
- [x] LoRA fine-tuning (single L20, 49 min) — 8/9 acceptance criteria passed
- [x] Comparison report (verdict archived in eval/基线结果存档.md)
- [ ] Demo (Gradio + RAG to close the SMILES/CAS exact-match gap)

## Safety Notes

- Medical hallucination is the primary failure mode: refusal training (T6) and conservative dose answers (T5, never conclude "safe/overdose" from batch input amounts) are first-class data tasks, and refusal behavior is tracked in every evaluation round.
- 17 herb labels in the source DB conflict with *Pharmacopoeia 2020*; the cleaned dataset follows the pharmacopoeia and keeps the original label for audit (`Toxicity_conflict` / `Toxicity_original` columns).
