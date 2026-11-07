# 领域 LLM LoRA 微调 + TCM 毒性垂直问答

> **状态：第一轮训练闭环完成（2026-11-06），9 项验收 8 过 1 未达。**

## 成果一览

| 指标 | 基线 Qwen3.5-9B (zs) | 微调后 (zs) | 验收线 |
|---|---|---|---|
| 分级准确率 | 4.8% | **90.5%** | ≥85% ✅ |
| 毒性误标率 ⚠ | 60% | **2.5%** | ≤8% ✅ |
| T6 拒答正确率 | 47.6% | **97.6%** | ≥85% ✅ |
| T5 保守性（不下断言） | 22.2% | **100%** | ≥90% ✅ |
| think 泄漏率 | 100% | **0%** | ≤2% ✅ |
| 冲突 17 味准确率 | 0% | **85.7%** | ≥85% ✅ |
| 开放式字符 F1 | 0.105 | **0.801** | ≥0.55 ✅ |
| SMILES/CAS/PubChem 精确匹配 | 0% | 5.9% ~ 0% | ❌ 结构性缺口，归口 RAG |

完整对比与误标案例分析见 [docs/训练与评测.md](docs/训练与评测.md)；
训练成本：单卡（L20）49 分钟，bf16 LoRA rank32 全线性层，3 epochs。
**开源发布**：[合并版模型](https://modelscope.cn/models/Venture/TCMTox-QA-Qwen3.5-9B) · [LoRA adapter](https://modelscope.cn/models/Venture/TCMTox-QA-Qwen3.5-9B-lora) · [TCMToxQA-SFT 数据集](https://modelscope.cn/datasets/Venture/TCMToxQA-SFT)（ModelScope）

## 流水线（五步可复现）

```bash
# ① 数据：毒性分级清洗（229味×药典2020对齐，43→72味，17处冲突修正）
python3 tcmtox-llm/scripts/clean_herb_toxicity.py
# ② 数据：处方解析（拼音+炮制前缀+剂量 → 方剂-药材-剂量结构化表）
python3 tcmtox-llm/scripts/parse_formula_prescriptions.py
# ③ 数据：SFT 构造（T1~T7 七类任务，6234 条，实体分组切分零泄漏）
python3 tcmtox-llm/scripts/build_sft_data.py
# ④ 训练（LLaMA-Factory，单卡 ~50 分钟）
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train tcmtox-llm/train/qlora_qwen35_9b.yaml
# ⑤ 评测（vLLM 离线批推理 + 自动评分，基线/微调同条件对比）
python tcmtox-llm/eval/run_baseline.py --model <模型路径> --condition zeroshot
python tcmtox-llm/eval/score.py tcmtox-llm/eval/results/*.jsonl --save
```

评测设计要点：训练/推理模板严格对齐（`qwen3_5_nothink` + 剥空 think 块）、
zero/few-shot 双条件、实体分组切分防泄漏、拒答与保守回答作为一等评测指标。

## 目录结构

```
tcmtox-llm/
├── scripts/
│   ├── clean_herb_toxicity.py         # 毒性分级清洗（药典2020对齐）
│   ├── parse_formula_prescriptions.py # 处方解析器
│   └── build_sft_data.py              # T1~T7 SFT 构造器
├── data/
│   ├── dataset_info.json              # LLaMA-Factory 数据集注册
│   ├── formula_herbs.csv              # 方剂-药材-剂量（再生，gitignore）
│   └── sft/*.jsonl                    # 训练/验证/测试（再生，gitignore）
├── train/qlora_qwen35_9b.yaml         # QLoRA/LoRA 配置（含合并导出命令）
├── eval/
│   ├── run_baseline.py                # vLLM 离线批推理（zs/fs，--limit 抽检）
│   ├── score.py                       # 自动评分器（9 指标 + 多文件对比）
│   └── results/                       # 评测产物（服务器推送）
├── docs/
│   ├── 数据构建.md                    # SFT 数据集设计（T1~T7 定义与产出）
│   └── 训练与评测.md                  # 评估方案、基线/微调结果与验收判定
└── demo/                              # 下一阶段：Gradio + RAG
```

## 路线图

| 阶段 | 状态 | 产出 |
|---|---|---|
| 1. 数据构造 | ✅ 完成 | 6,234 条 SFT 数据 + 方剂-药材结构化表 |
| 2. 基线评测 | ✅ 完成 | zs/fs 双条件基线，正式存档 |
| 3. 微调训练 | ✅ 完成 | bf16 LoRA，49 min，eval loss 0.361（无过拟合） |
| 4. 评测对比 | ✅ 完成 | **8/9 验收通过** |
| 5. Demo（Gradio + RAG） | ⬛ 下一阶段 | RAG 同时闭合 SMILES/CAS 精确匹配缺口 |
| 6. 第二轮（可选） | 💤 待定 | DPO 行为精修（误拒校准）、关系表数据扩充（依赖网站恢复） |

## 已知边界

- **SMILES/CAS/PubChem 精确复现**：实体分组切分下测试成分在训练中从未出现，
  任意标识符无法泛化记忆——属检索问题而非训练问题，由 Demo 的 RAG 层解决
- **误拒率 4.8%**：拒答训练的轻度副作用，限内；第二轮 DPO 校准（用模型错误拒答
  构造偏好对）为候选方案
- **数据边界**：`tcmtox_data/` 原始数据与 `data/` 衍生产物不入库（遵循原库许可），
  全部由脚本确定性再生（SEED=42）
