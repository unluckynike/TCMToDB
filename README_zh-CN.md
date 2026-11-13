# TCMToDB

[English](README.md) | 简体中文

[![Model](https://img.shields.io/badge/ModelScope-TCMTox--QA--Qwen3.5--9B-blue)](https://modelscope.cn/models/Venture/TCMTox-QA-Qwen3.5-9B) [![Dataset](https://img.shields.io/badge/Dataset-TCMToxQA--SFT-green)](https://modelscope.cn/datasets/Venture/TCMToxQA-SFT)

基于山东大学 [TCMToxDB](https://www.sdu-idea.cn/TCMToxDB)（毒性中药数据库）开放数据的应用与模型实验仓库。

> ⚠️ 仅供研究/实验使用。非医疗产品，不构成任何医疗建议。

## 项目内容

**进行中的子项目：[`tcmtox-llm/`](tcmtox-llm) —— 毒性中药领域 LLM 微调问答**

| 流水线阶段 | 做什么 | 关键数字 |
|---|---|---|
| 毒性分级清洗 | 229 味库内药材对齐《中国药典》（2020 版）83 种毒性药材；统一标注大小写；以药典为权威修正冲突（原值保留独立列） | 43 → **72** 味有分级；修正 **17** 处冲突（如川乌"小毒"→"大毒"） |
| 处方解析器 | 拼音词典匹配 + 炮制前缀剥离（炒/制/醋/酒…） + 剂量抽取 | **4,522** 条方剂-药材-剂量记录；覆盖 **1,289 / 1,534** 首;**零误配**（刻意禁用模糊匹配——早期试验中会把无毒药误标为毒性药） |
| SFT 数据构造 | 7 类任务：单药问答、药典安全、方剂分析、靶点/成分事实、保守剂量对比、**拒答与边界训练**、反向检索 | **6,234** 条（train 5,512 / val 329 / test 393）；实体分组切分，**0/3,640 组跨集泄漏**；拒答任务占 test ≥ **22%** |

设计文档：[数据构建](tcmtox-llm/docs/数据构建.md) · [训练与评测](tcmtox-llm/docs/训练与评测.md)

## 仓库结构

```
TCMToDB/
├── private/                   # 私人备忘（gitignore 隔离）
├── tcmtox_data/               # 原始与衍生数据（gitignore，可再生成）
├── tcmtox_scripts/            # 数据探查脚本
└── tcmtox-llm/                # 子项目：毒性中药 LLM 问答（方向 D）
    ├── scripts/               # 清洗 → 解析 → SFT 构造器
    ├── data/                  # 衍生产物（gitignore，可再生成）
    ├── train/                 # 训练配置（QLoRA）
    ├── eval/                  # 评测方案与脚本
    ├── rag/         # RAG 检索层 + Neo4j 图谱 + 问答服务
    └── docs/      # 数据构建/训练评测/RAG 设计文档
```

## 数据来源与许可

- 来源：TCMToxDB 官方下载页（4 份 CSV：方剂/草药/成分/靶点），以及经公开药师科普整理的《中国药典》（2020 版）毒性药材分级名单
- 本仓库**不分发任何数据文件**，仅包含生成数据的脚本；底层数据的使用遵循原数据库的许可条款

## 开源发布（ModelScope）

- 🤖 **模型**（合并版，下载即用）：https://modelscope.cn/models/Venture/TCMTox-QA-Qwen3.5-9B
- 🔧 **LoRA adapter**（86MB，需挂载基座）：https://modelscope.cn/models/Venture/TCMTox-QA-Qwen3.5-9B-lora
- 📊 **数据集**（TCMToxQA-SFT，6,234 条）：https://modelscope.cn/datasets/Venture/TCMToxQA-SFT

## 快速再生

```bash
# 1. 毒性分级清洗（229 味 × 药典 2020）
python3 tcmtox-llm/scripts/clean_herb_toxicity.py
# 2. 处方解析 → formula_herbs.csv
python3 tcmtox-llm/scripts/parse_formula_prescriptions.py
# 3. SFT 数据构造 → data/sft/*.jsonl
python3 tcmtox-llm/scripts/build_sft_data.py
```

随机种子固定（SEED=42），每个脚本读取 `tcmtox_data/` 并输出可再生文件。

## 进度

- [x] 方向调研与决策（毒性预测 / 处方审查 / DTI / LLM → 选定 LLM 优先）
- [x] 分级清洗、处方解析、SFT 构造（6,234 条）
- [x] 基线与训练评估（Qwen3.5-9B，多张 L20 48GB）
- [x] 基线评测（zs/fs 全量）——SFT 后分级准确率 4.8% → 90.5%
- [x] LoRA 微调（单卡 L20，49 分钟）——8/9 验收通过
- [x] 对比评测报告（判定见 tcmtox-llm/docs/训练与评测.md）
- [x] RAG 检索层（五轮迭代，系统级验收 9/9；SMILES 精确匹配 5.9% → 52.9%）
- [x] Neo4j 知识图谱（1,804 节点 / ~4,600 边，含十八反十九畏配伍禁忌边）
- [x] 流式问答服务 + 本草×朱砂可视化界面（检索过程/图谱/统计三面板）

## 安全性说明

- 医疗幻觉是首要失败模式：拒答训练（T6）与保守剂量回答（T5，绝不从投料量下"超量/安全"结论）是一等数据任务，每轮评测都跟踪拒答行为
- 源库 17 味药材标注与《药典 2020》冲突；清洗后数据以药典为准，并保留原标注供审计（`Toxicity_conflict` / `Toxicity_original` 列）
