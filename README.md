# TCMToDB

围绕山东大学 [TCMToxDB](https://www.sdu-idea.cn/TCMToxDB)（毒性中药数据库）开放数据的应用与模型实验仓库。

## 仓库结构

```
TCMToDB/
├── docs/
│   └── 方向规划.md            # 四个可展开方向的调研与决策记录
├── tcmtox_data/               # 原始数据（gitignore，不入库）
│   ├── TCMTox_{formula,herb,ingredient,target}.csv
│   ├── TCMTox_herb_clean.csv          # 衍生：毒性分级清洗（药典2020对齐）
│   └── yd2020_toxic83.csv             # 衍生：药典83种毒性药材参考表
├── tcmtox_scripts/            # 数据探查脚本
└── tcmtox-llm/                # 子项目：毒性中药 LLM 问答（方向D）
    ├── README.md
    ├── docs/指令数据设计.md    # T1~T7 任务设计与实际产出记录
    ├── scripts/
    │   ├── clean_herb_toxicity.py         # 毒性分级清洗（43→72味）
    │   ├── parse_formula_prescriptions.py # 处方解析（拼音+炮制前缀+剂量）
    │   └── build_sft_data.py              # SFT 数据构造器（6234条）
    └── data/                  # 衍生产物（gitignore，可由脚本再生）
        ├── formula_herbs.csv  # 方剂-药材-剂量结构化表
        └── sft/               # train/val/test + 分任务 jsonl
```

## 数据来源与许可

- 数据来自 TCMToxDB 官方下载页（4 份 CSV）与《中国药典》2020 年版毒性药材公开名单
- 本仓库**不分发任何数据文件**，仅包含生成数据的脚本；使用数据请遵循原数据库的许可条款

## 快速再生

```bash
# 1. 毒性分级清洗
python3 tcmtox-llm/scripts/clean_herb_toxicity.py
# 2. 处方解析
python3 tcmtox-llm/scripts/parse_formula_prescriptions.py
# 3. SFT 数据构造
python3 tcmtox-llm/scripts/build_sft_data.py
```

脚本间通过 `tcmtox_data/` 与 `tcmtox-llm/data/` 衔接，全部可独立重跑，随机种子固定（SEED=42）。
