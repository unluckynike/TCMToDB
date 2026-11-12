# RAG 检索层设计（v2 · 全离线单机方案）

> 目标：闭合 SMILES/CAS/PubChem 精确匹配缺口（验收线系统级 ≥90%），
> 同时验证 SFT vs RAG vs SFT+RAG 三方对比。
> **重心在指标提升，界面最简。**

## 一、部署形态：全离线单机（v2 定案）

**评测可复现性不能依赖间歇性外网**——索引与查询必须使用同一 provider 的向量，
provider 漂移 = 索引作废。故定案：

| 组件 | 选型 | 部署 | 来源（ModelScope，已验证可达） |
|---|---|---|---|
| 生成模型 | TCMTox-QA-Qwen3.5-9B（merged） | GPU0 · vLLM | 已就位 |
| Embedding | **BGE-m3**（1024 维，中英强，dense+sparse）或 Qwen3-Embedding-0.6B | GPU1 或 CPU | BAAI/bge-m3 |
| Rerank | **bge-reranker-v2-m3** | GPU1 或 CPU | 同 API 网关背后的同一开源模型，本地化无损 |
| 向量存储 | **Qdrant 本地模式**（`QdrantClient(path=...)`） | 进程内，零服务 | pip install qdrant-client |
| 精确查表 | SQLite + 实体名字典 | 本地 | 内置 |

API（newapi `qwen3-embedding`，4096 维）实测可用，降级为**可选增强**：
`EMBEDDING_PROVIDER=api|local` 经 `.env` 切换；**同一 collection 的索引与查询
强制同 provider**（collection 元数据记录 provider 名与维度，启动时校验）。

预留（第二轮）：Neo4j（图谱实例暂不可达 + 关系表未到位，双重前提未满足）；
关系表到手后启用多跳机制问答（药材→成分→靶点→毒性类型），router 输出结构
已预留 `graph_context` 字段。

## 二、检索架构：实体路由 + 双路检索

```
问题 → 【实体路由】（规则层，无模型参与，拒答保护）
  ├─ 实体命中（药名/方名/成分名，含拼音变体与炮制前缀归一化）
  │    → 精确查表（SQLite）：返回该实体全部字段记录
  ├─ 无实体但为语义问题（反向/列举/描述类）
  │    → embedding 召回 top50 → rerank 精排 top5 → 返回记录片段
  └─ T6 模式（实体不在库 / 关系缺失 / 医疗建议）
       → 不检索，直接问模型（保护拒答行为——检索上下文可能诱导编造）
```

- 精确查表覆盖 ~85% 问题（实体锚定型），是 SMILES/CAS 缺口的闭合路径
- 语义检索仅服务模糊问题；rerank 对中药语料的区分度已实测验证（川乌查询
  正例 0.71 / 负例 0.00）
- 实体词典复用处方解析器的归一化表（拼音变体 3 条 + 炮制前缀）

## 三、上下文注入格式（A/B 实验变量）

```
[库内记录，供参考]
- 成分：gentiopicrin ｜ SMILES: ... ｜ CAS: 20867-06-3 ｜ 来源表：TCMTox_ingredient

[问题] gentiopicrin 的 SMILES 是什么？
```

- **策略 A（全字段）**：命中实体的全部字段注入——信息全，token 多
- **策略 B（字段选择）**：按问题类型注入相关字段（问题类型→字段映射表）——省 token
- 两者各跑一轮评测对比；答案须注明"依据库内记录"，T6 类行为保持不检索

## 四、实施步骤

| 步骤 | 内容 | 产物 | 依赖 |
|---|---|---|---|
| S1 | 语料构建：4 表 → 检索语料（每实体一条记录文档） | `rag/build_corpus.py` | 无 |
| S2 | 实体路由器：词典 + 归一化 + 问题类型分类 + 拒答保护 | `rag/router.py` | 无 |
| S3 | 检索客户端：EmbeddingProvider 抽象（local/api）+ Qdrant 本地模式 + rerank 客户端 + numpy 降级 | `rag/retriever.py` | 本地模型下载 |
| S3.5 | 【服务器】模型下载（BGE-m3 + reranker ~5GB）→ 建 collection → 灌库 | collection `tcmtodb` | S1/S3 |
| S4 | RAG pipeline：`run_baseline.py --rag` 开关，复用同一评分器 | `rag/rag_pipeline.py` | S2/S3 |
| S5 | 带 RAG 重评（zs 条件）→ 四列对比：基线 / SFT / SFT+RAG（策略A/B） | `results/*rag*.jsonl` | S4 |
| S6 | 结果分析 → 存档与 README 更新 | 验收对照 | S5 |

本地模型下载（服务器，ModelScope 已验证可达）：
```bash
modelscope download --model BAAI/bge-m3 --local_dir ~/codeWorkSpace/llms/bge-m3
modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir ~/codeWorkSpace/llms/bge-reranker-v2-m3
```

## 五、系统级验收线（在原九项上追加 RAG 列）

| 指标 | 纯 SFT 现状 | +RAG 验收线 |
|---|---|---|
| SMILES / CAS / PubChem 精确匹配 | 5.9% / 0% | **≥90%** |
| 分级准确率 | 90.5% | ≥90.5%（不回归） |
| T6 拒答正确率 | 97.6% | ≥97.6%（不回归） |
| 误拒率 | 4.8% | ≤5% |
| T5 保守性 | 100% | 100% |
| 开放式 F1 | 0.801 | ≥0.85（锚定效应） |

## 六、风险与预案

| 风险 | 预案 |
|---|---|
| 检索上下文诱导模型给库外药编答案（H：拒答回归） | T6 类问题不检索（路由层拦截）；评测单独盯 T6 回归 |
| 长上下文稀释（策略 A token 膨胀） | 切策略 B；cutoff_len 4096 内可控 |
| API 与本地 embedding 混用导致向量错乱 | collection 元数据记录 provider+维度，查询时强制校验 |
| 本地 embedding 模型选型犹豫 | BGE-m3 先行（MIT 许可、生态成熟）；Qwen3-Embedding-0.6B 作为消融对比项 |
