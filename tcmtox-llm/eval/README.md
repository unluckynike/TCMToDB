# 评测与训练执行手册（服务器端）

> 前置：项目已 rsync 至 `~/codeWorkSpace/tcmtox`；模型在 `~/codeWorkSpace/llms/Qwen3.5-9B`；
> llamaf 环境可用；swift 环境已 `pip install vllm`。

## 1. 渲染格式检查（不加载模型，秒级）

```bash
conda activate swift   # 或 llamaf（需要 transformers）
cd ~/codeWorkSpace/tcmtox
python tcmtox-llm/eval/run_baseline.py \
    --model ~/codeWorkSpace/llms/Qwen3.5-9B \
    --condition zeroshot --dry-run --limit 2
```

确认输出无 `<think>` 思考段、以 assistant 生成提示结尾。

## 2. 基线评测（全量 393 条，单卡 3~5 分钟）

```bash
conda activate swift && cd ~/codeWorkSpace/tcmtox
CUDA_VISIBLE_DEVICES=0 python tcmtox-llm/eval/run_baseline.py \
    --model ~/codeWorkSpace/llms/Qwen3.5-9B --condition zeroshot
CUDA_VISIBLE_DEVICES=0 python tcmtox-llm/eval/run_baseline.py \
    --model ~/codeWorkSpace/llms/Qwen3.5-9B --condition fewshot
```

快速抽检加 `--limit 20`。产出：`tcmtox-llm/eval/results/<模型名>_<condition>.jsonl`

## 3. 评分（无模型依赖，本机/服务器均可）

```bash
python tcmtox-llm/eval/score.py tcmtox-llm/eval/results/*.jsonl
```

重点看三行：**分级准确率 / 毒性误标率 / T6拒答正确率**，外加冲突17味专项。
`detail` 字段给出误标案例前 10 条供人工审。

## 4. QLoRA 训练（LLaMA-Factory，单卡 ~30 分钟）

```bash
conda activate llamaf && cd ~/codeWorkSpace/tcmtox
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train tcmtox-llm/train/qlora_qwen35_9b.yaml
# loss 曲线：tensorboard --logdir saves/tcmtox-qlora-v1/runs
```

训练完成后合并导出（配置文件尾部有完整命令）：

```bash
llamafactory-cli export --model_name_or_path ~/codeWorkSpace/llms/Qwen3.5-9B \
    --adapter_name_or_path saves/tcmtox-qlora-v1 \
    --template qwen3_5_nothink --export_dir saves/tcmtox-merged-v1 \
    --export_size 4 --export_legacy_format false
```

## 5. 微调模型评测（与基线严格同条件）

```bash
conda activate swift && cd ~/codeWorkSpace/tcmtox
CUDA_VISIBLE_DEVICES=0 python tcmtox-llm/eval/run_baseline.py \
    --model saves/tcmtox-merged-v1 --condition zeroshot --tag finetuned
CUDA_VISIBLE_DEVICES=0 python tcmtox-llm/eval/run_baseline.py \
    --model saves/tcmtox-merged-v1 --condition fewshot --tag finetuned
python tcmtox-llm/eval/score.py tcmtox-llm/eval/results/*.jsonl   # 基线+微调同表对比
```

## 已知问题预案

- vLLM 报 flash-linear-attention 缺失 → `pip install flash-linear-attention`
- vLLM 对多模态 checkpoint 初始化慢/报 mm 相关错 → 尝试加 `--max-model-len 2048`；仍不行把报错贴回
- OOM → `--gpu-mem 0.7` 或训练 YAML 里 `per_device_train_batch_size: 4`（grad_accum 翻倍）
- LF 训练报 qwen3_5_nothink 模板不认识 → llamaf 环境不是最新源码安装，回看环境搭建步骤
