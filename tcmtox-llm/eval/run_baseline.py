# -*- coding: utf-8 -*-
"""
基线/微调模型评测：vLLM 离线批处理推理

用法（服务器, swift 环境）：
  conda activate swift
  cd ~/tcmtox
  # 零样本全量
  python tcmtox-llm/eval/run_baseline.py \
      --model /home/zhl/codeWorkSpace/llms/Qwen3.5-9B \
      --condition zeroshot
  # few-shot 全量
  python tcmtox-llm/eval/run_baseline.py \
      --model /home/zhl/codeWorkSpace/llms/Qwen3.5-9B --condition fewshot
  # 快速抽检 20 条（不加载模型，看渲染格式用 --dry-run）
  python tcmtox-llm/eval/run_baseline.py --model <path> --condition zeroshot --limit 20
  python tcmtox-llm/eval/run_baseline.py --model <path> --condition zeroshot --dry-run

设计要点：
  - vLLM 离线 LLM 类（不是 vllm serve），单卡即可，连续批处理
  - enable_thinking 显式关闭（Qwen3.5 混合思考），与训练模板 qwen3_5_nothink 对齐
  - few-shot 示例从 train 集（非 test）按同任务类型采样，给底座最大的公平机会
  - 输出 JSONL 保留 gold 与全部 meta，score.py 单文件即可评分
  - 微调模型评测：--model 指向合并导出后的目录，其余参数不变 → 严格可比
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SFT_DIR = REPO / "tcmtox-llm" / "data" / "sft"
DEFAULT_MODEL = "/home/zhl/codeWorkSpace/llms/Qwen3.5-9B"
SYSTEM_PROMPT = ""  # 不加 system prompt：训练数据无 system，评测保持一致


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_conversation(sample, exemplars):
    """组装对话：few-shot 时前缀同任务示例（user/assistant 交替），无 system"""
    msgs = []
    for ex in exemplars:
        msgs.append({"role": "user", "content": ex["instruction"]})
        msgs.append({"role": "assistant", "content": ex["output"]})
    content = sample["instruction"]
    if sample.get("input"):
        content += "\n" + sample["input"]
    msgs.append({"role": "user", "content": content})
    return msgs


def pick_exemplars(train, task, rng, k=3):
    """为一条测试样本挑选 k 个 few-shot 示例（只从 train 集取）。

    策略：优先取与测试题相同任务类型的样本（T1a 的题配 T1a 的示例），
    让示例示范的是同类问题的答题格式与口径；同任务不足 k 条时用其他任务补齐。

    零泄漏是结构保证的：数据切分按实体（药/方/靶点）分组，示例中出现的
    实体必然与测试题实体不同——示例只教格式，结构上不可能泄答案。

    rng 由调用方传入且种子固定，保证每次运行挑选结果可复现。
    """
    same = [r for r in train if r["meta"]["task"] == task]
    pool = same if len(same) >= k else same + [r for r in train if r not in same]
    return rng.sample(pool, min(k, len(pool)))


# 与 LLaMA-Factory qwen3_5_nothink 训练模板对齐：
# 该模板的 assistant 段不含任何 think 块，而 Qwen 官方 chat template 在
# enable_thinking=False 时会插入空块 "<think>\n\n</think>\n"，需剥除以保证训练/推理格式一致
EMPTY_THINK_RE = re.compile(r"<think>\s*</think>\n*")


def render_prompts(llm_or_tok, convs, tokenizer=None):
    """渲染对话 → 剥空 think 块 → 返回最终 prompt 字符串列表"""
    tok = tokenizer if tokenizer is not None else llm_or_tok.get_tokenizer()
    rendered = [
        tok.apply_chat_template(
            c, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for c in convs
    ]
    return [EMPTY_THINK_RE.sub("", r) for r in rendered]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=str(SFT_DIR / "test.jsonl"))
    ap.add_argument("--train", default=str(SFT_DIR / "train.jsonl"))
    ap.add_argument("--condition", choices=["zeroshot", "fewshot"], default="zeroshot")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只评 N 条（按任务分层抽样）")
    ap.add_argument("--out-dir", default=str(REPO / "tcmtox-llm" / "eval" / "results"))
    ap.add_argument("--tag", default="", help="结果文件前缀，默认取模型目录名")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--dry-run", action="store_true", help="只渲染 prompt 不加载模型")
    args = ap.parse_args()

    tag = args.tag or Path(args.model).name
    test = load_jsonl(args.data)
    rng = random.Random(42)

    if args.condition == "fewshot":
        train = load_jsonl(args.train)
    else:
        train = []

    # --limit：按任务轮转抽样，保证 20 条抽检覆盖各任务
    if args.limit > 0:
        by_task = defaultdict(list)
        for s in test:
            by_task[s["meta"]["task"]].append(s)
        picked, tasks = [], sorted(by_task)
        i = 0
        while len(picked) < args.limit:
            t = tasks[i % len(tasks)]
            if by_task[t]:
                picked.append(by_task[t].pop(0))
            i += 1
        test = picked

    convs = []
    for s in test:
        ex = pick_exemplars(train, s["meta"]["task"], rng) if train else []
        convs.append(build_conversation(s, ex))

    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        s, c = test[0], convs[0]
        raw = tok.apply_chat_template(
            c, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        final = render_prompts(None, [c], tokenizer=tok)[0]
        print("===== 原始渲染（enable_thinking=False）=====")
        print(raw)
        print("===== 剥空 think 块后（实际送入模型的 prompt，与训练模板一致）=====")
        print(final)
        return

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    prompts = render_prompts(llm, convs)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}_{args.condition}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s, o in zip(test, outs):
            rec = {
                "instruction": s["instruction"],
                "input": s.get("input", ""),
                "response": o.outputs[0].text,
                "gold": s["output"],
                "meta": s["meta"],
                "condition": args.condition,
                "model": args.model,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"完成：{len(test)} 条，耗时 {dt:.0f}s → {out_path}")
    print(f"下一步：python tcmtox-llm/eval/score.py {out_path}")


if __name__ == "__main__":
    main()
