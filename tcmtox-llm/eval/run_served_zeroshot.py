# -*- coding: utf-8 -*-
"""
本模型（vLLM serve 常驻）在评测集上零样本跑分，产出 score.py 兼容 JSONL。
与 run_multillm.py 的区别：走 GPU 服务器 vllm serve 的 /v1/completions，
prompt 用原始模板串（与训练格式逐字符一致，nothink 无空 think 块）。
用于论文"通用 LLM vs 领域微调模型"对比表的微调侧。

用法（Mac，需服务器 vLLM 已启动）：
  python3 tcmtox-llm/eval/run_served_zeroshot.py
  python3 tcmtox-llm/eval/run_served_zeroshot.py --limit 5   # 冒烟

输出：eval/results/{--name}_zeroshot.jsonl（同 score.py 格式，断点续跑）
"""
import argparse
import json
import time
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parents[2]
env = {}
for line in (BASE / "tcmtox-llm" / "rag" / ".env").read_text(encoding="utf-8").split("\n"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()

LLM_BASE_URL = env.get("LLM_BASE_URL", "").rstrip("/")
LLM_MODEL = env.get("LLM_MODEL", "tcmtox-qa")
if not LLM_BASE_URL:
    raise SystemExit("缺少 LLM_BASE_URL，请在 tcmtox-llm/rag/.env 中配置")


def chat(question, max_tokens=512, retries=3):
    # 原始模板串，与 qwen3_5_nothink 训练格式逐字符一致（无空 think 块）
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    for attempt in range(retries):
        try:
            r = requests.post(f"{LLM_BASE_URL}/completions",
                              json={"model": LLM_MODEL, "prompt": prompt,
                                    "temperature": 0, "max_tokens": max_tokens},
                              timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["text"], None
        except Exception as e:
            if attempt == retries - 1:
                return "", f"{type(e).__name__}: {str(e)[:80]}"
            time.sleep(3 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE / "tcmtox-llm" / "data" / "sft" / "test.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=str(BASE / "tcmtox-llm" / "eval" / "results"))
    ap.add_argument("--name", default="finetuned-v1_fulltest")
    args = ap.parse_args()

    test = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit > 0:
        test = test[:args.limit]
    print(f"test 样本: {len(test)}", flush=True)

    out_path = Path(args.out_dir) / f"{args.name}_zeroshot.jsonl"
    # 断点续跑：已有输出则跳过已完成的 instruction
    done = set()
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(l)["instruction"])
            except Exception:
                pass
        print(f"断点续跑：已完成 {len(done)} 条", flush=True)

    todo = [s for s in test if s["instruction"] not in done]
    t0, errs, n = time.time(), 0, 0
    from concurrent.futures import ThreadPoolExecutor
    CHUNK = 50
    with open(out_path, "a", encoding="utf-8") as f:
        for ci in range(0, len(todo), CHUNK):
            chunk = todo[ci:ci + CHUNK]
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda s: (s, chat(s["instruction"])), chunk))
            for s, (ans, err) in results:
                if err:
                    errs += 1
                    if errs <= 3 or errs % 25 == 0:
                        print(f"[warn] {err}", flush=True)
                rec = {"instruction": s["instruction"], "input": "",
                       "response": ans, "gold": s["output"],
                       "meta": s["meta"], "condition": "zeroshot",
                       "model": "finetuned-v1(tcmtox-qa)", "rag": {"used": False}}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            el = time.time() - t0
            print(f"[进度] {ci + len(chunk)}/{len(todo)} | {el:.0f}s | 错误 {errs}",
                  flush=True)
    print(f"✓ 完成：新增 {n} 条，错误 {errs}，耗时 {time.time()-t0:.0f}s → {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
