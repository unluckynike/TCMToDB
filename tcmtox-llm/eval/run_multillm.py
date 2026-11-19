# -*- coding: utf-8 -*-
"""
多 LLM 横向评测：通用 LLM（API）在 test 集上零样本跑分，产出 score.py 兼容 JSONL
用于论文"通用 LLM vs 领域微调模型"对比表。

用法（Mac）：
  python3 tcmtox-llm/eval/run_multillm.py --models DeepSeek-V4.1-Flash,Gemma-4
  python3 tcmtox-llm/eval/run_multillm.py --models DeepSeek-V4.1-Flash --limit 50   # 快试

输出：eval/results/multi_{模型名}_zeroshot.jsonl（同 score.py 格式）
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

EVAL2_BASE = env.get("EVAL2_BASE", "").rstrip("/")
EVAL2_KEY = env.get("EVAL2_KEY", "")
if not EVAL2_BASE or not EVAL2_KEY:
    raise SystemExit("缺少 EVAL2_BASE / EVAL2_KEY，请在 tcmtox-llm/rag/.env 中配置")
HEADERS = {"Authorization": f"Bearer {EVAL2_KEY}", "Content-Type": "application/json"}


REASONING_MODELS = {"DeepSeek-V4.1-Flash"}  # 推理模型需显式关思考，否则 512 token 全被 reasoning 吃光（content=null）


def chat(model, question, max_tokens=512, retries=3):
    payload = {"model": model,
               "messages": [{"role": "user", "content": question}],
               "temperature": 0, "max_tokens": max_tokens}
    if model in REASONING_MODELS:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    for attempt in range(retries):
        try:
            r = requests.post(f"{EVAL2_BASE}/chat/completions", headers=HEADERS,
                              json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"], None
        except Exception as e:
            if attempt == retries - 1:
                return "", f"{type(e).__name__}: {str(e)[:80]}"
            time.sleep(3 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=env.get("EVAL2_MODELS", "DeepSeek-V4.1-Flash,Gemma-4"))
    ap.add_argument("--data", default=str(BASE / "tcmtox-llm" / "data" / "sft" / "test.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=str(BASE / "tcmtox-llm" / "eval" / "results"))
    args = ap.parse_args()

    test = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit > 0:
        test = test[:args.limit]
    print(f"test 样本: {len(test)}", flush=True)

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        safe = model.replace("/", "_").replace(".", "-")
        out_path = Path(args.out_dir) / f"multi_{safe}_zeroshot.jsonl"
        # 断点续跑：已有输出则跳过已完成的 instruction
        done = set()
        if out_path.exists():
            for l in open(out_path, encoding="utf-8"):
                try:
                    done.add(json.loads(l)["instruction"])
                except Exception:
                    pass
            print(f"[{model}] 断点续跑：已完成 {len(done)} 条", flush=True)

        todo = [s for s in test if s["instruction"] not in done]
        t0, errs, n = time.time(), 0, 0
        from concurrent.futures import ThreadPoolExecutor
        CHUNK = 50
        with open(out_path, "a", encoding="utf-8") as f:
            for ci in range(0, len(todo), CHUNK):
                chunk = todo[ci:ci + CHUNK]
                with ThreadPoolExecutor(max_workers=8) as pool:
                    results = list(pool.map(lambda s: (s, chat(model, s["instruction"])), chunk))
                for s, (ans, err) in results:
                    if err:
                        errs += 1
                        if errs <= 3 or errs % 25 == 0:
                            print(f"[warn] {model}: {err}", flush=True)
                    rec = {"instruction": s["instruction"], "input": "",
                           "response": ans, "gold": s["output"],
                           "meta": s["meta"], "condition": "zeroshot",
                           "model": model, "rag": {"used": False}}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                el = time.time() - t0
                print(f"[进度] {model}: {ci + len(chunk)}/{len(todo)} | {el:.0f}s | 错误 {errs}",
                      flush=True)
        print(f"✓ {model} 完成：新增 {n} 条，错误 {errs}，耗时 {time.time()-t0:.0f}s → {out_path}",
              flush=True)


if __name__ == "__main__":
    main()
