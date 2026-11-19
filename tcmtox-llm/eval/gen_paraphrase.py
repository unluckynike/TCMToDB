# -*- coding: utf-8 -*-
"""
鲁棒性评测管线①：问法改写——把 test 集指令改写成真人口语问法（模板外问法），
用于量化"模板合成分 → 真实问句分布"的性能落差（论文鲁棒性小节 / 合集下篇前置测量）。

改写走 newapi 网关 DeepSeek（nothink），与 run_multillm.py 同构：8 线程、断点续跑。

用法（Mac）：
  python3 tcmtox-llm/eval/gen_paraphrase.py            # 全量 1313
  python3 tcmtox-llm/eval/gen_paraphrase.py --limit 5  # 冒烟

输出：data/sft/test_para.jsonl（instruction=改写问法，output/meta 沿用原样本，
meta.original_instruction 留作配对键；改写失败/原样返回的样本跳过不落盘）
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

PROMPT = """你是数据改写助手。把下面的中医药问题改写成普通用户更口语化的问法：
- 只改问法，不改问题的实体、意图和边界（该拒答的问题改写后仍应被拒答，需要谨慎回答的仍需谨慎）
- 不要回答问题，不要解释，只输出改写后的问题本身
- 长度与原问题相当，保持原问题的语言（中文问题改中文，英文问题改英文）

原问题：{q}"""


def paraphrase(q, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post(f"{EVAL2_BASE}/chat/completions", headers=HEADERS,
                              json={"model": "DeepSeek-V4.1-Flash",
                                    "messages": [{"role": "user", "content": PROMPT.format(q=q)}],
                                    "temperature": 0.7, "max_tokens": 256,
                                    "chat_template_kwargs": {"enable_thinking": False}},
                              timeout=120)
            r.raise_for_status()
            out = (r.json()["choices"][0]["message"]["content"] or "").strip()
            return out, None
        except Exception as e:
            if attempt == retries - 1:
                return "", f"{type(e).__name__}: {str(e)[:80]}"
            time.sleep(3 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE / "tcmtox-llm" / "data" / "sft" / "test.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(BASE / "tcmtox-llm" / "data" / "sft" / "test_para.jsonl"))
    args = ap.parse_args()

    test = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit > 0:
        test = test[:args.limit]
    print(f"原样本: {len(test)}", flush=True)

    out_path = Path(args.out)
    # 断点续跑：已完成改写的 original_instruction 跳过
    done = set()
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(l)["meta"]["original_instruction"])
            except Exception:
                pass
        print(f"断点续跑：已完成 {len(done)} 条", flush=True)

    todo = [s for s in test if s["instruction"] not in done]
    t0, errs, skipped, n = time.time(), 0, 0, 0
    from concurrent.futures import ThreadPoolExecutor
    CHUNK = 50
    with open(out_path, "a", encoding="utf-8") as f:
        for ci in range(0, len(todo), CHUNK):
            chunk = todo[ci:ci + CHUNK]
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda s: (s, paraphrase(s["instruction"])), chunk))
            for s, (para, err) in results:
                if err:
                    errs += 1
                    if errs <= 3 or errs % 25 == 0:
                        print(f"[warn] {err}", flush=True)
                    continue  # 失败不落盘，下轮断点续跑会重试
                # 原样返回（模型没改）或空输出 = 退化改写，跳过
                if not para or para == s["instruction"]:
                    skipped += 1
                    continue
                rec = {"instruction": para, "input": "", "output": s["output"],
                       "meta": {**s["meta"], "original_instruction": s["instruction"]}}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            el = time.time() - t0
            print(f"[进度] {ci + len(chunk)}/{len(todo)} | {el:.0f}s | 错误 {errs} | 退化跳过 {skipped}",
                  flush=True)
    print(f"✓ 完成：改写 {n} 条，错误 {errs}，退化跳过 {skipped}，耗时 {time.time()-t0:.0f}s → {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
