# -*- coding: utf-8 -*-
"""
RAG 推理管线：路由 → 检索 → prompt 组装 → vLLM 生成
输出与 run_baseline.py 同格式的 JSONL，可直接用 score.py 评分对比

用法（服务器，swift 环境）：
  conda activate swift && cd ~/codeWorkSpace/tcmtox
  CUDA_VISIBLE_DEVICES=0 python tcmtox-llm/rag/rag_pipeline.py \
      --model saves/tcmtox-merged-v1 --condition zeroshot --tag finetuned-rag \
      --strategy b
评分：
  python tcmtox-llm/eval/score.py tcmtox-llm/eval/results/*finetuned-rag*.jsonl
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "tcmtox-llm" / "rag"))

from router import EntityRouter                 # noqa: E402
from retriever import Retriever, make_embedding, make_reranker, make_store, load_env  # noqa: E402

EMPTY_THINK_RE = re.compile(r"<think>\s*</think>\n*")
CONTEXT_HEADER = ("以下是与问题相关的库内记录，供回答参考；"
                  "若记录不足以回答或问题超出库范围，请如实说明，不要编造。\n\n")

# 策略 B：按问题类型选字段（precise 记录的 payload 字段）
QTYPE_FIELDS = {
    "identifier": ["smiles", "cas", "pubchem", "formula", "weight"],
    "grade": ["grade", "grade_source", "dosage", "precaution"],
    "safety": ["dosage", "precaution", "grade"],
    "mechanism": ["toxicity_types", "description", "uniprot"],
    "formula": ["toxic_herbs", "diseases"],
    "advice": [], "list": [], "other": [],
}


def render_record(rec, strategy, qtype="other"):
    """语料记录 → 注入文本。A=全字段（text 截断 800），B=按问题类型选字段"""
    if strategy == "a":
        body = rec["text"][:800]
    else:
        if rec["entity_type"] == "herb" and qtype in QTYPE_FIELDS:
            fields = QTYPE_FIELDS[qtype]
            pay = rec.get("payload", {})
            parts = [f"{k}: {pay[k]}" for k in fields if pay.get(k)]
            body = f"{rec['title']}。" + "；".join(parts) if parts else rec["text"][:500]
        else:
            body = rec["text"][:500]
    return f"- [{rec['entity_type']}] {body}"


def assemble(question, route_result, semantic_records, strategy):
    """上下文组装：精确记录（全保留）+ 语义记录（去重、截断）"""
    if not route_result.retrieve_semantic and not route_result.precise_records:
        return question, ""          # T6 保护：无上下文
    blocks, seen = [], set()
    for rec in route_result.precise_records + semantic_records:
        rid = rec["id"]
        if rid in seen:
            continue
        seen.add(rid)
        blocks.append(render_record(rec, strategy, route_result.qtype))
    if not blocks:
        return question, ""
    context = CONTEXT_HEADER + "\n".join(f"[{i+1}] {b}" for i, b in enumerate(blocks)) + "\n\n"
    return context + question, context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saves/tcmtox-merged-v1")
    ap.add_argument("--data", default=str(BASE / "tcmtox-llm" / "data" / "sft" / "test.jsonl"))
    ap.add_argument("--condition", default="zeroshot", choices=["zeroshot", "fewshot"])
    ap.add_argument("--strategy", choices=["a", "b"], default="b",
                    help="a=全字段注入，b=按问题类型选字段")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=str(BASE / "tcmtox-llm" / "eval" / "results"))
    ap.add_argument("--tag", default="", help="默认 <模型名>-rag")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--recall-k", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--backend", choices=["qdrant", "numpy"], default="qdrant")
    ap.add_argument("--embed", choices=["local", "api"], default="local")
    ap.add_argument("--rerank", choices=["api", "local", "none"], default="api")
    ap.add_argument("--dry-run", action="store_true", help="只打印路由与组装结果，不加载模型")
    args = ap.parse_args()

    tag = args.tag or (Path(args.model).name + "-rag")
    test = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit > 0:
        test = test[:args.limit]

    env = load_env()
    embedding = make_embedding(args.embed, env)
    reranker = make_reranker(args.rerank, env)
    store = make_store(args.backend, env)
    retriever = Retriever(BASE / "tcmtox-llm" / "rag" / "corpus.jsonl",
                          embedding, store, reranker,
                          recall_k=args.recall_k, top_k=args.top_k)

    # 路由器：实体词典/语料来自 rag 目录
    from router import EntityRouter
    rag_dir = BASE / "tcmtox-llm" / "rag"
    router = EntityRouter(rag_dir / "entities.json", rag_dir / "corpus.jsonl")

    if args.dry_run:
        for s in test[:5]:
            r = router.route(s["instruction"])
            ctx, _ = assemble(s["instruction"], r, [], args.strategy)
            print(f"[{r.qtype}] 检索={'Y' if r.retrieve_semantic else 'N'} "
                  f"精确={len(r.precise_records)} | {s['instruction'][:40]}")
        print("(dry-run 不含语义检索结果；完整效果需加载 embedding 模型)")
        return

    retriever.ensure_index()

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=args.gpu_mem,
              trust_remote_code=True)
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    convs, metas = [], []
    t0 = time.time()
    n_retrieved = 0
    for s in test:
        r = router.route(s["instruction"])
        sem = (retriever.semantic(s["instruction"])
               if r.retrieve_semantic else [])
        prompt, ctx = assemble(s["instruction"], r, sem, args.strategy)
        if ctx:
            n_retrieved += 1
        msgs = [{"role": "user", "content": prompt}]
        convs.append(msgs)
        metas.append((s, r, ctx))

    rendered = [tok.apply_chat_template(c, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
                for c in convs]
    rendered = [EMPTY_THINK_RE.sub("", t) for t in rendered]

    outs = llm.generate(rendered, sp)
    dt = time.time() - t0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}_{args.condition}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s, o, (sample, r, ctx) in zip(test, outs, metas):
            rec = {"instruction": sample["instruction"], "input": "",
                   "response": o.outputs[0].text, "gold": sample["output"],
                   "meta": sample["meta"], "condition": args.condition,
                   "model": args.model,
                   "rag": {"used": bool(ctx), "strategy": args.strategy,
                           "qtype": r.qtype}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"完成：{len(test)} 条，耗时 {dt:.0f}s，注入上下文 {n_retrieved} 条 → {out_path}")
    print(f"下一步：python tcmtox-llm/eval/score.py {out_dir}/*.jsonl --save")


if __name__ == "__main__":
    main()
