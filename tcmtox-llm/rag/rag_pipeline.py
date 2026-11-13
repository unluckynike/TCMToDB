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
CONTEXT_HEADER_SAFETY = (CONTEXT_HEADER
                         + "回答请涵盖两方面：①该药材的毒性分级；②与问题相关的用法用量与注意事项。\n\n")

# 策略 B：按问题类型选字段（precise 记录的 payload 字段）
QTYPE_FIELDS = {
    "identifier": ["smiles", "cas", "pubchem", "formula", "weight"],
    "grade": ["grade", "grade_source", "dosage", "precaution"],
    "safety": ["grade", "dosage", "precaution"],
    "mechanism": ["toxicity_types", "description", "uniprot"],
    "formula": ["toxic_herbs", "diseases"],
    "advice": [], "list": [], "other": [],
}


def sanitize(text):
    """注入卫生：剥离"原标XX"弃用标签说明——防止弃用标签经上下文复活
    （实测：RAG 上下文含'原标Slightly Toxic'会诱导模型输出旧标签，分级 90.5→66.7 的主因之一）"""
    return re.sub(r"，?原标[^；）\n]*", "", str(text))


def render_record(rec, strategy, qtype="other"):
    """语料记录 → 注入文本。A=全字段（text 截断 800），B=按问题类型选字段"""
    if strategy == "a":
        body = rec["text"][:800]
    else:
        if rec["entity_type"] == "herb" and qtype in QTYPE_FIELDS:
            fields = QTYPE_FIELDS[qtype]
            pay = rec.get("payload", {})
            parts = [f"{k}: {sanitize(pay[k])}" for k in fields if pay.get(k)]
            body = f"{rec['title']}。" + "；".join(parts) if parts else sanitize(rec["text"][:500])
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
    header = (CONTEXT_HEADER_SAFETY if route_result.qtype == "safety"
              else CONTEXT_HEADER)
    context = header + "\n".join(f"[{i+1}] {b}" for i, b in enumerate(blocks)) + "\n\n"
    return context + question, context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saves/tcmtox-merged-v1")
    ap.add_argument("--data", default=str(BASE / "tcmtox-llm" / "data" / "sft" / "test.jsonl"))
    ap.add_argument("--condition", default="zeroshot", choices=["zeroshot", "fewshot"])
    ap.add_argument("--strategy", choices=["a", "b"], default="b",
                    help="a=全字段注入，b=按问题类型选字段")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-first", type=int, default=0,
                    help="跳过数据集前 N 条（配合多次挖掘避免负例重复）")
    ap.add_argument("--out-dir", default=str(BASE / "tcmtox-llm" / "eval" / "results"))
    ap.add_argument("--tag", default="", help="默认 <模型名>-rag")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--recall-k", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--backend", choices=["qdrant", "numpy"], default="qdrant")
    ap.add_argument("--embed", choices=["local", "api"], default="api")
    ap.add_argument("--rerank", choices=["api", "local", "none"], default="api")
    ap.add_argument("--llm-mode", choices=["api", "local"], default="api",
                    help="api=调 GPU 服务器 vllm serve 的 OpenAI 兼容接口；local=进程内加载")
    ap.add_argument("--dry-run", action="store_true", help="只打印路由与组装结果，不加载模型")
    args = ap.parse_args()

    tag = args.tag or (Path(args.model).name + "-rag")
    test = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.skip_first > 0:
        test = test[args.skip_first:]
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
        convs.append([{"role": "user", "content": prompt}])
        metas.append((s, r, ctx))

    # prompt 渲染：api 模式直接用与 qwen3_5_nothink 训练模板逐字符一致的常量
    # 格式（dry-run 实测验证过），零 transformers 依赖；local 模式走 transformers
    if args.llm_mode == "api":
        rendered = ["<|im_start|>user\n" + c[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n"
                    for c in convs]
        rendered = [EMPTY_THINK_RE.sub("", t) for t in rendered]

    if args.llm_mode == "api":
        # GPU 服务器 vllm serve 常驻，走 /v1/completions（prompt 与训练格式逐字符一致）
        import requests as rq
        from concurrent.futures import ThreadPoolExecutor
        base = env.get("LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
        headers = {"Authorization": f"Bearer {env.get('LLM_API_KEY', 'EMPTY')}"}
        model_name = env.get("LLM_MODEL", "tcmtox-qa")

        def gen_one(prompt):
            r = rq.post(f"{base}/completions", headers=headers,
                        json={"model": model_name, "prompt": prompt,
                              "temperature": 0, "max_tokens": args.max_tokens},
                        timeout=180)
            r.raise_for_status()
            return r.json()["choices"][0]["text"]

        with ThreadPoolExecutor(max_workers=8) as pool:
            texts = list(pool.map(gen_one, rendered))
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        rendered = [tok.apply_chat_template(c, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                    for c in convs]
        rendered = [EMPTY_THINK_RE.sub("", t2) for t2 in rendered]
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model, dtype="bfloat16",
                  max_model_len=4096, gpu_memory_utilization=args.gpu_mem,
                  trust_remote_code=True)
        sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)
        outs = llm.generate(rendered, sp)
        texts = [o.outputs[0].text for o in outs]
    dt = time.time() - t0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}_{args.condition}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s, (sample, r, ctx), o in zip(test, metas, texts):
            rec = {"instruction": sample["instruction"], "input": "",
                   "response": o, "gold": sample["output"],
                   "meta": sample["meta"], "condition": args.condition,
                   "model": args.model,
                   "rag": {"used": bool(ctx), "strategy": args.strategy,
                           "qtype": r.qtype}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"完成：{len(test)} 条，耗时 {dt:.0f}s，注入上下文 {n_retrieved} 条 → {out_path}")
    print(f"下一步：python tcmtox-llm/eval/score.py {out_dir}/*.jsonl --save")


if __name__ == "__main__":
    main()
