# -*- coding: utf-8 -*-
"""
独立建索引：语料全量向量化 → Qdrant 灌库（一次性，~19k 条）
避免在评测/应用运行时隐式触发长耗时索引。

用法：
  python tcmtox-llm/rag/build_index.py                # 按 .env 的 provider
  python tcmtox-llm/rag/build_index.py --provider api # 强制走网关 bge-m3
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retriever import Retriever, make_embedding, make_reranker, make_store, load_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["local", "api"], default="local")
    ap.add_argument("--rerank", choices=["api", "local", "none"], default="none",
                    help="建索引不需要 rerank，默认跳过")
    ap.add_argument("--backend", choices=["qdrant", "numpy"], default="qdrant")
    args = ap.parse_args()

    env = load_env()
    embedding = make_embedding(args.provider, env)
    store = make_store(args.backend, env)
    retriever = Retriever(Path(__file__).resolve().parent / "corpus.jsonl",
                          embedding, store, reranker=None)

    t0 = time.time()
    retriever.ensure_index()
    print(f"✓ 索引就绪，耗时 {time.time()-t0:.0f}s（provider={embedding.name}，{embedding.dim} 维）")


if __name__ == "__main__":
    main()
