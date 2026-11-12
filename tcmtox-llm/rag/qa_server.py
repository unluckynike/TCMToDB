# -*- coding: utf-8 -*-
"""
TCMToxDB 毒性中药智能问答服务：三路检索可视化 + 知识图谱探索 + 流式生成
跑在本机 (localhost:7860)，零 CORS 问题：
  · 生成     → GPU 服务器 vLLM（OpenAI 兼容接口，地址见 .env）流式透传
  · 语义检索 → Qdrant 服务实例 + embedding/rerank 网关
  · 精确查表 → 语料字典
  · 图谱     → Neo4j 实例 Cypher 子图

启动：conda activate swift（或 base）&& python tcmtox-llm/rag/qa_server.py
访问：http://localhost:7860
"""
import json
import sys
from pathlib import Path

import requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router import EntityRouter                                    # noqa: E402
from retriever import Retriever, make_embedding, make_reranker, \
    make_store, load_env                                          # noqa: E402

BASE = Path(__file__).resolve().parents[2]
RAG_DIR = Path(__file__).resolve().parent
env = load_env(RAG_DIR / ".env")

app = FastAPI(title="TCMToxDB 毒性中药智能问答", docs_url=None, redoc_url=None)

# ---------- 单例服务 ----------
router = EntityRouter(RAG_DIR / "entities.json", RAG_DIR / "corpus.jsonl")
embedding = make_embedding(env.get("EMBEDDING_PROVIDER", "api"), env)
reranker = make_reranker(env.get("RERANK_PROVIDER", "api"), env)
store = make_store(env.get("RAG_BACKEND", "qdrant"), env)
retriever = Retriever(RAG_DIR / "corpus.jsonl", embedding, store, reranker,
                      recall_k=50, top_k=5)

from neo4j import GraphDatabase                                    # noqa: E402
neo4j_driver = GraphDatabase.driver(env.get("NEO4J_URI", "bolt://localhost:7687"),
                                    auth=(env.get("NEO4J_USER", "neo4j"),
                                          env.get("NEO4J_PASSWORD", "")))
LLM_BASE = env.get("LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
LLM_MODEL = env.get("LLM_MODEL", "tcmtox-qa")
LLM_KEY = env.get("LLM_API_KEY", "EMPTY")

GRADE_COLORS = {"大毒": "#e74c3c", "有毒": "#e67e22", "小毒": "#f1c40f"}


def element_id(node):
    return node.element_id


# ---------- 静态页 ----------
@app.get("/")
def index():
    return FileResponse(RAG_DIR / "qa.html")


@app.get("/api/health")
def health():
    ok = {}
    try:
        ok["vllm"] = requests.get(f"{LLM_BASE}/models", timeout=4).ok
    except Exception:
        ok["vllm"] = False
    try:
        ok["qdrant"] = requests.get(
            f"http://{env.get('QDRANT_HOST')}:6333/collections", timeout=4).ok
    except Exception:
        ok["qdrant"] = False
    try:
        with neo4j_driver.session() as s:
            s.run("RETURN 1")
        ok["neo4j"] = True
    except Exception:
        ok["neo4j"] = False
    try:
        ok["newapi"] = requests.get(f"{env.get('NEWAPI_BASE')}/v1/models",
                                    headers={"Authorization": f"Bearer {env.get('NEWAPI_KEY')}"},
                                    timeout=4).ok
    except Exception:
        ok["newapi"] = False
    return ok


# ---------- 知识图谱 ----------
def _node_json(node, extra_color=None):
    props = dict(node)
    label = props.get("cn_name") or props.get("gene") or node.element_id
    etype = "formula" if "Formula" in node.labels else "herb"
    color = "#1abc9c" if etype == "formula" else (
        extra_color or GRADE_COLORS.get(props.get("grade"), "#95a5a6"))
    return {"id": element_id(node), "label": label, "group": etype,
            "color": color, "props": props}


def _rel_json(rel):
    return {"id": str(rel.id), "from": rel.start_node.element_id,
            "to": rel.end_node.element_id, "type": rel.type,
            "label": rel.get("dose") or "", "arrows": "to"}


@app.get("/api/graph")
def graph(grade: str = "大毒", limit: int = 220):
    """按分级展示：Formula -CONTAINS-> Herb(指定分级) + 这些药材间的冲突边"""
    with neo4j_driver.session() as s:
        rels = list(s.run("MATCH (f:Formula)-[r:CONTAINS]->(h:Herb {grade:$g}) "
                          "RETURN f, r, h LIMIT $lim", g=grade, lim=limit))
        herb_ids = list({element_id(x["h"]) for x in rels})
        conflicts = []
        if herb_ids:
            conflicts = list(s.run("MATCH (a:Herb)-[c:CONFLICT_WITH]->(b:Herb) "
                                   "WHERE elementId(a) IN $ids AND elementId(b) IN $ids "
                                   "RETURN a, c, b", ids=herb_ids))
    nodes, edges, seen_n, seen_e = {}, [], set(), set()
    for row in rels:
        for key in ("f", "h"):
            nid = element_id(row[key])
            if nid not in seen_n:
                nodes[nid] = _node_json(row[key])
                seen_n.add(nid)
        rid = str(row["r"].id)
        if rid not in seen_e:
            edges.append({"id": rid, "from": element_id(row["r"].start_node),
                          "to": element_id(row["r"].end_node),
                          "label": row["r"].get("dose") or "",
                          "color": "#16a085", "arrows": "to"})
            seen_e.add(rid)
    for row in conflicts:
        for key in ("a", "b"):
            nid = element_id(row[key])
            if nid not in seen_n:
                nodes[nid] = _node_json(row[key], extra_color="#c0392b")
                seen_n.add(nid)
        rid = str(row["c"].id)
        if rid not in seen_e:
            edges.append({"id": rid, "from": element_id(row["c"].start_node),
                          "to": element_id(row["c"].end_node),
                          "color": "#e74c3c", "dashes": True, "arrows": "to",
                          "label": "配伍禁忌"})
            seen_e.add(rid)
    return {"nodes": list(nodes.values()), "edges": edges,
            "grade": grade, "counts": {"nodes": len(nodes), "edges": len(edges)}}


@app.get("/api/graph/neighbor")
def graph_neighbor(name: str):
    """以某实体为中心的一跳邻域（药材/方剂通用）"""
    with neo4j_driver.session() as s:
        rows = list(s.run("MATCH (c) WHERE c.cn_name = $name "
                          "OPTIONAL MATCH (c)-[r]-(n) "
                          "RETURN c, r, n LIMIT 80", name=name))
    nodes, edges, seen_n, seen_e = {}, [], set(), set()
    center = None
    for row in rows:
        cid = element_id(row["c"])
        center = cid
        if cid not in seen_n:
            nodes[cid] = _node_json(row["c"])
            seen_n.add(cid)
        if row["r"] is None:
            continue
        other = row["n"]
        nid = element_id(other)
        if nid not in seen_n:
            nodes[nid] = _node_json(other)
            seen_n.add(nid)
        rid = str(row["r"].id)
        if rid not in seen_e:
            edges.append({"id": rid, "from": element_id(row["r"].start_node),
                          "to": element_id(row["r"].end_node),
                          "type": row["r"].type, "color": "#16a085",
                          "arrows": "to"})
            seen_e.add(rid)
    return {"nodes": list(nodes.values()), "edges": edges, "center": center}


@app.get("/api/stats")
def stats():
    with neo4j_driver.session() as s:
        herbs = s.run("MATCH (h:Herb) RETURN h.grade AS g, count(*) AS c").data()
        formulas = s.run("MATCH (f:Formula) RETURN count(*) AS c").single()["c"]
        conflicts = s.run("MATCH (:Herb)-[r:CONFLICT_WITH]->(:Herb) RETURN count(*) AS c").single()["c"]
        contains = s.run("MATCH (:Formula)-[r:CONTAINS]->(:Herb) RETURN count(*) AS c").single()["c"]
    return {"herbs_by_grade": {r["g"] or "无分级": r["c"] for r in herbs},
            "formulas": formulas, "conflict_edges": conflicts, "contains_edges": contains}


# ---------- RAG 问答（SSE 流式） ----------
class AskReq(BaseModel):
    question: str


def _compact(rec):
    return {"entity_type": rec.get("entity_type"),
            "title": rec.get("title"),
            "snippet": rec.get("text", "")[:180],
            "grade": rec.get("payload", {}).get("grade", ""),
            "dose": rec.get("payload", {}).get("dosage", ""),
            "precaution": rec.get("payload", {}).get("precaution", "")}


@app.post("/api/ask")
def ask(req: AskReq):
    q = req.question.strip()

    def sse():
        # 阶段 1：路由
        route = router.route(q)
        yield ("data: " + json.dumps({
            "type": "route", "qtype": route.qtype,
            "entities": [f"{t}:{n}" for t, n in route.entities],
            "retrieve": route.retrieve_semantic,
            "precise": len(route.precise_records),
            "reason": route.reason}, ensure_ascii=False) + "\n\n")

        # 阶段 2：检索
        sem = retriever.semantic(q) if route.retrieve_semantic else []
        records = []
        seen = set()
        for rec in route.precise_records + sem:
            if rec["id"] not in seen:
                records.append(rec)
                seen.add(rec["id"])
        yield ("data: " + json.dumps({"type": "records",
                                      "records": [_compact(r) for r in records[:6]]},
                                     ensure_ascii=False) + "\n\n")

        # 阶段 3：prompt 组装（与评测管线一致的口径）
        from rag_pipeline import assemble, QTYPE_FIELDS
        blocks, seen = [], set()
        for rec in records[:6]:
            if rec["id"] not in seen:
                seen.add(rec["id"])
                blocks.append(f"- [{rec.get('entity_type')}] {rec.get('text', '')[:500]}")
        context = ""
        if blocks and route.retrieve_semantic or route.precise_records:
            header = ("以下是与问题相关的库内记录，供回答参考；"
                      "若记录不足以回答或问题超出库范围，请如实说明，不要编造。"
                      "（如涉及毒性分级，请先明确说明分级；"
                      "请直接给出简洁的最终答案，不要展示推理过程）\n\n")
            context = header + "\n".join(blocks) + "\n\n"
        messages = [{"role": "user", "content": context + q}]

        # 阶段 4：vLLM 流式生成透传
        acc = ""
        try:
            resp = requests.post(f"{LLM_BASE}/chat/completions",
                                 headers={"Authorization": f"Bearer {LLM_KEY}"},
                                 json={"model": LLM_MODEL, "messages": messages,
                                       "temperature": 0, "max_tokens": 512,
                                       "stream": True,
                                       "chat_template_kwargs": {"enable_thinking": False}},
                                 stream=True, timeout=180)
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"].get("content")
                except Exception:
                    delta = None
                if delta:
                    acc += delta
                    yield ("data: " + json.dumps({"type": "token", "text": delta},
                                                 ensure_ascii=False) + "\n\n")
        except Exception as e:
            yield ("data: " + json.dumps({"type": "error",
                                          "text": f"生成服务异常: {e}"},
                                         ensure_ascii=False) + "\n\n")
        yield ("data: " + json.dumps({"type": "done", "answer": acc},
                                     ensure_ascii=False) + "\n\n")

    return StreamingResponse(sse(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
