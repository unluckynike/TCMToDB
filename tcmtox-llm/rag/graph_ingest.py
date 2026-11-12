# -*- coding: utf-8 -*-
"""
Neo4j mini-graph 灌库：用现有数据构建毒性中药知识图谱（不等关系表）

图结构：
  节点  (:Herb {cn_name, pinyin, grade, grade_source, dosage, precaution, pregnancy})
  节点  (:Formula {cn_name, en_name, diseases})
  边    (:Formula)-[:CONTAINS {dose, grade, prefix}]->(:Herb)      ← 处方解析产物 4,522 条
  边    (:Herb)-[:CONFLICT_WITH]->(:Herb)                          ← 药典"不宜与X同用"解析（十八反十九畏）
  属性  Herb.pregnancy = 禁用/慎用（来自药典注意事项）

用法（服务器，swift 环境，需先 docker compose up 启动 Neo4j）：
  conda activate swift
  python tcmtox-llm/rag/graph_ingest.py
"""
import re
import sys
from pathlib import Path

from neo4j import GraphDatabase

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retriever import load_env                                   # noqa: E402
import csv

def norm(v):
    return (v or "").strip()

def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    env = load_env(Path(__file__).resolve().parent / ".env")
    uri = env.get("NEO4J_URI", "bolt://localhost:7687")
    driver = GraphDatabase.driver(uri, auth=(env.get("NEO4J_USER", "neo4j"),
                                             env.get("NEO4J_PASSWORD", "")))
    driver.verify_connectivity()
    print(f"✓ Neo4j 已连接: {uri}")

    herbs = load_csv(BASE / "tcmtox_data" / "TCMTox_herb_clean.csv")
    yd = load_csv(BASE / "tcmtox_data" / "yd2020_toxic83.csv")
    formulas = load_csv(BASE / "tcmtox_data" / "TCMTox_formula.csv")
    fh = load_csv(BASE / "tcmtox-llm" / "data" / "formula_herbs.csv")

    with driver.session() as s:
        # ---------- 约束（幂等） ----------
        for stmt in ["CREATE CONSTRAINT herb_cn IF NOT EXISTS FOR (h:Herb) REQUIRE h.cn_name IS UNIQUE",
                     "CREATE CONSTRAINT formula_cn IF NOT EXISTS FOR (f:Formula) REQUIRE f.cn_name IS UNIQUE"]:
            s.run(stmt)

        # ---------- Herb 节点（229 + 药典补充） ----------
        herb_rows = []
        for h in herbs:
            cn = norm(h["Herb_cn_name"])
            grade_src = norm(h["Toxicity_source"])
            herb_rows.append({
                "cn_name": cn, "pinyin": norm(h["Herb_pinyin_name"]),
                "grade": norm(h["Toxicity_grade"]),
                "grade_source": "药典2020" if "药典" in grade_src else "TCMToxDB",
                "properties": norm(h["Properties"]),
                "meridians": norm(h["Meridians"]),
            })
        preg_forbidden, preg_caution = [], []
        for y in yd:
            name, prec = norm(y["yd_name"]), norm(y["precaution"])
            row = {"cn_name": name, "grade": norm(y["grade"]),
                   "dosage": norm(y["dosage"]), "precaution": prec,
                   "grade_source": "药典2020"}
            if name not in {r["cn_name"] for r in herb_rows}:
                herb_rows.append(row)                     # 药典独有品种补节点
            if "孕妇禁用" in prec or "孕妇禁服" in prec:
                preg_forbidden.append(name)
            elif "孕妇慎用" in prec:
                preg_caution.append(name)
        s.run("UNWIND $rows AS r MERGE (h:Herb {cn_name: r.cn_name}) "
              "SET h += r", rows=herb_rows)
        print(f"✓ Herb 节点: {len(herb_rows)}")

        # 孕妇禁忌属性（回填到对应节点）
        s.run("UNWIND $names AS n MATCH (h:Herb {cn_name: n}) SET h.pregnancy = '禁用'",
              names=preg_forbidden)
        s.run("UNWIND $names AS n MATCH (h:Herb {cn_name: n}) SET h.pregnancy = '慎用'",
              names=preg_caution)
        print(f"✓ 孕妇禁用 {len(preg_forbidden)} / 慎用 {len(preg_caution)}")

        # ---------- Formula 节点 + CONTAINS 边 ----------
        formula_rows, contains_rows = [], []
        for fm in formulas:
            name = norm(fm["Form_name"])
            if name:
                formula_rows.append({"cn_name": name,
                                     "en_name": norm(fm["Form_name_en"]),
                                     "diseases": norm(fm["Diseases"])})
        s.run("UNWIND $rows AS r MERGE (f:Formula {cn_name: r.cn_name}) SET f += r",
              rows=formula_rows)
        for r in fh:
            dose = f"{r['dosage_value']}{r['dosage_unit']}" if norm(r["dosage_value"]) else None
            contains_rows.append({"formula": norm(r["Form_name"]),
                                  "herb": norm(r["herb_cn_name"]),
                                  "prefix": norm(r["prefix_cn"]),
                                  "dose": dose,
                                  "grade": norm(r["Toxicity_grade"])})
        s.run("UNWIND $rows AS r "
              "MATCH (f:Formula {cn_name: r.formula}) "
              "MERGE (h:Herb {cn_name: r.herb}) "          # 方剂中的药材可能在229表之外，补节点
              "MERGE (f)-[c:CONTAINS]->(h) "
              "SET c.dose = r.dose, c.grade = r.grade, c.prefix = r.prefix",
              rows=contains_rows)
        print(f"✓ Formula 节点: {len(formula_rows)} ｜ CONTAINS 边: {len(contains_rows)}")

        # ---------- CONFLICT_WITH 边（药典"不宜与X同用"解析） ----------
        conflict_rows = []
        for y in yd:
            prec = norm(y["precaution"])
            m = re.search(r"不宜与(.+?)同用", prec)
            if not m:
                continue
            for agent in [a.strip() for a in re.split(r"[、，,]", m.group(1)) if a.strip()]:
                conflict_rows.append({"herb": norm(y["yd_name"]), "agent": agent})
        s.run("UNWIND $rows AS r "
              "MERGE (a:Herb {cn_name: r.agent}) "          # 禁忌对象可能不在229表，补节点
              "WITH a, r MATCH (h:Herb {cn_name: r.herb}) "
              "MERGE (h)-[:CONFLICT_WITH {source: '药典2020注意事项'}]->(a)",
              rows=conflict_rows)
        print(f"✓ CONFLICT_WITH 边: {len(conflict_rows)}")

        # ---------- 验证查询 ----------
        probe = s.run("MATCH (f:Formula)-[c:CONTAINS]->(h:Herb {grade:'大毒'}) "
                      "RETURN f.cn_name AS f, h.cn_name AS h, c.dose AS d LIMIT 5").data()
        print("\n== 验证：含大毒药材的方剂（前5）==")
        for row in probe:
            print(f"  {row['f']} ← {row['h']} ({row['d']})")

    driver.close()
    print("\n✓ 图谱灌库完成。示例查询（浏览器 http://localhost:7474）：")
    print("  MATCH (f:Formula)-[:CONTAINS]->(h:Herb {grade:'大毒'}) RETURN f, h LIMIT 20")
    print("  MATCH (h:Herb)-[:CONFLICT_WITH]->(a:Herb {cn_name:'半夏'}) RETURN h, a")


if __name__ == "__main__":
    main()
