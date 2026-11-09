# -*- coding: utf-8 -*-
"""
RAG 语料构建：4 张表 → 检索语料（每实体一条文档）+ 实体词典（路由器用）

产物（tcmtox-llm/rag/，均为衍生数据不入库）：
  corpus.jsonl   检索语料：id / entity_type / title / text（embedding 输入）/ payload（结构化字段，注入 prompt）
  entities.json  实体词典：{"herbs": {中文名: {...}}, "formulas": {...}, "ingredients": {...}}

语料规模：herb 229 + 药典补充 ~13 + ingredient 9377 + target 7885 + formula 1534 ≈ 1.9 万条
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "tcmtox_data"
OUT = BASE / "tcmtox-llm" / "rag"

def norm(v):
    return (v or "").strip()

def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def sec(label, value):
    """拼文档字段的工具：空值跳过"""
    value = norm(value)
    return f"{label}：{value}；" if value else ""

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = []          # 检索语料
    entities = {"herbs": {}, "formulas": {}, "ingredients": {}}

    # ---------------- 1) 药材（清洗表 229 味，合并药典用量/注意事项） ----------------
    herbs = load_csv(DATA / "TCMTox_herb_clean.csv")
    yd_all = {norm(y["yd_name"]): y for y in load_csv(DATA / "yd2020_toxic83.csv")}
    for h in herbs:
        cn = norm(h["Herb_cn_name"])
        py = " ".join(norm(h["Herb_pinyin_name"]).split())
        grade = norm(h["Toxicity_grade"])
        grade_src = norm(h["Toxicity_source"])
        grade_txt = f"毒性分级：{grade}（依据{('《中国药典》2020年版' if '药典' in grade_src else 'TCMToxDB库内标注')}）" if grade else "毒性分级：无分级记录（《中国药典》2020年版未列入分级名单，库内亦无标注）"
        y = yd_all.get(cn, {"dosage": "", "precaution": ""})   # 合并药典用法用量/注意事项
        text = (f"{cn}（{py}）。{grade_txt}。"
                + sec("性味归经", h["Properties"] and f"{h['Properties']}｜{h['Meridians']}")
                + sec("用法用量", y["dosage"])
                + sec("注意事项", y["precaution"])
                + sec("功效", h["Function"])
                + sec("主治", h["Indication"])
                + sec("临床毒性表现", h["Clinical_manifestations"])
                + sec("药用部位", h["UsePart"])
                + sec("拉丁学名", h["Herb_latin_name"]))
        payload = {"entity_type": "herb", "cn_name": cn, "pinyin": py,
                   "grade": grade, "grade_source": grade_src,
                   "dosage": norm(y["dosage"]), "precaution": norm(y["precaution"]),
                   "herb_id": norm(h["HERB_id"])}
        corpus.append({"id": f"herb:{norm(h['HERB_id'])}", "entity_type": "herb",
                       "title": cn, "text": text, "payload": payload})
        entities["herbs"][cn] = {"pinyin": py, "herb_id": norm(h["HERB_id"]),
                                 "grade": grade, "dosage": norm(y["dosage"]),
                                 "precaution": norm(y["precaution"])}

    # ---------------- 2) 药典补充（83 种中不在 229 表的，如红粉/金钱白花蛇/蕲蛇等） ----------------
    yd = load_csv(DATA / "yd2020_toxic83.csv")
    herb_cns = {norm(h["Herb_cn_name"]) for h in herbs}
    extra = 0
    for y in yd:
        name = norm(y["yd_name"])
        if name in herb_cns:
            continue  # 已在药材文档中覆盖
        text = (f"{name}（《中国药典》2020年版毒性中药）。"
                + sec("毒性分级", y["grade"])
                + sec("用法用量", y["dosage"])
                + sec("注意事项", y["precaution"]))
        payload = {"entity_type": "herb", "cn_name": name, "grade": norm(y["grade"]),
                   "grade_source": "《中国药典》2020年版", "herb_id": ""}
        corpus.append({"id": f"yd:{name}", "entity_type": "herb",
                       "title": name, "text": text, "payload": payload})
        entities["herbs"][name] = {"pinyin": "", "herb_id": "", "grade": norm(y["grade"])}
        extra += 1

    # ---------------- 3) 化学成分（9377） ----------------
    ingredients = load_csv(DATA / "TCMTox_ingredient.csv")
    for x in ingredients:
        name = norm(x["Name"])
        if not name:
            continue
        text = (f"化学成分 {name}。"
                + sec("分子式", x["Formula"])
                + sec("SMILES", x["Simles"])
                + sec("CAS号", x["CAS_ID"])
                + sec("分子量", x["Weight"])
                + sec("PubChem CID", x["PubChem_ID"])
                + sec("别名", x["Alias"][:200]))
        payload = {"entity_type": "ingredient", "name": name,
                   "smiles": norm(x["Simles"]), "cas": norm(x["CAS_ID"]),
                   "pubchem": norm(x["PubChem_ID"]), "formula": norm(x["Formula"])}
        corpus.append({"id": f"ingredient:{name}", "entity_type": "ingredient",
                       "title": name, "text": text, "payload": payload})
        entities["ingredients"][name] = {"cas": norm(x["CAS_ID"])}

    # ---------------- 4) 蛋白靶点（7885） ----------------
    TOX_CN = {"Neurotoxicity": "神经毒性", "Nephrotoxicity": "肾毒性", "Hepatotoxicity": "肝毒性",
              "Reproductive toxicity": "生殖毒性", "Cardiotoxicity": "心脏毒性",
              "Thyrotoxicity": "甲状腺毒性", "Pulmonary toxicity": "肺毒性", "Ototoxicity": "耳毒性",
              "Immunotoxicity": "免疫毒性", "Embryotoxicity": "胚胎毒性", "Myelotoxicity": "骨髓毒性",
              "Myotoxicity": "肌肉毒性", "Gastrointestinal toxicity": "胃肠道毒性",
              "Ocular toxicity": "眼毒性", "Osteotoxicity": "骨毒性", "Dermatotoxicity": "皮肤毒性"}
    targets = load_csv(DATA / "TCMTox_target.csv")
    for t in targets:
        gene = norm(t["Gene_name"])
        if not gene:
            continue
        tox = [TOX_CN.get(x.strip(), x.strip()) for x in norm(t["Toxicity"]).split(";") if x.strip()]
        desc = norm(t["Description"])
        text = (f"蛋白靶点 {gene}"
                + (f"（{desc}）" if desc else "")
                + "。"
                + (sec("毒性类型标注", "、".join(tox)) if tox else "无毒性类型标注。")
                + sec("UniProt", t["UniProt_ID"]))
        payload = {"entity_type": "target", "gene": gene,
                   "uniprot": norm(t["UniProt_ID"]), "toxicity_types": tox,
                   "description": desc}
        corpus.append({"id": f"target:{gene}", "entity_type": "target",
                       "title": gene, "text": text, "payload": payload})

    # ---------------- 5) 方剂（1534 有处方者，附解析出的毒性药材清单） ----------------
    formulas = load_csv(DATA / "TCMTox_formula.csv")
    fh = {}   # 方名 -> [(毒性药材+前缀+分级+剂量)]
    for r in load_csv(BASE / "tcmtox-llm" / "data" / "formula_herbs.csv"):
        if norm(r["Toxicity_grade"]):
            dose = f"{r['dosage_value']}{r['dosage_unit']}" if r["dosage_value"] else "剂量未标注"
            pre = f"{r['prefix_cn']}" if norm(r["prefix_pinyin"]) else ""
            fh.setdefault(r["Form_name"], []).append(
                f"{pre}{r['herb_cn_name']}（{r['Toxicity_grade']}，{dose}）")
    for fm in formulas:
        name = norm(fm["Form_name"])
        if not norm(fm["Prescription"]):
            continue
        tox_list = "、".join(fh.get(name, [])) or "未检出分级毒性药材"
        text = (f"方剂「{name}」（{norm(fm['Form_name_en'])}）。"
                + sec("剂型", fm["Type"])
                + sec("主治", fm["Diseases"])
                + sec("适应症", fm["Indications"])
                + sec("功效", fm["Efficacy"])
                + f"处方（原文拼音）：{norm(fm['Prescription'])}。"
                + f"解析出的分级毒性药材：{tox_list}。"
                + "注：处方剂量为制备总投料量，不代表单次服用量。")
        payload = {"entity_type": "formula", "cn_name": name,
                   "en_name": norm(fm["Form_name_en"]),
                   "diseases": norm(fm["Diseases"]), "toxic_herbs": fh.get(name, [])}
        corpus.append({"id": f"formula:{name}", "entity_type": "formula",
                       "title": name, "text": text, "payload": payload})
        entities["formulas"][name] = {"en_name": norm(fm["Form_name_en"])}

    # ---------------- 5.5) 反向索引文档：药材 → 含其方剂列表（T7d 语义可检索） ----------------
    herb_to_formulas = {}
    for fname, herbs_in in fh.items():
        for h in herbs_in:
            cn = re.search(r"(.+?)（", h)  # "牵牛子（炒，有毒，200g）" → 牵牛子
            key = cn.group(1) if cn else h
            herb_to_formulas.setdefault(key, []).append(fname)
    rev = 0
    for herb_cn, forms in sorted(herb_to_formulas.items(), key=lambda kv: -len(kv[1])):
        if len(forms) < 3:
            continue  # 出现太少的药材不需要反向文档
        forms_sorted = sorted(set(forms))
        text = (f"含有{herb_cn}的方剂共 {len(forms_sorted)} 首，例如："
                + "、".join(forms_sorted[:20]) + " 等。"
                + "完整清单与各方的毒性药材投料量见方剂-药材结构化数据。")
        payload = {"entity_type": "herb_formulas", "cn_name": herb_cn,
                   "formula_count": len(forms_sorted)}
        corpus.append({"id": f"herbformulas:{herb_cn}", "entity_type": "herb_formulas",
                       "title": herb_cn, "text": text, "payload": payload})
        rev += 1

    # ---------------- 写出 ----------------
    with open(OUT / "corpus.jsonl", "w", encoding="utf-8") as f:
        for r in corpus:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "entities.json", "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=1)

    stats = Counter(r["entity_type"] for r in corpus)
    lens = sorted(len(r["text"]) for r in corpus)
    print(f"语料总量: {len(corpus)}")
    print("分类型:", dict(stats))
    print(f"药典补充文档: {extra}")
    print(f"文档长度: p50={lens[len(lens)//2]} p90={lens[int(len(lens)*0.9)]} max={lens[-1]}")
    print(f"实体词典: herbs {len(entities['herbs'])} / formulas {len(entities['formulas'])} / "
          f"ingredients {len(entities['ingredients'])}")
    print(f"产物: {OUT/'corpus.jsonl'}  {OUT/'entities.json'}")


if __name__ == "__main__":
    main()
