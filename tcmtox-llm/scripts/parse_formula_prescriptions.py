# -*- coding: utf-8 -*-
"""
处方解析器：TCMToxDB Formula 处方文本（拼音+炮制前缀+剂量）→ 方剂-药材-剂量结构化表

词典 = TCMTox_herb_clean.csv 的 229 味毒性中药（目的：识别处方中的**毒性药**，
非毒性药不命中是正常现象，不算解析失败）。

匹配级联（顺序很重要，先整串后剥前缀，保护"知母/枳壳/生地黄"这类名字本身含前缀字的药）：
  1. 去剂量 → 去括号备注 → 整串精确匹配
  2. 剥 1~2 个炮制前缀 token 后精确匹配（记录前缀）
  3. 拼音变体模糊匹配（difflib ≥ 0.85，如 dai huang → da huang）
  4. 未命中（非毒性药 / 变体未覆盖）

输出:
  - tcmtox-llm/data/formula_herbs.csv  结构化表（方向D的T3/T5数据源，方向B可直接复用）
  - 解析报告（stdout）
"""
import csv
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "tcmtox_data"
OUT = BASE / "tcmtox-llm" / "data"

# 炮制前缀拼音 → 中文（记录用）
PREFIX_CN = {
    "chao": "炒", "chao jiao": "炒焦", "chao tan": "炒炭", "jiao": "焦", "tan": "炭",
    "zhi": "制", "zhi chao": "制炒", "sheng": "生", "xian": "鲜", "gan": "干",
    "cu": "醋", "cu chao": "醋炒", "cu zhi": "醋制", "jiu": "酒", "jiu chao": "酒炒",
    "jiu zhi": "酒制", "yan": "盐", "yan chao": "盐炒", "yan zhi": "盐制",
    "mi": "蜜", "mi chao": "蜜炒", "mi zhi": "蜜制", "mi jiu": "蜜酒",
    "jiang": "姜", "jiang chao": "姜炒", "jiang zhi": "姜汁制", "dan": "胆",
    "fu": "麸", "fu chao": "麸炒", "sha": "砂", "hua": "滑", "kuai": "块",
    "duan": "煅", "wei": "煨", "bao": "炮",
    "ming": "明", "fa": "法", "shu": "熟", "hei": "黑", "dou": "豆",
    "tu": "土", "chen": "陈",
}
# 注：不做前缀的字：du/ dou? —— "shu 熟" 亦为药名首字可能误剥？整串优先，安全
# 误剥风险示例核对：zhi mu 知母(整串不在毒药表→剥zhi剩mu不命中→保持未命中，正确)
#                  sheng di huang 生地黄(剥后 di huang 非毒药，不命中，正确)

# 已核实的系统性拼音变体（处方表 → 草药表词典键）
# ⚠️ 只收"同一味药的不同罗马字写法"，逐个人工核实；
#    禁止 difflib 自动模糊匹配——229 毒药词典小而密，会把陈皮误配成蟾皮、地黄误配成大黄
ALIAS = {
    "dai huang": "da huang",       # 大黄：处方表通篇写 Dai（151方）
    "chuan qiong": "chuan xiong",  # 川芎：芎 异拼 qiong/xiong（处方表 238 次全为 qiong）
    "dan nan xing": "tian nan xing",  # 胆南星：药典单列品，基源为天南星加胆汁制
}


def load_herb_dict():
    d = {}
    with open(DATA / "TCMTox_herb_clean.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = " ".join(r["Herb_pinyin_name"].lower().split())
            d[key] = {
                "cn": r["Herb_cn_name"], "id": r["HERB_id"],
                "grade": r.get("Toxicity_grade", ""),
            }
    return d


def split_items(prescription):
    """处方文本 → 药材条目列表"""
    return [x.strip() for x in re.split(r"[,;，；\n]", prescription) if x.strip()]


def parse_item(item, herb_dict, keys):
    """单个条目 → dict 或 None(未命中)"""
    raw = item.strip()
    # 1) 剥括号备注（乳香（制）等）
    note = ""
    m = re.search(r"[（(]([^（）()]*)[）)]", raw)
    if m:
        note = m.group(1).strip()
        raw = (raw[:m.start()] + raw[m.end():]).strip()
    # 2) 剥尾部剂量
    dm = re.search(r"([\d.]+)\s*([A-Za-z\u4e00-\u9fa5]+)\s*$", raw)
    dose_val, dose_unit = "", ""
    if dm:
        dose_val, dose_unit = dm.group(1), dm.group(2)
        raw = raw[:dm.start()].strip()
    tokens = raw.lower().split()
    if not tokens:
        return None
    full = " ".join(tokens)

    def make(method, key, prefixes):
        info = herb_dict[key]
        return {
            "herb_cn_name": info["cn"], "HERB_id": info["id"],
            "Toxicity_grade": info["grade"],
            "name_raw": full, "prefix_pinyin": " ".join(prefixes),
            "prefix_cn": PREFIX_CN.get(" ".join(prefixes), " ".join(prefixes)),
            "dosage_value": dose_val, "dosage_unit": dose_unit,
            "note": note, "match_method": method,
        }

    # 3) 整串精确
    if full in herb_dict:
        return make("exact", full, [])
    # 4) 剥前缀（先1后2个token，剩余部分须整串命中）
    for k in (1, 2):
        if len(tokens) > k:
            rest = " ".join(tokens[k:])
            if rest in herb_dict:
                return make("prefix", rest, tokens[:k])
    # 5) 别名
    alias_key = ALIAS.get(full)
    if alias_key and alias_key in herb_dict:
        return make("alias", alias_key, [])
    # 6) 前缀+别名组合（醋大黄等罕见变体）
    for k in (1, 2):
        if len(tokens) > k:
            rest = " ".join(tokens[k:])
            alias_key = ALIAS.get(rest)
            if alias_key and alias_key in herb_dict:
                return make("prefix+alias", alias_key, tokens[:k])
    # 7) 未命中（非毒性药或变体未覆盖，不做模糊匹配以防误标毒性）
    return None


def main():
    herb_dict = load_herb_dict()
    keys = list(herb_dict.keys())

    with open(DATA / "TCMTox_formula.csv", encoding="utf-8-sig") as f:
        formulas = list(csv.DictReader(f))

    rows = []
    stat = Counter()
    unmatched = Counter()
    per_formula = []

    for fm in formulas:
        name, presc = fm["Form_name"].strip(), fm["Prescription"].strip()
        n_items = n_toxic = 0
        if presc:
            for item in split_items(presc):
                n_items += 1
                hit = parse_item(item, herb_dict, keys)
                if hit:
                    stat[hit["match_method"]] += 1
                    n_toxic += 1
                    rows.append({
                        "Form_name": name, "Form_name_en": fm["Form_name_en"],
                        **hit,
                    })
                else:
                    stat["unmatched"] += 1
                    norm = re.sub(r"[（(][^（）()]*[）)]", "", item)
                    norm = re.sub(r"[\d.]+\s*[A-Za-z\u4e00-\u9fa5]+\s*$", "", norm)
                    unmatched[" ".join(norm.lower().split())] += 1
        per_formula.append((name, n_items, n_toxic))

    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["Form_name", "Form_name_en", "name_raw", "prefix_pinyin", "prefix_cn",
              "herb_cn_name", "HERB_id", "Toxicity_grade",
              "dosage_value", "dosage_unit", "note", "match_method"]
    with open(OUT / "formula_herbs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- 报告 ----
    total_items = sum(stat.values())
    print("== 条目级统计 ==")
    for k, v in stat.most_common():
        print("  %-14s %6d (%.1f%%)" % (k, v, 100 * v / total_items))
    toxic_rows = [r for r in rows]
    print("\n== 毒药命中去重药材数:", len({r["herb_cn_name"] for r in toxic_rows}), "==")
    parsed = [(n, a, b) for n, a, b in per_formula if a]
    covered = [(n, a, b) for n, a, b in parsed if b]
    print("有处方方剂 %d，其中检出毒性药 %d 方" % (len(parsed), len(covered)))
    print("检出毒性药条目总数:", len(rows))
    print("\n== 命中药材清单（按出现频次 Top 60）==")
    herb_freq = Counter(r["herb_cn_name"] for r in rows)
    for k, v in herb_freq.most_common(60):
        g = next(r["Toxicity_grade"] for r in rows if r["herb_cn_name"] == k)
        print("  %4d  %-8s [%s]" % (v, k, g or "无分级"))
    print("\n== 未命中 Top 60（多为非毒性药，属正常；重点排查其中实为毒性药的变体）==")
    for k, v in unmatched.most_common(60):
        print("  %4d  %s" % (v, k))


if __name__ == "__main__":
    main()
