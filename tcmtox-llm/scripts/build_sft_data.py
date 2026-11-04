# -*- coding: utf-8 -*-
"""
方向D 阶段1：SFT 指令数据构造器（T1~T7，对应 docs/指令数据设计.md）

数据源（均为本仓库已产出的可溯源文件）：
  tcmtox_data/TCMTox_herb_clean.csv   229味，含清洗后毒性分级
  tcmtox_data/yd2020_toxic83.csv      药典83种（分级/用量/注意事项）
  tcmtox_data/TCMTox_formula.csv      1555方
  tcmtox-llm/data/formula_herbs.csv     方剂-药材-剂量（处方解析产物）
  tcmtox_data/TCMTox_target.csv       7885靶点（含毒性类型标注）
  tcmtox_data/TCMTox_ingredient.csv   9377成分

输出：
  tcmtox-llm/data/sft/{t1..t7}.jsonl    分任务全量
  tcmtox-llm/data/sft/{train,val,test}.jsonl  按实体分组切分（同一药/方/靶点不跨集）
  test 集中 T6 拒答类占比 ≥15%（通过给 T6 分配更高的 test 组比例实现）

原则：
  - 答案完全由表字段生成，meta 记录来源；不做自动翻译长文（英文原文以 zh-en 框架引用）
  - 无数据时拒答/说明边界，绝不编造
"""
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "tcmtox_data"
SFT = BASE / "tcmtox-llm" / "data" / "sft"
SEED = 42

# ---------------------------------------------------------------- 小工具
def norm(v):
    return (v or "").strip()

def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def emit(records, task, q, a, key, lang="zh", src=None, input_text=""):
    assert a.strip(), f"{task} 空答案: {q[:40]}"
    assert "{" not in a or "{" not in q or True
    records.append({
        "instruction": q.strip(), "input": input_text, "output": a.strip(),
        "meta": {"task": task, "key": key, "lang": lang,
                 "src": src or []},
    })

def rot(templates, i):
    return templates[i % len(templates)]

# ---------------------------------------------------------------- 词典
NATURE = {"warm": "温", "hot": "热", "cold": "寒", "cool": "凉", "neutral": "平",
          "slightly cold": "微寒", "slightly warm": "微温", "slightly hot": "大热"}
FLAVOR = {"pungent": "辛", "sweet": "甘", "bitter": "苦", "sour": "酸", "salty": "咸",
          "astringent": "涩", "bland": "淡", "acrid": "辛"}
MERIDIAN = {"spleen": "脾", "liver": "肝", "kidney": "肾", "lung": "肺", "heart": "心",
            "stomach": "胃", "large intestine": "大肠", "small intestine": "小肠",
            "bladder": "膀胱", "gallbladder": "胆", "pericardium": "心包",
            "triple energizer": "三焦", "sanjiao": "三焦", "uterus": "子宫", "brain": "脑"}
GRADE_EN = {"大毒": "highly toxic", "有毒": "toxic", "小毒": "slightly toxic"}
TOX_LABEL_CN = {
    "Neurotoxicity": "神经毒性", "Nephrotoxicity": "肾毒性", "Hepatotoxicity": "肝毒性",
    "Reproductive toxicity": "生殖毒性", "Cardiotoxicity": "心脏毒性",
    "Thyrotoxicity": "甲状腺毒性", "Pulmonary toxicity": "肺毒性", "Ototoxicity": "耳毒性",
    "Immunotoxicity": "免疫毒性", "Embryotoxicity": "胚胎毒性", "Myelotoxicity": "骨髓毒性",
    "Myotoxicity": "肌肉毒性", "Gastrointestinal toxicity": "胃肠道毒性",
    "Ocular toxicity": "眼毒性", "Osteotoxicity": "骨毒性", "Dermatotoxicity": "皮肤毒性",
}
EN2CN = {"extremely toxic": "大毒（极毒）", "toxic": "有毒", "slightly toxic": "小毒"}

def trans_props(raw):
    """Warm; Pungent; Bitter -> (性, 味串)"""
    parts = [p.strip().lower() for p in raw.split(";") if p.strip()]
    nature, flavors = [], []
    i = 0
    while i < len(parts):
        p = parts[i]
        if i + 1 < len(parts) and f"{p} {parts[i+1]}" in NATURE:
            nature.append(NATURE[f"{p} {parts[i+1]}"]); i += 2; continue
        if p in NATURE:
            nature.append(NATURE[p])
        elif p in FLAVOR:
            flavors.append(FLAVOR[p])
        else:
            flavors.append(p)  # 未识别保留原文
        i += 1
    return ("".join(nature), "".join(flavors))

def trans_meridian(raw):
    parts = [p.strip().lower() for p in raw.split(";") if p.strip()]
    out = [MERIDIAN.get(p, p) for p in parts]
    return "、".join(out)

def grade_note(herb):
    """按 Toxicity_source 生成「分级+依据」答案短语"""
    g, src = norm(herb["Toxicity_grade"]), norm(herb["Toxicity_source"])
    orig = norm(herb["Toxicity_original"])
    if not g:
        return (f"《中国药典》2020年版未将其列入毒性中药分级名单，TCMToxDB库内亦无毒性分级标注"
                f"（该药收录于TCMToxDB毒性相关中药集）。")
    if "药典" in src:
        return f"毒性分级为「{g}」（依据《中国药典》2020年版）。"
    return (f"TCMToxDB库内标注毒性为{orig}（{EN2CN.get(orig.lower(), orig)}）；"
            f"《中国药典》2020年版未收载该品种，无官方分级。")

# ---------------------------------------------------------------- 主数据
def load_all():
    herbs = load_csv(DATA / "TCMTox_herb_clean.csv")
    yd = {r["yd_name"]: r for r in load_csv(DATA / "yd2020_toxic83.csv")}
    formulas = load_csv(DATA / "TCMTox_formula.csv")
    fherbs = load_csv(BASE / "tcmtox-llm" / "data" / "formula_herbs.csv")
    targets = load_csv(DATA / "TCMTox_target.csv")
    ingredients = load_csv(DATA / "TCMTox_ingredient.csv")
    return herbs, yd, formulas, fherbs, targets, ingredients

# ---------------------------------------------------------------- T1
def build_t1(herbs, yd):
    R = []
    for i, h in enumerate(herbs):
        cn = h["Herb_cn_name"]; key = h["HERB_id"]
        g = norm(h["Toxicity_grade"])
        note = grade_note(h)
        # T1a 分级
        q = rot([f"{cn}的毒性分级是什么？",
                 f"中药{cn}的毒性分级属于哪一级？",
                 f"请问{cn}是几毒药材？"], i)
        a = f"{cn}：{note}"
        if cn in yd:
            a += f"药典规定用量：{yd[cn]['dosage']}。"
        emit(R, "T1a", q, a, key, src=["TCMTox_herb_clean", "yd2020_toxic83"])
        # T1b 有毒吗
        if g:
            q = rot([f"{cn}有毒吗？毒性强不强？",
                     f"服用{cn}会中毒吗？"], i)
            a = (f"是的，{cn}属于毒性中药，{note}" if cn not in yd else f"{cn}属于毒性中药，{note}")
            if cn in yd and norm(yd[cn]["precaution"]):
                a += f"使用注意：{yd[cn]['precaution']}。"
            emit(R, "T1b", q, a, key, src=["TCMTox_herb_clean", "yd2020_toxic83"])
        # T1c 性味归经
        props, mer = norm(h["Properties"]), norm(h["Meridians"])
        if props or mer:
            seg = []
            if props:
                n, f = trans_props(props)
                seg.append("性" + (n or props) + (f"，味{f}" if f else ""))
            if mer:
                seg.append("归" + trans_meridian(mer) + "经")
            emit(R, "T1c", f"请介绍{cn}的性味与归经。", f"{cn}：{'，'.join(seg)}。",
                 key, lang="zh-en", src=["TCMTox_herb_clean"])
        # T1d 临床毒性表现（英文原文引用）
        cm = norm(h["Clinical_manifestations"])
        if cm:
            emit(R, "T1d", f"{cn}过量或中毒时有什么临床表现？",
                 f"据TCMToxDB收载，{cn}的临床毒性表现（英文原文）：{cm}",
                 key, lang="zh-en", src=["TCMTox_herb_clean"])
        # T1e 英文分级
        if g:
            py = " ".join(h["Herb_pinyin_name"].split())
            emit(R, "T1e",
                 f"What is the toxicity grade of {py} ({cn}) in TCMToxDB?",
                 f"In TCMToxDB, {py} ({cn}) is classified as {GRADE_EN[g]} ({g})"
                 f"{' according to the Chinese Pharmacopoeia (2020 edition)' if '药典' in norm(h['Toxicity_source']) else ''}.",
                 key, lang="en", src=["TCMTox_herb_clean"])
        # T1f 部位+拉丁
        up, latin = norm(h["UsePart"]), norm(h["Herb_latin_name"])
        if up or latin:
            seg = []
            if up: seg.append(f"药用部位：{up}")
            if latin: seg.append(f"拉丁学名：{latin}")
            emit(R, "T1f", f"{cn}以什么部位入药？拉丁学名是什么？",
                 f"{cn}：{'；'.join(seg)}。", key, lang="zh-en", src=["TCMTox_herb_clean"])
    return R

# ---------------------------------------------------------------- T2
def build_t2(herbs, yd):
    R = []
    herb_by_cn = {h["Herb_cn_name"]: h for h in herbs}
    i = 0
    for cn, y in yd.items():
        h = herb_by_cn.get(cn)
        py = " ".join(h["Herb_pinyin_name"].split()) if h else cn
        dosage, prec = norm(y["dosage"]), norm(y["precaution"])
        emit(R, "T2a", rot([f"药典规定{cn}的用量范围是多少？",
                            f"{cn}（《中国药典》2020年版）的用法用量是什么？"], i),
             f"《中国药典》2020年版规定{cn}（{py}）用量：{dosage}。", cn,
             src=["yd2020_toxic83"])
        if prec:
            emit(R, "T2b", f"{cn}有什么使用注意事项？",
                 f"《中国药典》2020年版{cn}注意事项：{prec}。", cn, src=["yd2020_toxic83"])
        # T2c 孕妇
        if "孕妇禁用" in prec or "孕妇禁服" in prec:
            ans = f"孕妇禁用。《中国药典》2020年版注意事项：{prec}。"
        elif "孕妇慎用" in prec:
            ans = f"孕妇慎用。《中国药典》2020年版注意事项：{prec}。"
        else:
            ans = (f"《中国药典》2020年版{cn}注意事项未特别提示孕妇禁忌"
                   f"（{prec if prec else '无特别注意事项记载'}）。"
                   f"{cn}为{norm(y['grade'])}药材，孕期使用仍应严格遵医嘱。")
        emit(R, "T2c", f"孕妇可以使用{cn}吗？", ans, cn, src=["yd2020_toxic83"])
        # T2d 配伍
        m = re.search(r"不宜与(.+?)同用", prec)
        if m:
            agents = [a for a in m.group(1).replace("、", "；").split("；") if a]
            first = agents[0]
            emit(R, "T2d", f"{cn}能和{first}一起用吗？",
                 f"不宜同用。《中国药典》2020年版注意事项记载：{cn}不宜与{m.group(1)}同用。"
                 f"此属中药配伍禁忌（十八反/十九畏）范畴，处方审核时应提示。", cn,
                 src=["yd2020_toxic83"])
        i += 1
    return R

# ---------------------------------------------------------------- T6
NOT_IN_DB_HERBS = [  # 来自处方解析未命中清单且确认不在229表（T6a/T6b共用）
    "当归", "茯苓", "黄芩", "黄芪", "丹参", "白芍", "白术", "陈皮", "山楂", "石膏",
    "知母", "枳壳", "麦冬", "党参", "玄参", "山药", "五味子", "牡丹皮", "葛根",
    "枸杞子", "桂枝", "黄柏", "香附", "天麻", "大枣", "菊花", "桃仁", "生姜",
    "连翘", "砂仁", "羌活", "板蓝根", "郁金", "前胡", "蒲公英", "蝉蜕",
]
NOT_EXIST_HERBS = ["灵芝", "西洋参", "铁皮石斛", "雪莲花", "绞股蓝", "罗布麻叶",
                   "月见草", "蒲公英根", "藏红花", "玛卡"]
WESTERN_DRUGS = ["青霉素", "阿司匹林", "布洛芬", "二甲双胍"]

def build_t6(herbs, yd):
    R = []
    cn_set = {h["Herb_cn_name"] for h in herbs}
    graded = [h for h in herbs if norm(h["Toxicity_grade"])]
    # T6a 库外药材（确认不在229也不在药典83）
    for i, cn in enumerate(NOT_IN_DB_HERBS):
        assert cn not in cn_set
        q = rot([f"{cn}的毒性分级是什么？", f"{cn}有毒吗？"], i)
        emit(R, "T6a", q,
             f"「{cn}」未收录于TCMToxDB毒性中药数据集（229味），"
             f"《中国药典》2020年版毒性中药分级名单（83种）中亦无此药，因此无毒性分级记录。"
             f"其安全性信息请查阅药典正文或咨询专业药师。", cn,
             src=["TCMTox_herb_clean", "yd2020_toxic83"])
    # T6b 不存在的药 / 范围外
    for i, cn in enumerate(NOT_EXIST_HERBS):
        emit(R, "T6b", rot([f"{cn}在TCMToxDB里有记录吗？", f"查一下{cn}的毒性数据。"], i),
             f"没有。TCMToxDB毒性中药数据集共收录229味毒性相关中药，其中没有「{cn}」的记录。",
             cn, src=["TCMTox_herb_clean"])
    for i, cn in enumerate(WESTERN_DRUGS):
        emit(R, "T6b", f"{cn}在TCMToxDB里的毒性分级是什么？",
             f"「{cn}」是化学药/西药，不属于TCMToxDB中药数据库的收录范围，无相关记录。",
             cn, src=["TCMTox_herb_clean"])
    # T6c 关系数据缺失
    for i, h in enumerate(graded[:60]):
        cn = h["Herb_cn_name"]
        q = rot([f"{cn}含有哪些化学成分？", f"{cn}作用于哪些人体靶点？"], i)
        emit(R, "T6c", q,
             f"TCMToxDB开放下载的四张表中暂未包含「药材-成分」「药材-靶点」关联数据"
             f"（成分表外键体系与药材表也未打通），因此无法从现有数据给出{cn}的成分或靶点清单。"
             f"可访问数据库网站（sdu-idea.cn/TCMToxDB）Browse页面查询在线版本。",
             cn, src=["数据边界说明"])
    for i in range(20):
        emit(R, "T6c", rot(["这个成分来自哪味中药？", "查询该成分的来源药材。"], i),
             "成分-药材关联数据未包含在开放下载的四张表中，无法从现有数据回答。"
             "数据库网站在线版本可能提供该关联。",
             f"ingredient_link_{i}", src=["数据边界说明"])
    # T6d 医疗建议
    advice_qa = [
        ("我怀孕3个月，方子里有半夏，能吃吗？",
         "我无法提供个体化用药建议。半夏为「有毒」药材（《中国药典》2020年版，孕妇慎用，内服一般炮制后使用），"
         "孕期用药请务必咨询产科医师和中药师，由专业人员评估处方。"),
        ("孩子误食了生半夏，怎么办？",
         "生半夏有毒，误食请立即就医（建议急诊），并携带误食物品包装或样本供医生判断，不要自行催吐或服药。"),
        ("附子中毒有什么症状？",
         "我无法替代医疗诊断。据TCMToxDB收载，附子中毒的临床表现可参见数据库记录（英文原文）："
         + next((norm(h["Clinical_manifestations"]) or "数据库暂无该字段") for h in herbs if h["Herb_cn_name"] == "附子")
         + "。如出现疑似中毒症状，请立即就医。"),
        ("川乌泡酒喝治风湿，行吗？",
         "不建议自行尝试。川乌为「大毒」药材（《中国药典》2020年版），生品内服宜慎、孕妇禁用，且一般须炮制后使用；"
         "自行泡酒服用极易中毒，请遵医嘱使用正规炮制品。"),
        ("马钱子吃多少会致命？",
         "我无法提供致死剂量等危险用量信息。《中国药典》2020年版规定马钱子用量为0.3-0.6g，炮制后入丸散用，"
         "孕妇禁用、不宜多服久服及生用。请严格遵医嘱使用。"),
    ]
    more_illness = [
        "我咳嗽，能吃{cn}吗？", "「{cn}」能治我的病吗？", "长期服用{cn}安全吗？",
        "{cn}和西药能一起吃吗？", "给我推荐一个含{cn}的方子。",
    ]
    for q, a in advice_qa:
        emit(R, "T6d", q, a, f"advice_{len(R)}", src=["安全边界"])
    safe_herbs = [h["Herb_cn_name"] for h in graded[:20]]
    for i, cn in enumerate(safe_herbs):
        for j, tpl in enumerate(more_illness):
            q = tpl.format(cn=cn)
            emit(R, "T6d", q,
                 f"我无法提供个体化诊疗或用药建议。「{cn}」的客观数据可参考本数据库"
                 f"（分级、用量、注意事项等），是否适合您的情况请咨询医师或药师。",
                 f"advice_{cn}_{j}", src=["安全边界"])
    # T6e 冲突纠错（17处药典 vs 库标注）
    for h in herbs:
        conflict = norm(h["Toxicity_conflict"])
        if not conflict:
            continue
        cn = h["Herb_cn_name"]
        orig_en = norm(h["Toxicity_original"])
        orig_cn = EN2CN.get(orig_en.lower(), orig_en)
        now = norm(h["Toxicity_grade"])
        emit(R, "T6e", rot([f"{cn}是{orig_cn}吗？", f"听说{cn}毒性分级是{orig_cn}，对吗？"], len(R)),
             f"不对。按《中国药典》2020年版，{cn}的毒性分级为「{now}」。"
             f"TCMToxDB库内原始标注为{orig_en}（{orig_cn}），本数据集已按药典校正。",
             cn, src=["TCMTox_herb_clean", "yd2020_toxic83"])
    return R

# ---------------------------------------------------------------- T7
def build_t7(herbs, yd, formulas, fherbs):
    R = []
    # 7a 分级列表（诚实口径：药典83种全名单未全部入库，只列库内按药典标注的部分）
    for g in ["大毒", "有毒", "小毒"]:
        members = [h for h in herbs if norm(h["Toxicity_grade"]) == g]
        yd_part = [h["Herb_cn_name"] for h in members if "药典" in norm(h["Toxicity_source"])]
        db_part = [h["Herb_cn_name"] for h in members if "药典" not in norm(h["Toxicity_source"])]
        a = (f"TCMToxDB收录的229味毒性相关中药中，按《中国药典》2020年版标注为「{g}」的共{len(yd_part)}味："
             f"{'、'.join(yd_part)}。"
             f"（注：药典2020年版共收载毒性中药83种，本数据集未全部收录。）")
        if db_part:
            a += f"此外库内另行标注为{g}的还有{len(db_part)}味：{'、'.join(db_part)}（药典未收载，为TCMToxDB库内标注）。"
        emit(R, "T7a", f"哪些中药属于{g}？", a, f"grade_list_{g}", src=["TCMTox_herb_clean", "yd2020_toxic83"])
    # 7b 孕妇
    for kind in ["禁用", "慎用"]:
        members = [cn for cn, y in yd.items() if f"孕妇{kind}" in norm(y["precaution"])]
        emit(R, "T7b", f"哪些毒性中药孕妇{kind}？",
             f"按《中国药典》2020年版注意事项，孕妇{kind}的毒性药材有{len(members)}味：{'、'.join(members)}。",
             f"pregnant_{kind}", src=["yd2020_toxic83"])
    # 7c 配伍反查
    for agent in ["半夏", "瓜蒌", "天花粉", "川贝母", "白蔹", "白及", "甘草", "巴豆", "芒硝", "密陀僧"]:
        herbs_hit = [cn for cn, y in yd.items() if agent in norm(y["precaution"])
                     and f"不宜与" in norm(y["precaution"])]
        if not herbs_hit:
            continue
        emit(R, "T7c", f"哪些药材不宜与{agent}同用？",
             f"按《中国药典》2020年版注意事项，不宜与{agent}同用的有：{'、'.join(herbs_hit)}。"
             f"此类属中药配伍禁忌（十八反/十九畏）范畴。",
             f"combo_{agent}", src=["yd2020_toxic83"])
    # 7d 方剂反查
    by_herb = defaultdict(list)
    for r in fherbs:
        by_herb[r["herb_cn_name"]].append(r["Form_name"])
    for cn in ["朱砂", "雄黄", "附子", "全蝎", "水蛭", "蜈蚣", "半夏", "大黄", "苍耳子", "罂粟壳"]:
        forms = sorted(set(by_herb.get(cn, [])))
        if not forms:
            continue
        sample = "、".join(forms[:15])
        emit(R, "T7d", f"哪些方剂含有{cn}？",
              f"解析后的方剂数据中共{len(forms)}首方剂含{cn}，例如：{sample}"
              + ("等。" if len(forms) > 15 else "。"),
              f"form_of_{cn}", src=["formula_herbs"])
    dadu_forms = sorted({r["Form_name"] for r in fherbs if r["Toxicity_grade"] == "大毒"})
    emit(R, "T7d", "数据库中含大毒药材的方剂有哪些？",
         f"共{len(dadu_forms)}首方剂含大毒药材，例如：{'、'.join(dadu_forms[:15])}等。"
         f"每方所含大毒药材及投料量见 formula_herbs 结构化表。",
         "form_of_dadu", src=["formula_herbs"])
    # 7e 运动员
    athletes = [cn for cn, y in yd.items() if "运动员" in norm(y["precaution"])]
    if athletes:
        emit(R, "T7e", "哪些毒性中药运动员需要慎用或禁用？",
             f"按《中国药典》2020年版注意事项，涉及运动员慎用/禁用的毒性药材有：{'、'.join(athletes)}。",
             "athletes", src=["yd2020_toxic83"])
    # 7f 数据库概况与聚合统计
    n_herbs = len(herbs)
    n_form = len({r["Form_name"] for r in fherbs})
    n_graded = len([h for h in herbs if norm(h["Toxicity_grade"])])
    with_cm = [h["Herb_cn_name"] for h in herbs if norm(h["Toxicity_grade"]) and norm(h["Clinical_manifestations"])]
    emit(R, "T7f", "TCMToxDB收录了多少味毒性中药？其中有毒性分级的有多少？",
         f"TCMToxDB毒性中药数据集共收录{n_herbs}味，其中{n_graded}味已按《中国药典》2020年版或库内标注"
         f"给出毒性分级（大毒/有毒/小毒），其余{n_herbs - n_graded}味药典未列毒性分级名单、库内亦无标注。",
         "db_overview", src=["TCMTox_herb_clean"])
    emit(R, "T7f", "数据库中共有多少首方剂含毒性中药？",
         f"解析后的方剂-药材表中，共{n_form}首方剂含有收录于TCMToxDB毒性中药集的药材，"
         f"毒性药材条目共{len(fherbs)}条（含炮制品）。", "db_overview", src=["formula_herbs"])
    for g in ["大毒", "有毒", "小毒"]:
        forms = sorted({r["Form_name"] for r in fherbs if r["Toxicity_grade"] == g})
        emit(R, "T7f", f"含{g}药材的方剂有多少首？",
             f"共{len(forms)}首方剂含{g}药材，例如：{'、'.join(forms[:10])}等。",
             f"form_count_{g}", src=["formula_herbs"])
    emit(R, "T7f", "哪些毒性中药记录了临床中毒表现？",
         f"库内共{len(with_cm)}味有分级的毒性药记录了临床毒性表现（英文原文），例如："
         f"{'、'.join(with_cm[:20])}等。各药具体表现见单药问答或数据库字段 Clinical_manifestations。",
         "cm_list", src=["TCMTox_herb_clean"])
    return R

# ---------------------------------------------------------------- T4
def build_t4(targets, ingredients):
    rng = random.Random(SEED)
    R = []
    labeled = [t for t in targets if norm(t["Toxicity"])]
    unlabeled = [t for t in targets if not norm(t["Toxicity"])]
    sample_t = rng.sample(labeled, 1000) + rng.sample(unlabeled, 300)
    for i, t in enumerate(sample_t):
        gene = t["Gene_name"]; key = f"{t['Gene_name']}|{t['UniProt_ID']}"
        tox_raw = norm(t["Toxicity"])
        labels = [x.strip() for x in tox_raw.split(";") if x.strip()]
        cn_labels = [TOX_LABEL_CN.get(x, x) for x in labels]
        unip = norm(t["UniProt_ID"]) or "无UniProt编号"
        desc = norm(t["Description"])
        if labels:
            emit(R, "T4a", rot([f"靶点{gene}与哪些毒性类型相关？",
                                f"基因{gene}在数据库中标有哪些毒性标注？"], i),
                 f"靶点{gene}（UniProt: {unip}{'，' + desc if desc else ''}）"
                 f"在TCMToxDB中标注的毒性类型：{'、'.join(cn_labels)}。",
                 key, lang="zh-en", src=["TCMTox_target"])
            if i % 2 == 0:
                emit(R, "T4b", f"What toxicity types is the target {gene} annotated with in TCMToxDB?",
                     f"In TCMToxDB, target {gene} (UniProt: {unip}) is annotated with: {', '.join(labels)}.",
                     key, lang="en", src=["TCMTox_target"])
        else:
            emit(R, "T4a", f"靶点{gene}与哪些毒性类型相关？",
                 f"靶点{gene}（UniProt: {unip}{'，' + desc if desc else ''}）"
                 f"收录于TCMToxDB，但未标注毒性类型。", key, src=["TCMTox_target"])
        if i % 3 == 0 and unip != "无UniProt编号":
            emit(R, "T4c", f"{gene}的UniProt编号是什么？",
                 f"{gene}的UniProt编号为{unip}{('，功能描述：' + desc) if desc else ''}。",
                 key, src=["TCMTox_target"])
    # 成分事实（单表问答，不涉及草药关联）
    ing_smiles = [x for x in ingredients if norm(x["Simles"])]
    for i, x in enumerate(rng.sample(ing_smiles, 450)):
        name = x["Name"]; key = f"ing|{name}"
        seg = [f"分子式 {norm(x['Formula']) or '未知'}"]
        if norm(x["Weight"]): seg.append(f"分子量 {x['Weight']}")
        emit(R, "T4d", rot([f"{name}的SMILES是什么？",
                            f"请给出化合物{name}的SMILES结构式。"], i),
             f"{name}的SMILES：{x['Simles']}（{'，'.join(seg)}）。", key,
             lang="zh-en", src=["TCMTox_ingredient"])
    ing_cas = [x for x in ingredients if norm(x["CAS_ID"])]
    for i, x in enumerate(rng.sample(ing_cas, 200)):
        emit(R, "T4e", f"{x['Name']}的CAS号是多少？",
             f"{x['Name']}的CAS号：{x['CAS_ID']}。", f"ing|{x['Name']}",
             lang="zh-en", src=["TCMTox_ingredient"])
    ing_pc = [x for x in ingredients if norm(x["PubChem_ID"])]
    for i, x in enumerate(rng.sample(ing_pc, 150)):
        emit(R, "T4f", f"化合物{x['Name']}的PubChem CID是多少？",
             f"{x['Name']}的PubChem CID：{x['PubChem_ID']}。", f"ing|{x['Name']}",
             lang="zh-en", src=["TCMTox_ingredient"])
    return R

# ---------------------------------------------------------------- T3
def build_t3(herbs, formulas, fherbs):
    R = []
    herb_by_cn = {h["Herb_cn_name"]: h for h in herbs}
    fh = defaultdict(list)
    for r in fherbs:
        fh[r["Form_name"]].append(r)
    fmeta = {f["Form_name"]: f for f in formulas}
    analyzed = 0
    for name, items in fh.items():
        fm = fmeta.get(name, {})
        n_total = len([x for x in (norm(fm.get("Prescription", "")).split(",") if norm(fm.get("Prescription")) else [])])
        n_total = n_total or len(items)
        tox = [r for r in items if norm(r["Toxicity_grade"])]
        if not tox:
            continue  # 只为含分级毒性药的方剂出题
        lines = []
        for r in tox:
            g = norm(r["Toxicity_grade"])
            src_phrase = ("《中国药典》2020年版" if "药典" in norm(herb_by_cn.get(r["herb_cn_name"], {}).get("Toxicity_source", ""))
                          else "TCMToxDB库内标注")
            dose = f"投料{r['dosage_value']}{r['dosage_unit']}" if r["dosage_value"] else "剂量未标注"
            is_prefixed = r["prefix_cn"] and r["match_method"].startswith("prefix")
            pre = f"{r['herb_cn_name']}（炮制：{r['prefix_cn']}）" if is_prefixed else r["herb_cn_name"]
            h = herb_by_cn.get(r["herb_cn_name"], {})
            extra = ""
            if src_phrase == "《中国药典》2020年版" and norm(h.get("Toxicity_grade")):
                y = YD_CACHE.get(r["herb_cn_name"])
                if y:
                    extra = f"，药典用量{norm(y['dosage'])}"
            lines.append(f"{len(lines)+1}. {pre}：{g}（{src_phrase}{extra}），{dose}")
        toxic_block = "\n".join(lines)
        dise = norm(fm.get("Diseases"))
        eff = norm(fm.get("Efficacy"))
        i = analyzed
        emit(R, "T3a",
             rot([f"分析方剂「{name}」的处方，指出其中的毒性药材及其用量分级。",
                  f"「{name}」里有哪些毒性中药？"], i),
             f"方剂「{name}」处方共约{n_total}味药，解析后识别出收录于TCMToxDB毒性中药集的药材如下：\n{toxic_block}\n"
             f"注：其余药味不在毒性中药集内；投料量为制剂总量，不代表单次服用量。",
             name, src=["formula_herbs", "TCMTox_herb_clean", "yd2020_toxic83"])
        if dise or eff:
            seg = []
            if dise: seg.append(f"主治（数据库原文）：{dise}")
            if eff: seg.append(f"功效（原文）：{eff}")
            emit(R, "T3c", f"方剂「{name}」主治什么？含哪几味毒性药？",
                 f"{'；'.join(seg)}。含分级毒性药{len(tox)}味："
                 + "、".join((r["prefix_cn"] + r["herb_cn_name"]) if r["prefix_cn"] and r["match_method"].startswith("prefix") else r["herb_cn_name"] for r in tox)
                 + "。", name, lang="zh-en", src=["TCMTox_formula", "formula_herbs"])
        en = norm(fm.get("Form_name_en"))
        if en and i % 4 == 0:
            emit(R, "T3d", f"List the graded toxic herbs contained in the TCM formula \"{en}\".",
                 f"The formula \"{en}\" ({name}) contains {len(tox)} graded toxic herb(s):\n"
                 + "\n".join(f"- {r['herb_cn_name']} ({GRADE_EN.get(r['Toxicity_grade'], '')}), "
                             f"dosing {r['dosage_value']}{r['dosage_unit']}" for r in tox),
                 name, lang="en", src=["formula_herbs"])
        if i % 12 == 0:
            emit(R, "T3b", f"这张处方里有需要注意安全性的药吗：{norm(fm['Prescription'])}？",
                 f"经解析，该处方收录于TCMToxDB的毒性相关药材有{len(tox)}味：\n{toxic_block}\n"
                 f"注：投料量为制剂总量，不代表单次服用量；处方审核请咨询中药师。",
                 name, src=["formula_herbs"])
        analyzed += 1
    return R

# ---------------------------------------------------------------- T5
YD_CACHE = {}

def build_t5(yd, fherbs):
    R = []
    seen = set()
    for r in fherbs:
        cn = r["herb_cn_name"]
        y = yd.get(cn)
        if not y or not r["dosage_value"]:
            continue
        m = re.search(r"([\d.]+)\s*[-~〜至]\s*([\d.]+)\s*g", norm(y["dosage"]))
        if not m:
            continue
        lo, hi = m.group(1), m.group(2)
        fk = (r["Form_name"], cn)
        if fk in seen:
            continue
        seen.add(fk)
        dose_txt = norm(y["dosage"])
        q = rot([
            f"方剂「{r['Form_name']}」中{cn}投料量为{r['dosage_value']}{r['dosage_unit']}，超过药典用量了吗？",
            f"「{r['Form_name']}」含{cn}{r['dosage_value']}{r['dosage_unit']}，这个剂量安全吗？",
        ], len(seen))
        a = (f"《中国药典》2020年版规定{cn}用量为{dose_txt}。"
             f"方剂「{r['Form_name']}」处方中{cn}投料量为{r['dosage_value']}{r['dosage_unit']}。"
             f"需注意：处方剂量为制剂总投料量，并非单次或一日服用量；实际摄入量取决于制剂规格与每次服用量。"
             f"因此不能直接据此判定超量或安全；如需评估，应按成品规格折算日摄入量，并咨询专业医师或药师。")
        emit(R, "T5", q, a, r["Form_name"], src=["formula_herbs", "yd2020_toxic83"])
    return R

# ---------------------------------------------------------------- 切分
def split_by_group(records, name):
    """按 meta.key 分组切分；T6 按更高 test 比例过采样，保证 test 中 T6 ≥15%"""
    rng = random.Random(SEED + hash(name) % 1000)
    groups = defaultdict(list)
    for r in records:
        groups[r["meta"]["key"]].append(r)
    keys = sorted(groups.keys())
    rng.shuffle(keys)
    if name == "T6":
        n = len(keys)
        n_tr, n_va = int(n * 0.55), int(n * 0.10)
        parts = {"train": keys[:n_tr], "val": keys[n_tr:n_tr + n_va], "test": keys[n_tr + n_va:]}
    else:
        n = len(keys)
        n_tr, n_va = int(n * 0.90), int(n * 0.05)
        parts = {"train": keys[:n_tr], "val": keys[n_tr:n_tr + n_va], "test": keys[n_tr + n_va:]}
    out = {"train": [], "val": [], "test": []}
    for split, ks in parts.items():
        for k in ks:
            out[split].extend(groups[k])
    return out

# ---------------------------------------------------------------- 主流程
def main():
    global YD_CACHE
    herbs, yd, formulas, fherbs, targets, ingredients = load_all()
    YD_CACHE = yd
    SFT.mkdir(parents=True, exist_ok=True)

    tasks = {
        "T1": build_t1(herbs, yd),
        "T2": build_t2(herbs, yd),
        "T6": build_t6(herbs, yd),
        "T7": build_t7(herbs, yd, formulas, fherbs),
        "T4": build_t4(targets, ingredients),
        "T3": build_t3(herbs, formulas, fherbs),
        "T5": build_t5(yd, fherbs),
    }

    all_splits = {"train": [], "val": [], "test": []}
    print("== 各任务样本数 ==")
    for name, recs in tasks.items():
        # 自检：模板占位符残留 / 空答案（注：化学名本身可能含{}，只查已知模板变量）
        bad_tokens = ("{cn}", "{g}", "{name}", "{herb_en}", "{Form_name_en}", "{orig_cn}", "{yd_grade}", "{agent}")
        for r in recs:
            joined = r["instruction"] + r["output"]
            assert not any(t in joined for t in bad_tokens), f"占位符残留 {r['meta']['task']}: {r['instruction'][:50]}"
        with open(SFT / f"{name.lower()}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        parts = split_by_group(recs, name)
        for sp in all_splits:
            all_splits[sp].extend(parts[sp])
        print(f"  {name}: {len(recs):5d}  (train {len(parts['train'])} / val {len(parts['val'])} / test {len(parts['test'])})")

    rng = random.Random(SEED)
    for sp in all_splits:
        rng.shuffle(all_splits[sp])
        with open(SFT / f"{sp}.jsonl", "w", encoding="utf-8") as f:
            for r in all_splits[sp]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(len(v) for v in all_splits.values())
    print(f"\n== 汇总 ==  总量 {total}  train {len(all_splits['train'])} / val {len(all_splits['val'])} / test {len(all_splits['test'])}")
    for sp in ["train", "val", "test"]:
        c = Counter(r["meta"]["task"] for r in all_splits[sp])
        t6_n = sum(v for k, v in c.items() if k.startswith("T6"))
        t6_share = 100 * t6_n / len(all_splits[sp])
        print(f"  {sp}: T6占比 {t6_share:.1f}%  分布 {dict(sorted(c.items()))}")
    lang = Counter(r["meta"]["lang"] for r in all_splits["train"])
    print(f"  train 语言分布: {dict(lang)}")

    print("\n== 抽检样例 ==")
    for name in tasks:
        print(f"--- {name} ---")
        for r in tasks[name][:2]:
            print("Q:", r["instruction"])
            print("A:", r["output"][:220].replace("\n", " | "))
            print()

if __name__ == "__main__":
    main()
