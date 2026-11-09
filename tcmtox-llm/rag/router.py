# -*- coding: utf-8 -*-
"""
实体路由器：问题 → 实体抽取 + 问题类型分类 + 检索决策（含拒答保护）

设计要点（见 docs/RAG检索层设计.md）：
  - 规则层实现，不调用任何模型——毫秒级，行为完全确定
  - 拒答保护是核心：T6 类问题（实体不在库/关系缺失/医疗建议）不检索，
    让 SFT 模型的训练行为直接生效——检索上下文反而可能诱导编造
  - 结构上不区分"库内/库外药"的模型知识：实体词典查不到 → 不检索，
    把判断权交还给模型（它在 T6a 上已训练过 90%+ 的正确率）
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# 问题类型关键词（按优先级排序，先命中先定类）
# 注意顺序：机制/列表类必须在 grade 之前——"哪些中药属于大毒"是列表查询而非单药分级
QTYPE_RULES = [
    # 标识符类：问 SMILES/CAS/分子式——需要成分实体的精确查表
    ("identifier", r"SMILES|CAS号|PubChem|分子式|分子量|结构式"),
    # 医疗建议类：一律拒答，不检索
    ("advice", r"怎么治|如何治疗|能不能吃|该吃什么|开个方|吃多久|会好吗|剂量.*建议|推荐.*用量.*病"),
    # 机制/关系类：药材的成分/靶点——关系数据缺失，本轮拒答保护（不检索）
    ("mechanism", r"含有哪些(化学)?成分|有哪些成分|作用于哪些|哪些靶点|靶点有哪些|靶点与哪些|与哪些毒性类型|作用机制|是什么成分"),
    # 方剂分析类
    ("formula", r"方剂|处方|这个方|含有哪几味|组成"),
    # 反向列举类（plural 疑问式，先于 grade——"哪些中药属于大毒"是列表查询）
    ("list", r"哪些|有哪些|列出|清单|都有谁"),
    # 分级类
    ("grade", r"毒性分级|有毒吗|毒性强|是几毒|大毒|小毒|是小毒吗|分级属于|毒性数据|有没有毒"),
    # 安全用药类
    ("safety", r"用量|用法|注意事项|孕妇|禁用|慎用|配伍|一起用|同用|禁忌|运动员|儿童"),
]

# 拒答保护场景：这些 (qtype, 条件) 组合不检索，交还模型按 T6 训练行为应答
NO_RETRIEVE_QTYPES = {"advice"}          # 医疗建议永远不检索


@dataclass
class RouteResult:
    question: str
    qtype: str = "other"                      # identifier/advice/mechanism/formula/grade/safety/list/other
    entities: list = field(default_factory=list)   # [(entity_type, name)]
    precise_records: list = field(default_factory=list)  # 精确查表命中的语料记录
    retrieve_semantic: bool = True            # 是否执行语义检索
    reason: str = ""                          # 路由决策说明（调试/日志用）


class EntityRouter:
    """实体路由器：基于实体词典 + 规则的问题路由"""

    def __init__(self, entities_path, corpus_path):
        self.entities = json.load(open(entities_path, encoding="utf-8"))
        # 实体名 → 语料记录索引（精确查表用）
        self.by_name = {}
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                self.by_name.setdefault(r["title"], r)
        # 预编译实体名集合（按长度降序，长名优先命中，避免"甘草"截胡"炙甘草"式前缀）
        self._herb_names = sorted(
            (n for n in self.entities["herbs"] if n), key=len, reverse=True)
        self._formula_names = sorted(
            (n for n in self.entities["formulas"] if n), key=len, reverse=True)
        self._ingredient_names = set(self.entities["ingredients"])
        # 靶点基因名（entities 不含靶点，从语料 by_name 的 target 文档提取）
        self._target_names = {title for title, rec in self.by_name.items()
                              if rec.get("entity_type") == "target"}

    # ---------- 实体抽取 ----------
    def extract_entities(self, question):
        found = []
        seen = set()
        for name in self._herb_names:
            if name in question and name not in seen:
                found.append(("herb", name))
                seen.add(name)
        for name in self._formula_names:
            if name in question and name not in seen:
                found.append(("formula", name))
                seen.add(name)
        # 英文实体（成分名/基因名）做词边界匹配，避免子串误命中
        for m in re.finditer(r"[A-Za-z][A-Za-z0-9\-\s]{2,}", question):
            frag = m.group().strip()
            if frag in self._ingredient_names and frag not in seen:
                found.append(("ingredient", frag))
                seen.add(frag)
            elif frag in self._target_names and frag not in seen:
                found.append(("target", frag))
                seen.add(frag)
        return found

    # ---------- 问题类型 ----------
    def classify(self, question):
        for qtype, pattern in QTYPE_RULES:
            if re.search(pattern, question, re.I):
                return qtype
        return "other"

    # ---------- 主入口 ----------
    def route(self, question):
        """返回 RouteResult。顺序：守卫（拒答保护）在前，查表在后——
        保证被保护的问题不携带任何检索上下文。"""
        q = question.strip()
        qtype = self.classify(q)
        ents = self.extract_entities(q)
        result = RouteResult(question=q, qtype=qtype, entities=ents)

        # 守卫 1：医疗建议类，永不检索（T6d）
        if qtype == "advice":
            result.retrieve_semantic = False
            result.reason = "医疗建议类问题：不检索，按 T6d 训练行为拒答"
            return result

        # 守卫 2：药材的成分/靶点问题，关系数据缺失（T6c）
        if qtype == "mechanism" and any(t == "herb" for t, _ in ents):
            result.retrieve_semantic = False
            result.reason = "药材-成分/靶点关系缺失：不检索，按 T6c 训练行为拒答"
            return result

        # 守卫 3：分级/标识符问题但实体未命中（T6a/T6b）
        if qtype in ("grade", "identifier") and not ents:
            result.retrieve_semantic = False
            result.reason = "实体未命中：不检索（T6a/T6b 保护，防诱导编造）"
            return result

        # 守卫 4：SMILES/CAS 类问题但没有成分实体命中（防诱导编造）
        if qtype == "identifier" and not any(t == "ingredient" for t, _ in ents):
            result.retrieve_semantic = False
            result.reason = "SMILES/CAS 类问题无成分命中：不检索（防诱导编造）"
            return result

        # 查表：实体命中即返回其语料记录
        for etype, name in ents:
            rec = self.by_name.get(name)
            if rec:
                result.precise_records.append(rec)

        # 机制问题但非药材主语（如直接问基因）→ 放行语义检索
        result.reason = (f"实体命中 {len(ents)} 个，精确记录 "
                         f"{len(result.precise_records)} 条，放行语义检索")
        return result


if __name__ == "__main__":
    # 自测：路由器对典型问题的行为抽检
    base = Path(__file__).resolve().parent
    router = EntityRouter(base / "entities.json", base / "corpus.jsonl")
    cases = [
        "川乌的毒性分级是什么？",                      # grade+实体 → 检索
        "麦冬的毒性分级是什么？",                      # grade+实体未命中 → 不检索（T6a）
        "半夏含有哪些化学成分？",                      # mechanism+药材 → 不检索（T6c）
        "CDKN2A靶点与哪些毒性类型相关？",              # mechanism+非药材 → 检索
        "孕妇可以使用川乌吗？",                        # safety+实体 → 检索
        "哪些中药属于大毒？",                          # list → 检索
        "川乌中毒了怎么治？",                          # advice → 不检索
        "gentiopicrin的SMILES是什么？",               # identifier+成分实体 → 检索
        "枸杞子有毒吗？",                              # grade+实体未命中 → 不检索
    ]
    for q in cases:
        r = router.route(q)
        ents = ",".join(f"{t}:{n}" for t, n in r.entities) or "无"
        print(f"[{r.qtype:10s}] 检索={'Y' if r.retrieve_semantic else 'N'} "
              f"精确={len(r.precise_records)} 实体=({ents}) | {q}")
        print(f"             └ {r.reason}")
