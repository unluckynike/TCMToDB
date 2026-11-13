# -*- coding: utf-8 -*-
"""
DPO 偏好对构造器：从"模型在 train 集上的错误输出"挖掘偏好对

⚠️ 数据卫生铁律：输入的 mined jsonl 必须来自模型对 **train 集** 的推理
   （rag_pipeline.py --data data/sft/train.jsonl）。绝不能用 test 集
   评测结果——test 错误喂回训练 = 考题泄漏，评测报废。

四类偏好对（对应对称校准）：
  A 误拒矫正   chosen=gold  rejected=模型的拒答     （主菜，修误拒 4.8%）
  B 误标矫正   chosen=gold  rejected=答错的分级      （修残余误标）
  C T5 强化    chosen=gold  rejected=断言性回答      （巩固保守性）
  D 过度作答抑制 chosen=gold(拒答) rejected=该拒却作答 （对称护栏，防越修越敢答）

输出：LLaMA-Factory DPO 格式 {"instruction", "input", "chosen", "rejected"}
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "tcmtox-llm" / "eval"))
from score import extract_grade                                     # noqa: E402

REFUSAL_MARKERS = ["咨询", "就医", "医师", "药师", "无法提供", "无法替代", "无法回答",
                   "遵医", "请勿", "不能直接", "不可内服", "慎用", "未收录于",
                   "没有「", "无记录", "无毒性分级", "未列入", "无此药", "不属于",
                   "暂无该", "暂未包含", "无法从现有数据", "查无"]
T5_ASSERTIVE = [r"已经超过", r"超出药典", r"已超量", r"超量使用", r"存在超量",
                r"是安全的", r"可以安全", r"不安全", r"安全剂量",
                r"远超", r"远远超过", r"中毒风险", r"危及生命"]
ANSWERABLE_TASKS = ("T1a", "T1b", "T1c", "T1d", "T1e", "T1f",
                    "T2a", "T2b", "T2c", "T2d",
                    "T3a", "T3b", "T3c", "T3d", "T4a", "T4b", "T4c",
                    "T4d", "T4e", "T4f", "T5", "T7a", "T7b", "T7c", "T7d", "T7e", "T7f")
GRADE_TASKS = ("T1a", "T1b", "T6e", "T2c")


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def is_refusal(text):
    return any(m in text for m in REFUSAL_MARKERS)


def is_assertive(text):
    return any(re.search(p, text) for p in T5_ASSERTIVE)


def mine(pairs, cat, r, cap, stats):
    chosen, rejected = r["gold"].strip(), r["response"].strip()
    if (not chosen or not rejected or chosen == rejected
            or len(chosen) < 20 or len(rejected) < 10):
        stats[f"{cat}_skip"] += 1
        return
    if stats[cat] >= cap:
        stats[f"{cat}_cap"] += 1
        return
    pairs.append({"instruction": r["instruction"], "input": "",
                  "chosen": chosen, "rejected": rejected,
                  "meta": {"cat": cat, "task": r["meta"]["task"],
                           "src_file": r.get("model", "")}})
    stats[cat] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mined", nargs="+", help="train 集挖掘的评测 jsonl（可多文件）")
    ap.add_argument("--out", default=str(BASE / "tcmtox-llm" / "data" / "dpo" / "pairs.jsonl"))
    ap.add_argument("--cap-misrefusal", type=int, default=250)
    ap.add_argument("--cap-mislabel", type=int, default=80)
    ap.add_argument("--cap-t5", type=int, default=40)
    ap.add_argument("--cap-overeager", type=int, default=120)
    args = ap.parse_args()

    pairs, stats = [], Counter()
    seen = set()
    caps = {"A_misrefusal": args.cap_misrefusal, "B_mislabel": args.cap_mislabel,
            "C_t5": args.cap_t5, "D_overeager": args.cap_overeager}

    for path in args.mined:
        for r in load_jsonl(path):
            task = r["meta"]["task"]
            q = r["instruction"]
            resp, gold = r["response"].strip(), r["gold"].strip()
            key = (q, resp[:80])
            if key in seen:
                continue
            seen.add(key)

            # A 误拒：本该作答的任务却拒答了
            if task in ANSWERABLE_TASKS and is_refusal(resp) and not is_refusal(gold):
                mine(pairs, "A_misrefusal", r, caps["A_misrefusal"], stats)
                continue
            # B 误标：分级说错（给了具体分级但与 gold 不符）
            if task in GRADE_TASKS:
                pg, pr = extract_grade(gold), extract_grade(resp)
                if pg != "UNKNOWN" and pr in ("大毒", "有毒", "小毒") and pr != pg:
                    mine(pairs, "B_mislabel", r, caps["B_mislabel"], stats)
                    continue
            # C T5 断言违规
            if task == "T5" and is_assertive(resp):
                mine(pairs, "C_t5", r, caps["C_t5"], stats)
                continue
            # D 过度作答：T6d 该拒却给出个体化建议/推荐
            if task == "T6d" and not is_refusal(resp):
                mine(pairs, "D_overeager", r, caps["D_overeager"], stats)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p_ in pairs:
            f.write(json.dumps(p_, ensure_ascii=False) + "\n")

    print("== 偏好对统计 ==")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"总对数: {len(pairs)} → {out}")

    # 抽检 3 对
    print("\n== 抽检 ==")
    for p_ in pairs[:3]:
        print(f"[{p_['meta']['cat']}] Q: {p_['instruction'][:45]}")
        print(f"  chosen: {p_['chosen'][:70]}...")
        print(f"  rejected: {p_['rejected'][:70]}...")


if __name__ == "__main__":
    main()
