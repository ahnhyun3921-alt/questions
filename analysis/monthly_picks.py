"""매달 '문제 있는 질문'을 골라 개선함 페이지의 '골라 주세요'에 올릴 후보를 만든다.

1) 후보 뽑기:  python analysis/monthly_picks.py candidates --month 2026-11 [--keeps keeps.json]
   → output/monthly_picks/<month>.json  ({month, avg, items:[{key, text, form, cat, stats, reason, opts:[]}]})
   기준: 결론(답변+교체) 5회 이상이고, 평가 모델 사후 확률 P(진짜 답변율 < 평균 − 10%p) ≥ 0.8
         (board/data/insights.json — 같은 스냅숏일 때. 없으면 예전 K=10 추정 규칙).
   제외: 개선본 문구가 지금 서비스 문구와 달라 이미 수정이 기다리는 질문, 비활성 질문,
         원래 DB의 레벨 1(첫 질문) 기존 질문(원본 그대로 두기로 함),
         keeps.json(개선함 db `keeps` 컬렉션을 받은 것)에 최근 90일 안에 '그대로 두기'가 있는 질문.
2) 각 item의 opts에 수정안 2~3개를 채운다(사람 또는 루틴 세션이 직접 작성):
   {text, empathy, hint, leading, why} — ROUTINE.md 4단계 수정 원칙과 lint_questions.py 기준을 따른다.
3) 올리기 파일: python analysis/monthly_picks.py batch --month 2026-11
   → output/monthly_picks/<month>_batch.json (ArtifactData batch: picks/<month>-<key> set)
"""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from db_questions import original_first_ids  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data" / "snapshots"
OUT = ROOT / "output" / "monthly_picks"
BOARD = ROOT / "board" / "data" / "questions.json"
K, MIN_EXP, GAP = 10, 5, 0.10


def candidates(month, keeps_path):
    snap = sorted(SNAP.glob("*.csv"))[-1]
    st = json.loads((ROOT / "output" / "board_stats" / f"{snap.stem}.json").read_text())
    t = st["totals"]
    avg = t["ans"] / max(1, t["ans"] + t["rep"])
    s = pd.read_csv(snap, encoding="utf-8-sig")
    live_text = {str(k): v for k, v in zip(s["질문 ID"].map(lambda i: i if i < 661 else i + 24), s["질문 문구"])}
    rows = {r["key"]: r for r in json.loads(BOARD.read_text())["rows"]}
    kept = set()
    if keeps_path:
        cut = datetime.now(timezone.utc) - timedelta(days=90)
        for d in json.loads(Path(keeps_path).read_text()):
            data = d.get("data", d)
            at = data.get("at", "")
            if at and datetime.fromisoformat(at.replace("Z", "+00:00")) >= cut:
                kept.add(str(d.get("doc_id") or d.get("id") or data.get("key")))
    keep1 = {str(i) for i in original_first_ids()}   # 원래 첫 질문은 원본 그대로
    # 평가 모델(build_insights.py) 사후: P(진짜 답변율 < 평균 − 10%p). 같은 스냅숏일 때만 쓴다.
    ins_path = ROOT / "board" / "data" / "insights.json"
    ins = json.loads(ins_path.read_text()) if ins_path.exists() else {}
    post = ins.get("model", {}).get("post", {}) if ins.get("snapshot") == snap.stem else {}
    fix_p = ins.get("model", {}).get("policy", {}).get("fix_p", 0.8)
    items = []
    for k, v in st["q"].items():
        exp, ans, rep = v[0], v[1], v[2]
        r = rows.get(k)
        if not r or r["deleted"] or k in kept or k in keep1 or ans + rep < MIN_EXP:
            continue
        if r["text"].strip() != str(live_text.get(k, "")).strip():
            continue  # 이미 수정안이 개선본에 있고 반영을 기다림
        if post:
            if k not in post or post[k][1] < fix_p:
                continue
            p_low = post[k][1]
        else:   # 모델이 없으면 예전 규칙(평균 쪽으로 K=10 당긴 추정)
            if (ans + K * avg) / (ans + rep + K) >= avg - GAP:
                continue
            p_low = None
        rate = ans / (ans + rep) if ans + rep else 0
        items.append({"key": k, "text": r["text"], "form": r["form"], "cat": r["cat"], "level": r["level"],
                      "stats": {"date": snap.stem, "exp": exp, "ans": ans, "rep": rep, "rate": round(rate, 3), "avg": round(avg, 3)},
                      "reason": f"결론 {ans + rep}회 · 답변율 {rate:.0%} (평균 {avg:.0%}) · 교체 {rep}회"
                                + (f" · 평균보다 10%p 넘게 낮을 확률 {p_low:.0%}" if p_low is not None else ""),
                      "g": r["g"], "opts": []})
    items.sort(key=lambda x: x["stats"]["rate"])
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{month}.json"
    f.write_text(json.dumps({"month": month, "avg": round(avg, 3), "snapshot": snap.stem, "items": items}, ensure_ascii=False, indent=1))
    print(f"{month}: 후보 {len(items)}개 (평균 답변율 {avg:.0%}, 스냅숏 {snap.stem}) → {f}")


def batch(month):
    d = json.loads((OUT / f"{month}.json").read_text())
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    writes, bad = [], []
    for it in d["items"]:
        opts = [o for o in it["opts"] if all(str(o.get(x, "")).strip() for x in ("text", "empathy", "hint", "leading"))]
        if len(opts) < 2:
            bad.append(it["key"])
            continue
        writes.append({"op": "set", "collection": "picks", "doc_id": f"{month}-{it['key']}",
                       "data": {"key": it["key"], "month": month, "at": now, "reason": it["reason"], "stats": it["stats"],
                                "opts": [{k: o.get(k, "") for k in ("text", "empathy", "hint", "leading", "why")} for o in opts]}})
    f = OUT / f"{month}_batch.json"
    f.write_text(json.dumps(writes, ensure_ascii=False, indent=1))
    print(f"올릴 {len(writes)}개 → {f}" + (f" · 수정안 2개 미만이라 뺀 질문: {', '.join(bad)}" if bad else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["candidates", "batch"])
    ap.add_argument("--month", default=datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m"))
    ap.add_argument("--keeps")
    a = ap.parse_args()
    candidates(a.month, a.keeps) if a.cmd == "candidates" else batch(a.month)
