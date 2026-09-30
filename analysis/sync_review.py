"""검토 페이지(아티팩트 DB)에 올릴 문서를 만든다.

사용법: python analysis/sync_review.py
결과:   output/review_sync/qchunks/c0.json ... (질문 50개씩), output/review_sync/meta/run.json
        output/review_sync/batch.json  ← ArtifactData batch의 writes 배열 (그대로 전달)

DB 구조
  qchunks/c{n}      시스템이 쓰는 질문 데이터 (이 스크립트가 매번 덮어씀)
  meta/run          마지막 갱신 정보
  decisions/q{id}   사용자가 페이지에서 쓰는 선택·상태 (이 스크립트는 건드리지 않음)
"""
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lint_questions import lint  # noqa: E402
from db_questions import INTEREST_ID  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
SYNC = OUT / "review_sync"
CHUNK = 50


def num(x, nd=4):
    return None if pd.isna(x) else round(float(x), nd)


# 신규 후보 세트: (파일, 분류 라벨). 검토 화면에서 '신규 후보'(N)·'신규 연애'(L) 필터로 본다
NEW_SETS = [("new_questions_love.csv", "L. 신규 연애"),
            ("new_questions_self.csv", "N. 신규(자아분석)"),
            ("new_questions_fun.csv", "N. 신규(재미 보강)"),
            ("new_questions_scraped.csv", "N. 신규(외부 디깅)"),
            ("new_questions.csv", "N. 신규(내부 보강)")]


def new_rows():
    out = []
    gp = Path(__file__).parent / "new" / "guides_new.csv"
    guides = pd.read_csv(gp).set_index("신규 ID") if gp.exists() else None
    gcols = ("empathy_guide", "hint_guide", "leading_question_guide")
    for fn, cls in NEW_SETS:
        p = OUT / fn
        if not p.exists():
            continue
        nd = pd.read_csv(p).fillna("")
        nd = nd.sort_values("예상 답변율(결론 기준)", ascending=False)
        for _, r in nd.iterrows():
            src_text = r.get("원문") or r.get("참고 원문") or ""
            out.append({
                "kind": "new", "id": str(r["신규 ID"]), "q": r["문구"], "cat": r["관심사"], "level": int(r["레벨"]),
                "axisName": r.get("분석 축", "") or "",
                "axis": (f"{r['분석 축']} · {r['해석 가이드']}" if r.get("분석 축") else ""),
                "cls": cls, "rev": 0, "exp": 0, "ans": 0, "rep": 0, "prevExp": 0, "prevAns": 0,
                "est": num(r["예상 답변율(결론 기준)"]), "vsBase": num(r["기존 평균 대비(%p)"], 1),
                "form": r.get("형식", ""),
                "diag": (f"분석 축: {r['분석 축']} ({r['근거 개념']}) · 해석: {r['해석 가이드']} · 공감 포인트: {r['인기·검증 근거']}"
                         if r.get("분석 축") else f"{r['출처']}" + (f" · 원문: {src_text}" if src_text else "")),
                "principle": "" if r.get("분석 축") else r.get("인기·검증 근거", ""), "src": r.get("출처 URL", ""),
                "similar": f"{r['가장 비슷한 기존 질문']} (유사도 {r['글자 유사도']})",
                "lint": [m for _, m in lint(r["문구"])],
                "o1": r["문구"], "o2": "", "action": "신규 추가", "dPred": None, "needsRewrite": False,
                "rank": None, "gain": None, "post": None, "pBelow": None, "risk": "",
                "guides": ([str(guides.loc[r["신규 ID"], c]) for c in gcols]
                           if guides is not None and r["신규 ID"] in guides.index else []),
            })
    return out


def level1_block_rows():
    """DB id 661~684(레벨 1 첫 질문 묶음) 재작성안. id는 'D661'처럼 붙여 기존 통계 ID와 겹치지 않게 한다."""
    p = Path(__file__).parent / "new" / "level1_block.csv"
    if not p.exists():
        return []
    cat = {v: k for k, v in INTEREST_ID.items()}
    out = []
    for _, r in pd.read_csv(p).fillna("").iterrows():
        out.append({
            "kind": "new", "id": f"D{int(r['DB id'])}", "q": r["기존 문구"], "cat": cat[int(r["interest_id"])], "level": 1,
            "cls": "1. 레벨1 첫 질문(DB 661~684)", "rev": 0, "exp": 0, "ans": 0, "rep": 0, "prevExp": 0, "prevAns": 0,
            "est": None, "vsBase": None, "form": "",
            "diag": f"DB id {int(r['DB id'])} · 지금 문구: {r['기존 문구']} · {r['판정']}", "principle": "", "src": "",
            "similar": "", "lint": [m for _, m in lint(r["권장 문구"])],
            "o1": r["권장 문구"], "o2": "", "action": "레벨 1 재작성", "dPred": None, "needsRewrite": False,
            "rank": None, "gain": None, "post": None, "pBelow": None, "risk": "",
            "guides": [r["empathy_guide"], r["hint_guide"], r["leading_question_guide"]],
        })
    return out


def axis_ref_rows(axis, card, have):
    """축이 붙었지만 검토 목록에 없는(문구 유지) 기존 질문. 축 필터로 볼 때만 나오는 보기 전용 카드."""
    if axis is None:
        return []
    fin = pd.read_csv(OUT / "final_questions.csv").set_index("질문 ID")
    sc = card.set_index("질문 ID")
    out = []
    for i, t in axis.iterrows():
        if int(i) in have or i not in fin.index:
            continue
        out.append({
            "kind": "ref", "id": int(i), "q": fin.loc[i, "최종 권장 문구"], "cat": fin.loc[i, "관심사"], "level": int(sc.loc[i, "레벨"]),
            "cls": "V. 축 참고(문구 유지)", "rev": 0, "exp": int(sc.loc[i, "노출"]), "ans": int(sc.loc[i, "답변"]), "rep": int(sc.loc[i, "교체"]),
            "prevExp": 0, "prevAns": 0, "est": None, "post": None, "lint": [], "diag": "", "principle": "", "o1": "", "o2": "",
            "rank": None, "gain": None, "needsRewrite": False, "axisName": t["분석 축"], "axis": f"{t['분석 축']} · {t['해석 가이드']}",
        })
    return out


def category_rows(cf, card, axis):
    """기존 질문 관심사 변경 제안. id는 'K{통계 ID}'로 따로 둔다(문구 결정과 섞이지 않게)."""
    if not len(cf):
        return []
    fin = pd.read_csv(OUT / "final_questions.csv").set_index("질문 ID")
    sc = card.set_index("질문 ID")
    out = []
    for _, r in cf.iterrows():
        i = int(r["통계 질문 ID"])
        out.append({
            "kind": "cat", "id": f"K{i}", "q": fin.loc[i, "최종 권장 문구"], "cat": r["제안 관심사"], "catFrom": r["지금 관심사"],
            "level": int(sc.loc[i, "레벨"]), "cls": "K. 관심사 변경", "rev": 0,
            "exp": int(sc.loc[i, "노출"]), "ans": int(sc.loc[i, "답변"]), "rep": int(sc.loc[i, "교체"]), "prevExp": 0, "prevAns": 0,
            "est": None, "vsBase": None, "form": "",
            "diag": f"통계 #{i} · 관심사 {r['지금 관심사']} → {r['제안 관심사']} · 이유: {r['이유']}", "principle": "", "src": "",
            "similar": "", "lint": [], "o1": fin.loc[i, "최종 권장 문구"], "o2": "", "action": "관심사 변경",
            "dPred": None, "needsRewrite": False, "rank": None, "gain": None, "post": None, "pBelow": None, "risk": "",
            "guides": [],
            "axisName": axis.loc[i, "분석 축"] if axis is not None and i in axis.index else "",
            "axis": (f"{axis.loc[i, '분석 축']} · {axis.loc[i, '해석 가이드']}" if axis is not None and i in axis.index else ""),
        })
    return out


def main():
    card = pd.read_csv(OUT / "question_scorecard.csv")
    rw = pd.read_csv(Path(__file__).parent / "rewrites.csv").fillna("")
    scored = pd.read_csv(OUT / "rewrites_scored.csv").fillna("") if (OUT / "rewrites_scored.csv").exists() else None
    # 스코어카드에 붙어 있는 옛 수정안 열은 버리고 항상 최신 rewrites.csv를 쓴다
    card = card.drop(columns=[c for c in rw.columns if c != "질문 ID" and c in card.columns])
    d = card.merge(rw, on="질문 ID", how="left")
    if scored is not None:
        d = d.merge(scored[["질문 ID", "조치 유형", "예측 변화(%p)"]], on="질문 ID", how="left")
    txt = ["원인 진단", "수정 원칙", "수정안 1 (권장)", "수정안 2 (대안)", "조치 유형", "위험 특성"]
    d[[c for c in txt if c in d]] = d[[c for c in txt if c in d]].fillna("")
    pri_path = OUT / "v2" / "priority.csv"
    if pri_path.exists():
        pri = pd.read_csv(pri_path)[["id", "우선순위", "기대 추가 답변(월)", "사후 답변율"]].rename(columns={"id": "질문 ID"})
        d = d.merge(pri, on="질문 ID", how="left")
    else:
        d["우선순위"] = d["기대 추가 답변(월)"] = d["사후 답변율"] = np.nan
    cls = d["분류"].str[0]
    has_rw = d["수정안 1 (권장)"].fillna("") != ""
    # 검토 대상: A·B, 수정 후 관찰(F), 이미 수정안이 있는 질문
    sel = d[cls.isin(["A", "B", "F"]) | has_rw].copy()

    tp = Path(__file__).parent / "axis_tags.csv"
    axis = pd.read_csv(tp).set_index("질문 ID") if tp.exists() else None
    cf_path = Path(__file__).parent / "category_fixes_existing.csv"
    cf = pd.read_csv(cf_path) if cf_path.exists() else pd.DataFrame(columns=["통계 질문 ID"])
    cat_change = {int(r["통계 질문 ID"]): f"{r['지금 관심사']} → {r['제안 관심사']} ({r['이유']})" for _, r in cf.iterrows()}
    rows = []
    for _, r in sel.iterrows():
        o1, o2 = r.get("수정안 1 (권장)") or "", r.get("수정안 2 (대안)") or ""
        # '(유지)', '(저녁 노출)'처럼 원문을 그대로 두는 안은 비교하지 않는다
        if r["분류"][0] in "AB" and r["질문 문구"] in [t for t in (o1, o2) if t and not t.startswith("(")]:
            # 이미 수정안을 반영했는데도 여전히 나쁨 → 새 수정안 필요
            o1 = o2 = ""
        rows.append({
            "id": int(r["질문 ID"]), "q": r["질문 문구"], "cat": r["관심사"], "level": int(r["레벨"]),
            "cls": r["분류"], "rev": int(r["현재 Revision"]),
            "exp": int(r["노출"]), "ans": int(r["답변"]), "rep": int(r["교체"]),
            "prevExp": int(r["이전 Revision 노출"]), "prevAns": int(r["이전 Revision 답변"]),
            "est": num(r["추정 답변율(축소)"]), "pBelow": num(r["P(평균 미만)"]),
            "risk": r["위험 특성"] if isinstance(r["위험 특성"], str) else "",
            "lint": [m for _, m in lint(r["질문 문구"])],
            "diag": r.get("원인 진단") or "", "principle": r.get("수정 원칙") or "",
            "o1": o1, "o2": o2,
            "action": r.get("조치 유형") if isinstance(r.get("조치 유형"), str) else "",
            "dPred": num(r.get("예측 변화(%p)"), 2) if isinstance(r.get("예측 변화(%p)"), (int, float)) else None,
            "needsRewrite": (not o1) and r["분류"][0] in "AB",
            "rank": None if pd.isna(r["우선순위"]) else int(r["우선순위"]),
            "gain": num(r["기대 추가 답변(월)"], 2), "post": num(r["사후 답변율"]),
            "axisName": axis.loc[r["질문 ID"], "분석 축"] if axis is not None and r["질문 ID"] in axis.index else "",
            "catChange": cat_change.get(int(r["질문 ID"]), ""),
            "axis": (f"{axis.loc[r['질문 ID'], '분석 축']} · {axis.loc[r['질문 ID'], '해석 가이드']}"
                     if axis is not None and r["질문 ID"] in axis.index else ""),
        })
    order = {"A": 0, "B": 1, "F": 2}
    # 기대 이득(v2 우선순위) 순. 우선순위가 없는 항목(유지·오류 수정 등)은 분류 순으로 뒤에
    rows.sort(key=lambda x: (x["rank"] is None, x["rank"] or 0, order.get(x["cls"][0], 3), x["est"] if x["est"] is not None else 1))
    rows += axis_ref_rows(axis, card, {r["id"] for r in rows})
    rows += level1_block_rows() + category_rows(cf, card, axis) + new_rows()

    n = math.ceil(len(rows) / CHUNK)
    (SYNC / "qchunks").mkdir(parents=True, exist_ok=True)
    (SYNC / "meta").mkdir(parents=True, exist_ok=True)
    writes = []
    for i in range(n):
        p = SYNC / "qchunks" / f"c{i}.json"
        p.write_text(json.dumps({"items": rows[i * CHUNK:(i + 1) * CHUNK]}, ensure_ascii=False, allow_nan=False))
        writes.append({"op": "set", "collection": "qchunks", "doc_id": f"c{i}", "file_path": str(p)})
    summ = json.loads((OUT / "summary.json").read_text())
    meta = {
        "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunks": n, "count": len(rows),
        "baseRate": round(summ["base_answer_rate"], 4), "exposures": summ["totals"]["exp"],
        "classCounts": summ["class_counts"],
        "needsRewrite": sum(r["needsRewrite"] for r in rows),
    }
    (SYNC / "meta" / "run.json").write_text(json.dumps(meta, ensure_ascii=False))
    writes.append({"op": "set", "collection": "meta", "doc_id": "run", "file_path": str(SYNC / "meta" / "run.json")})
    (SYNC / "batch.json").write_text(json.dumps(writes, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False))
    for r in rows:
        if r["needsRewrite"]:
            print("needs rewrite:", r["id"], r["cls"], r["q"])


if __name__ == "__main__":
    main()
