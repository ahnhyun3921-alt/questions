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

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lint_questions import lint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
SYNC = OUT / "review_sync"
CHUNK = 50


def num(x, nd=4):
    return None if pd.isna(x) else round(float(x), nd)


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
    cls = d["분류"].str[0]
    has_rw = d["수정안 1 (권장)"].fillna("") != ""
    # 검토 대상: A·B, 수정 후 관찰(F), 이미 수정안이 있는 질문
    sel = d[cls.isin(["A", "B", "F"]) | has_rw].copy()

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
        })
    order = {"A": 0, "B": 1, "F": 2}
    rows.sort(key=lambda x: (order.get(x["cls"][0], 3), x["est"] if x["est"] is not None else 1))

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
