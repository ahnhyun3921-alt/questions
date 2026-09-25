"""검토 페이지의 선택 결과(decisions)를 저장소에 기록한다.

사용법:
  1) ArtifactData list (collection "decisions", out_dir "output/decisions_dump")로 내려받기
  2) python analysis/record_decisions.py output/decisions_dump
결과: data/decisions.csv (질문별 최신 상태), 요약 출력
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main(dump_dir: str):
    rows = []
    for f in sorted(Path(dump_dir).rglob("*.json")):
        doc = json.loads(f.read_text())
        body = doc.get("data", doc)
        if "id" not in body:
            continue
        rows.append({k: body.get(k) for k in
                     ["id", "status", "choice", "text", "original", "revAtChoice", "at", "appliedAt"]})
    out = ROOT / "data" / "decisions.csv"
    if not rows:
        print("선택 기록 없음")
        return
    df = pd.DataFrame(rows).sort_values("id")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df["status"].value_counts().to_string())
    # 반영함으로 표시했는데 통계상 Revision이 안 바뀐 질문 = 관리자 도구 반영 누락 의심
    card = pd.read_csv(ROOT / "output" / "question_scorecard.csv").set_index("질문 ID")
    applied = df[df.status == "applied"]
    miss = [int(r.id) for r in applied.itertuples()
            if r.id in card.index and card.loc[r.id, "현재 Revision"] <= (r.revAtChoice or 1)
            and card.loc[r.id, "질문 문구"] != r.text]
    if miss:
        print("반영함 표시했지만 문구가 아직 바뀌지 않은 질문:", miss)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/decisions_dump")
