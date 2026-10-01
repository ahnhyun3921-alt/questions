"""통계 스냅숏(data/snapshots/*.csv)을 '나답 질문 개선함' 페이지의 db 문서로 바꾼다.

사용법: python analysis/fetch_data.py && python analysis/build_board_stats.py
결과: output/board_stats/<날짜>.json  (문서 하나 = 그날 스냅숏 하나)
       output/board_stats/batch.json   (ArtifactData batch용 writes: stats/<날짜> set)
페이지는 db 컬렉션 `stats`를 읽어 질문별 근거(노출·답변·교체, 평균 대비, 추세)를 보여준다.

문서 모양:
  {date, fetchedAt, totals:{exp, ans, rep, nor}, q:{<DB id>:[노출, 답변, 교체, 미응답, Revision, 전체노출, 전체답변, 전체교체]}}
값은 '현재 Revision' 기준이고, 전체 Revision 누적도 함께 넣는다.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from db_questions import stats_to_db  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data" / "snapshots"
OUT = ROOT / "output" / "board_stats"


def num(x):
    return 0 if pd.isna(x) else int(x)


def build(path: Path) -> dict:
    s = pd.read_csv(path, encoding="utf-8-sig")
    q, tot = {}, {"exp": 0, "ans": 0, "rep": 0, "nor": 0}
    for _, r in s.iterrows():
        exp, ans, rep, nor = (num(r["현재 Revision 노출"]), num(r["현재 Revision 답변"]),
                              num(r["현재 Revision 교체"]), num(r["현재 Revision 미응답"]))
        q[str(stats_to_db(int(r["질문 ID"])))] = [exp, ans, rep, nor, num(r["현재 Revision"]),
                                                   num(r["전체 Revision 노출"]), num(r["전체 Revision 답변"]),
                                                   num(r["전체 Revision 교체"])]
        if r["상태"] == "ACTIVE":
            for k, v in zip(("exp", "ans", "rep", "nor"), (exp, ans, rep, nor)):
                tot[k] += v
    return {"date": path.stem, "fetchedAt": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "totals": tot, "q": q}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    writes = []
    for p in sorted(SNAP.glob("*.csv")):
        doc = build(p)
        f = OUT / f"{p.stem}.json"
        f.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")))
        writes.append({"op": "set", "collection": "stats", "doc_id": p.stem, "file_path": str(f)})
        t = doc["totals"]
        print(f"{p.stem}: 질문 {len(doc['q'])} · 노출 {t['exp']} · 답변율(결론) {t['ans'] / max(1, t['ans'] + t['rep']):.3f} · {f.stat().st_size // 1024}KB")
    (OUT / "batch.json").write_text(json.dumps(writes, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
