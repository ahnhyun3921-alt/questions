"""통계 사이트에서 질문 통계 CSV를 받아 저장한다.

사용법: python analysis/fetch_data.py
환경 변수(선택):
  NADAB_STATS_BASE    기본값 https://r9k2m7q3.shop
  NADAB_STATS_COOKIE  사이트에 로그인이 생기면 쿠키 문자열
  NADAB_STATS_TOKEN   사이트에 토큰 인증이 생기면 Bearer 토큰
결과: data/snapshots/YYYY-MM-DD.csv, nadab_daily_question_stats.csv(최신본)
"""
import io
import os
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = os.environ.get("NADAB_STATS_BASE", "https://r9k2m7q3.shop").rstrip("/")
QUERY = ("keyword=&interestCode=&questionLevel=&active=&minimumCurrentExposureCount=0"
         "&sort=CURRENT_REROLLED_COUNT&direction=DESC")
EXPECTED = ["질문 ID", "질문 문구", "관심사 코드", "관심사", "레벨", "상태", "현재 Revision",
            "현재 Revision 적용 시각", "현재 Revision 노출", "현재 Revision 답변", "현재 Revision 답변율(%)",
            "현재 Revision 교체", "현재 Revision 교체율(%)", "현재 Revision 미응답", "전체 Revision 노출",
            "전체 Revision 답변", "전체 Revision 답변율(%)", "전체 Revision 교체", "전체 Revision 미응답"]


def fetch() -> bytes:
    req = urllib.request.Request(f"{BASE}/stats/question/overview.csv?{QUERY}")
    if os.environ.get("NADAB_STATS_COOKIE"):
        req.add_header("Cookie", os.environ["NADAB_STATS_COOKIE"])
    if os.environ.get("NADAB_STATS_TOKEN"):
        req.add_header("Authorization", "Bearer " + os.environ["NADAB_STATS_TOKEN"])
    with urllib.request.urlopen(req, timeout=60) as r:
        if "csv" not in r.headers.get("Content-Type", ""):
            sys.exit(f"CSV가 아닌 응답: {r.headers.get('Content-Type')} (로그인이 필요해졌을 수 있음)")
        return r.read()


def main():
    raw = fetch()
    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    if list(df.columns) != EXPECTED:
        sys.exit(f"컬럼 구조가 바뀜: {list(df.columns)}")
    if len(df) < 100:
        sys.exit(f"행 수가 비정상적으로 적음: {len(df)}")
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snap = ROOT / "data" / "snapshots" / f"{day}.csv"
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_bytes(raw)
    shutil.copyfile(snap, ROOT / "nadab_daily_question_stats.csv")
    print(f"saved {snap.relative_to(ROOT)}: {len(df)} questions, "
          f"exposures {df['현재 Revision 노출'].sum()}, revisions>1: {(df['현재 Revision'] > 1).sum()}")


if __name__ == "__main__":
    main()
