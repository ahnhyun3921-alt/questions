"""관리자 DB 질문 원본(daily_questions_*.csv)과 통계 사이트 질문 ID를 잇는다.

통계 사이트의 '질문 ID'는 DB id와 다르다. DB id 661~684(2026-07-19에 들어간 24개 묶음)를
통계 사이트가 건너뛰기 때문에, 통계 ID 661 이상은 DB id가 24씩 밀려 있다.
875개 전부 문구로 대조해 확인한 규칙이다(stats_to_db).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INTEREST_ID = {"취향": 1, "감정": 2, "루틴": 3, "관계": 4, "사랑": 5, "가치관": 6}
GAP_START, GAP_LEN = 661, 24


def latest_db():
    files = sorted((ROOT / "data").glob("daily_questions_*.csv"))
    return pd.read_csv(files[-1], encoding="utf-8-sig") if files else None


def stats_to_db(stats_id: int) -> int:
    return stats_id if stats_id < GAP_START else stats_id + GAP_LEN


def check_mapping(stats: pd.DataFrame, db: pd.DataFrame) -> pd.DataFrame:
    """문구가 어긋나는 행을 돌려준다(비어 있어야 정상)."""
    norm = lambda t: str(t).strip().replace(" ", "")
    m = stats.assign(db_id=stats["질문 ID"].map(stats_to_db)).merge(db, left_on="db_id", right_on="id", how="left")
    return m[m["질문 문구"].map(norm) != m["question_text"].map(norm)]
