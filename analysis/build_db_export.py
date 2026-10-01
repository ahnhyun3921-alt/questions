"""관리자 DB(daily_questions)에 바로 반영할 수 있는 파일을 만든다.

사용법: python analysis/build_db_export.py
결과(output/db_export/):
  update_existing.csv  문구가 바뀌는 기존 질문 — DB id 기준, 새 문구 + 가이드 3종
  deactivate.csv       중복 통합으로 비활성화할 질문
  insert_new.csv       신규 질문 — DB 열 이름 그대로 + 참고 열
  update_level1_block.csv  DB id 661~684(레벨 1 첫 질문 묶음) — 가볍게 재작성한 문구 + 가이드
  db_issues.csv        DB 자체의 점검 사항(가이드 누락, 테스트 행, 중복 행)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from db_questions import INTEREST_ID, latest_db, original_first_ids  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
EXP = OUT / "db_export"
G = ["empathy_guide", "hint_guide", "leading_question_guide"]
NEW_SETS = [("new_questions_self.csv", "자아분석"), ("new_questions_fun.csv", "재미 보강"), ("new_questions_love.csv", "연애"),
            ("new_questions_scraped.csv", "외부 디깅"), ("new_questions.csv", "내부 보강")]


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    db = latest_db()
    f = pd.read_csv(OUT / "final_questions.csv")
    gw = pd.read_csv(Path(__file__).parent / "guides_rewrite.csv").fillna("")

    # 1) 문구가 바뀌는 기존 질문
    ch = f[f["가이드 점검"].fillna("") != ""].merge(gw, on="질문 ID", how="left", suffixes=("_현재", ""))
    for c in G:
        ch[c] = ch.apply(lambda r: r[c] if r["가이드 처리"] == "재작성" else r[f"{c}_현재"], axis=1)
    # 검토 페이지에서 수정안 1이 아닌 문구를 골랐다면 그 문구에 맞춰 쓴 가이드(guides_decided.csv)
    gd_path = Path(__file__).parent / "guides_decided.csv"
    if gd_path.exists():
        gd = pd.read_csv(gd_path, dtype=str).set_index("id")
        for i, r in ch.iterrows():
            k = str(r["질문 ID"])
            if k in gd.index and gd.loc[k, "문구"] == r["최종 권장 문구"]:
                for c in G:
                    ch.at[i, c] = gd.loc[k, c]
                ch.at[i, "가이드 처리"] = "재작성"
    up = ch.rename(columns={"DB id": "id", "최종 권장 문구": "question_text", "기존 문구": "기존 question_text", "질문 ID": "통계 질문 ID"})
    up = up[["id", "통계 질문 ID", "기존 question_text", "question_text", *G, "가이드 처리", "조치", "이유"]]
    keep1 = original_first_ids(db)   # 원래 첫 질문(레벨 1)은 원본 그대로
    up = up[~up["id"].astype(int).isin(keep1)]
    up.to_csv(EXP / "update_existing.csv", index=False, encoding="utf-8-sig")

    # 2) 비활성화
    de = f[f["최종 권장 문구"] == "(비활성화)"].rename(columns={"DB id": "id", "질문 ID": "통계 질문 ID"})
    de = de[~de["id"].astype(int).isin(keep1)]
    de[["id", "통계 질문 ID", "기존 문구", "이유"]].to_csv(EXP / "deactivate.csv", index=False, encoding="utf-8-sig")

    # 3) 신규
    gn = pd.read_csv(Path(__file__).parent / "new" / "guides_new.csv")
    parts = []
    for fn, label in NEW_SETS:
        n = pd.read_csv(OUT / fn)
        n["세트"] = label
        parts.append(n)
    nw = pd.concat(parts, ignore_index=True).merge(gn, on="신규 ID", how="left")
    miss = nw[nw[G[0]].isna()]["신규 ID"].tolist()
    if miss:
        raise SystemExit(f"가이드가 없는 신규 질문: {miss}")
    nw["interest_id"] = nw["관심사"].map(INTEREST_ID)
    nw = nw.rename(columns={"문구": "question_text", "레벨": "question_level"})
    cols = ["interest_id", "question_text", "question_level", *G, "신규 ID", "세트", "관심사", "형식",
            "예상 답변율(결론 기준)", "분석 축", "해석 가이드", "출처", "출처 URL"]
    nw[[c for c in cols if c in nw]].to_csv(EXP / "insert_new.csv", index=False, encoding="utf-8-sig")

    # 4) 레벨 1(첫 질문) 묶음 DB id 661~684: 가볍게 재작성한 문구 + 가이드
    lb_path = Path(__file__).parent / "new" / "level1_block.csv"
    lb = pd.read_csv(lb_path) if lb_path.exists() else pd.DataFrame()
    if len(lb):
        lb.rename(columns={"DB id": "id", "기존 문구": "기존 question_text", "권장 문구": "question_text"})[
            ["id", "interest_id", "question_level", "기존 question_text", "question_text", *G, "판정"]
        ].to_csv(EXP / "update_level1_block.csv", index=False, encoding="utf-8-sig")

    # 5) DB 자체 점검
    live = db[db.deleted_at.isna()].copy()
    issues = []
    for _, r in live[live.empathy_guide.isna()].iterrows():
        issues.append((r.id, r.question_text, "가이드 3종 없음 (레벨 1 첫 질문 묶음, 2026-07-19 추가, 통계 사이트 목록에는 없음)"))
    for _, r in live[live.question_text.str.contains("QA|테스트|test", case=False, na=False)].iterrows():
        issues.append((r.id, r.question_text, "테스트용으로 보이는 행이 활성 상태"))
    norm = live.question_text.str.replace(" ", "").str.strip()
    for t, g in live.groupby(norm):
        if len(g) > 1:
            issues.append((",".join(map(str, sorted(g.id))), g.question_text.iloc[0], f"같은 문구가 {len(g)}개 활성"))
    iss = pd.DataFrame(issues, columns=["DB id", "question_text", "점검 사항"])
    fixed = set(up["id"].astype(str)) | set(de["id"].astype(str))
    block = set(lb["DB id"].astype(str)) if len(lb) else set()

    def resolved(row):
        ids = str(row["DB id"]).split(",")
        if len(ids) == 1 and ids[0] in block:
            return "해소(update_level1_block.csv에 문구·가이드)"
        if len(ids) > 1 and any(i in fixed | block for i in ids):
            return "해소(한쪽 문구 변경·비활성화)"
        return ""
    iss["수정본에서"] = iss.apply(resolved, axis=1)
    iss.to_csv(EXP / "db_issues.csv", index=False, encoding="utf-8-sig")

    print(f"수정 {len(up)}개 (가이드 재작성 {(up['가이드 처리'] == '재작성').sum()}, 유지 {(up['가이드 처리'] == '유지').sum()})"
          f" · 비활성화 {len(de)}개 · 신규 {len(nw)}개 · 레벨1 묶음 {len(lb)}개 · DB 점검 {len(issues)}건")


if __name__ == "__main__":
    main()
