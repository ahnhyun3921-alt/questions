"""관리자 DB 원본(daily_questions_*.csv)과 같은 형식으로, 수정·신규·비활성화를 모두 적용한 전체본을 만든다.

사용법: python analysis/build_full_db.py
결과:   output/daily_questions_revised.csv, output/daily_questions_revised.xlsx

- 기존 열(id ~ deleted_at)은 원본과 같다. 뒤에 참고 열(변경 구분 등)을 붙였다.
- 문구가 바뀐 행은 question_text·가이드 3종·updated_at을 바꾼다.
- 비활성화는 원본처럼 deleted_at을 채운다.
- 신규 행은 id를 비워 둔다(DB가 번호를 매긴다). 신규 ID 열로 구분한다.
- 검토 페이지 선택(data/decisions.csv)을 따른다: 신규·레벨1 묶음에서 '제외(skipped)'는 빼고,
  고른 문구(chosen/applied)가 있으면 그 문구를 쓴다.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from db_questions import INTEREST_ID, latest_db, original_first_ids, stats_to_db  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "output" / "db_export"
G = ["empathy_guide", "hint_guide", "leading_question_guide"]
DB_COLS = ["id", "interest_id", "question_text", "question_level", *G, "created_at", "updated_at", "deleted_at"]


def main():
    now = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S.000 +0900")
    db = latest_db()[DB_COLS].copy()
    db["변경 구분"] = db["deleted_at"].map(lambda x: "이미 삭제됨" if isinstance(x, str) and x else "유지")
    for c in ["통계 질문 ID", "기존 question_text", "신규 ID", "세트", "분석 축"]:
        db[c] = ""
    db = db.astype(object)

    dec_path = ROOT / "data" / "decisions.csv"
    dec = pd.read_csv(dec_path, dtype={"id": str}).set_index("id") if dec_path.exists() else pd.DataFrame()

    def decided(key):
        if not len(dec) or key not in dec.index:
            return None, None
        r = dec.loc[key]
        return r["status"], (r["text"] if isinstance(r.get("text"), str) else None)

    idx = db.set_index("id").index

    def put(i, **kw):
        pos = idx.get_loc(i)
        for k, v in kw.items():
            db.iat[pos, db.columns.get_loc(k)] = v

    # 1) 기존 질문 문구 수정(검토 페이지 선택은 final_questions에 이미 반영됨)
    up = pd.read_csv(EXP / "update_existing.csv")
    tags = pd.read_csv(Path(__file__).parent / "axis_tags.csv").set_index("질문 ID")
    for _, r in up.iterrows():
        cur = db.loc[db.id == r["id"]].iloc[0]
        was_deleted = isinstance(cur["deleted_at"], str) and cur["deleted_at"] != ""
        put(r["id"], **{"기존 question_text": cur["question_text"],
                        "question_text": r["question_text"], **{c: r[c] for c in G}, "updated_at": now,
                        "변경 구분": "이미 삭제됨(다시 쓸 때를 위한 수정안)" if was_deleted else f"문구 수정(가이드 {r['가이드 처리']})",
                        "통계 질문 ID": r["통계 질문 ID"]})
    # 축 태그는 통계 ID 기준
    for sid, t in tags.iterrows():
        i = stats_to_db(int(sid))
        if i in idx:
            put(i, **{"분석 축": t["분석 축"], "통계 질문 ID": sid})

    # 1-2) 기존 질문 관심사 조정(analysis/category_fixes_existing.csv)
    db["관심사 변경"] = pd.Series([""] * len(db), index=db.index, dtype=object)
    cf_path = Path(__file__).parent / "category_fixes_existing.csv"
    if cf_path.exists():
        for _, r in pd.read_csv(cf_path).iterrows():
            if decided(f"K{int(r['통계 질문 ID'])}")[0] == "skipped":   # 검토 페이지에서 '원래대로'
                continue
            i = stats_to_db(int(r["통계 질문 ID"]))
            if i in original_first_ids():   # 원래 첫 질문은 원본 그대로
                continue
            put(i, interest_id=INTEREST_ID[r["제안 관심사"]], **{"관심사 변경": f"{r['지금 관심사']}→{r['제안 관심사']}"})

    # 2) 비활성화(중복 통합)
    for _, r in pd.read_csv(EXP / "deactivate.csv").iterrows():
        put(r["id"], deleted_at=now, **{"변경 구분": "비활성화(중복 통합)"})

    # 3) 레벨 1 첫 질문 묶음(661~684)
    gd_path = Path(__file__).parent / "guides_decided.csv"
    gdec = pd.read_csv(gd_path, dtype=str).set_index("id") if gd_path.exists() else pd.DataFrame()
    lb = pd.read_csv(EXP / "update_level1_block.csv")
    # 검토에서 고르지 않은 레벨 1 묶음 질문은 원본 문구를 그대로 두고 비어 있던 가이드만 채운다
    ko_path = Path(__file__).parent / "new" / "level1_keep_original.csv"
    keep_orig = pd.read_csv(ko_path, dtype={"id": int}).set_index("id") if ko_path.exists() else pd.DataFrame()
    for _, r in lb.iterrows():
        st, txt = decided(f"D{r['id']}")
        if st == "skipped":
            continue
        if r["id"] in keep_orig.index and st not in ("chosen", "applied"):
            put(r["id"], **{c: keep_orig.loc[r["id"], c] for c in G}, updated_at=now, **{"변경 구분": "레벨1 묶음 원본 유지(가이드 추가)"})
            continue
        gs = {c: r[c] for c in G}
        if txt and f"D{r['id']}" in gdec.index and gdec.loc[f"D{r['id']}", "문구"] == txt:
            gs = {c: gdec.loc[f"D{r['id']}", c] for c in G}
        put(r["id"], **{"기존 question_text": r["기존 question_text"], "question_text": txt or r["question_text"],
                        **gs, "updated_at": now, "변경 구분": "레벨1 묶음 재작성"})
    for i in db.loc[db.question_text.str.contains("QA 소셜 집계", na=False), "id"]:
        put(i, deleted_at=now, **{"변경 구분": "비활성화(테스트 행)"})

    # 4) 신규 — DB id는 기존 최대 id 다음부터 매기고 analysis/new/new_db_ids.csv에 고정한다(다시 만들어도 같은 번호)
    idmap_path = Path(__file__).parent / "new" / "new_db_ids.csv"
    idmap = (pd.read_csv(idmap_path, dtype=str).set_index("신규 ID")["DB id"].to_dict() if idmap_path.exists() else {})
    # 개선함 페이지에서 만든 질문(db `added`)이 쓴 id는 건너뛴다 — 루틴이 ArtifactData로 받아 둔 목록
    page_ids_path = Path(__file__).parent / "new" / "page_added_ids.txt"
    page_ids = [int(x) for x in page_ids_path.read_text().split() if x.isdigit()] if page_ids_path.exists() else []
    next_id = max([int(x) for x in db["id"] if str(x).isdigit()] + [int(v) for v in idmap.values()] + page_ids) + 1
    nw = pd.read_csv(EXP / "insert_new.csv")
    keep = []
    for _, r in nw.iterrows():
        st, txt = decided(r["신규 ID"])
        if st == "skipped":
            continue
        if r["신규 ID"] not in idmap:
            idmap[r["신규 ID"]] = str(next_id)
            next_id += 1
        keep.append({"id": idmap[r["신규 ID"]], "interest_id": r["interest_id"], "question_text": txt or r["question_text"],
                     "question_level": r["question_level"], **{c: r[c] for c in G},
                     "created_at": now, "updated_at": now, "deleted_at": "", "변경 구분": "신규",
                     "신규 ID": r["신규 ID"], "세트": r["세트"], "분석 축": r.get("분석 축", "") if isinstance(r.get("분석 축"), str) else ""})
    pd.DataFrame(sorted(idmap.items(), key=lambda kv: int(kv[1])), columns=["신규 ID", "DB id"]).to_csv(idmap_path, index=False, encoding="utf-8-sig")
    # 5) 개선함 페이지에서 만든 신규(analysis/new/page_added.csv — import_page_added.py로 가져옴). id는 페이지가 매긴 그대로
    pa_path = Path(__file__).parent / "new" / "page_added.csv"
    if pa_path.exists():
        have = {str(k["id"]) for k in keep} | {str(x) for x in db["id"]}
        for _, r in pd.read_csv(pa_path, dtype={"DB id": str}).fillna("").iterrows():
            if r["DB id"] in have:
                continue
            keep.append({"id": r["DB id"], "interest_id": int(r["interest_id"]), "question_text": r["문구"], "question_level": int(r["question_level"]),
                         "empathy_guide": r["empathy_guide"], "hint_guide": r["hint_guide"], "leading_question_guide": r["leading_question_guide"],
                         "created_at": r["만든 시각"] or now, "updated_at": r["만든 시각"] or now, "deleted_at": "", "변경 구분": "신규",
                         "신규 ID": "P" + r["DB id"], "세트": "페이지에서 생성", "분석 축": r["분석 축"]})
    out = pd.concat([db, pd.DataFrame(keep)], ignore_index=True)
    out["deleted_at"] = out["deleted_at"].fillna("")

    # 6) 관심사 재배정 일괄 적용(analysis/recat_applied.csv — 개선함 '관심사 재배정' 후보 전체).
    #    사람이 중심인 질문은 가치관보다 관계·사랑이 우선. '관심사 변경'은 원본 DB 관심사 기준으로 다시 적는다.
    ra_path = Path(__file__).parent / "recat_applied.csv"
    if ra_path.exists():
        orig = pd.read_csv(sorted((ROOT / "data").glob("daily_questions_*.csv"))[-1], dtype=str).set_index("id")["interest_id"]
        name = {v: k for k, v in INTEREST_ID.items()}
        out["id"] = out["id"].astype(str)
        for _, r in pd.read_csv(ra_path, dtype=str).iterrows():
            m = out["id"] == r["DB id"]
            if not m.any():
                continue
            out.loc[m, "interest_id"] = INTEREST_ID[r["제안 관심사"]]
            o = name.get(int(float(orig[r["DB id"]]))) if r["DB id"] in orig.index else None
            out.loc[m, "관심사 변경"] = f"{o}→{r['제안 관심사']}" if o and o != r["제안 관심사"] else ""

    out.to_csv(ROOT / "output" / "daily_questions_revised.csv", index=False, encoding="utf-8-sig")
    live = out[out.deleted_at == ""]
    out["관심사 변경"] = out["관심사 변경"].fillna("")
    summ = pd.concat([out["변경 구분"].value_counts().rename("행 수").reset_index(),
                      pd.DataFrame([{"변경 구분": "(별도) 기존 질문 관심사 변경", "행 수": int((out["관심사 변경"] != "").sum())}])])
    with pd.ExcelWriter(ROOT / "output" / "daily_questions_revised.xlsx") as w:
        summ.to_excel(w, sheet_name="요약", index=False)
        out.to_excel(w, sheet_name="전체", index=False)
        live.to_excel(w, sheet_name="활성만", index=False)
        out[(out["변경 구분"] != "유지") | (out["관심사 변경"] != "")].to_excel(w, sheet_name="바뀌는 것만", index=False)
        out[out["관심사 변경"] != ""].to_excel(w, sheet_name="관심사 변경", index=False)
        for ws in w.book.worksheets:
            ws.freeze_panes = "D2"
    print(f"전체 {len(out)}행 · 활성 {len(live)}개 (레벨1 {int((live.question_level == 1).sum())}) · 가이드 빈 칸 {int(live[G].isna().any(axis=1).sum())}")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
