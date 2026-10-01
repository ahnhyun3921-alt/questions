"""875개 전체 질문의 최종 권장 문구 목록.

사용법: python analysis/build_final_list.py
결과:   output/final_questions.csv, output/final_questions.xlsx
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lint_questions import lint  # noqa: E402
from db_questions import latest_db, stats_to_db  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"


def main():
    card = pd.read_csv(OUT / "question_scorecard.csv")
    rw = pd.read_csv(Path(__file__).parent / "rewrites.csv").fillna("")
    sc = pd.read_csv(OUT / "rewrites_scored.csv")[["질문 ID", "조치 유형", "예측 변화(%p)"]]
    pri = pd.read_csv(OUT / "v2" / "priority.csv")[["id", "우선순위", "기대 추가 답변(월)"]].rename(columns={"id": "질문 ID"})
    d = card[["질문 ID", "관심사", "레벨", "상태", "분류", "노출", "답변", "교체", "질문 문구"]].merge(rw, on="질문 ID", how="left")
    d = d.merge(sc, on="질문 ID", how="left").merge(pri, on="질문 ID", how="left")
    d = d.fillna({"원인 진단": "", "수정 원칙": "", "수정안 1 (권장)": "", "수정안 2 (대안)": "", "조치 유형": ""})

    dec_path = ROOT / "data" / "decisions.csv"
    dec = pd.read_csv(dec_path, dtype={"id": str}).set_index("id") if dec_path.exists() else pd.DataFrame()
    d["검토 상태"] = d["질문 ID"].astype(str).map(dec["status"]) if len(dec) else None
    d["검토에서 고른 문구"] = d["질문 ID"].astype(str).map(dec["text"]) if len(dec) else None

    def final(r):
        # 검토 페이지에서 사용자가 고른 문구가 있으면 그것이 최우선
        if isinstance(r.get("검토에서 고른 문구"), str) and r.get("검토 상태") in ("chosen", "applied"):
            # 수정안 2가 '(새 질문 없이 중복만 제거)' 같은 처리 지시였다면 문구가 아니라 비활성화
            if "중복만 제거" in r["검토에서 고른 문구"]:
                return "(비활성화)"
            return r["검토에서 고른 문구"]
        o1 = r["수정안 1 (권장)"]
        if o1.startswith("(통합"):
            return "(비활성화)"
        if not o1 or o1.startswith("(유지"):
            return r["질문 문구"]
        return o1

    def kind(r):
        if r["조치 유형"]:
            return r["조치 유형"]
        if r["상태"] == "INACTIVE":
            return "비활성(유지)"
        return "유지"

    def why(r):
        if r["원인 진단"]:
            return r["원인 진단"]
        c = r["분류"][0]
        if c == "D":
            return "성과 우수 (평균보다 확실히 높음)"
        if any(l == "가이드" for l, _ in lint(r["질문 문구"])):
            return "40자 초과지만 양자택일형 (길이 효과는 유의하지 않음, 형식은 상위권)"
        if c == "E":
            return "아직 노출 없음 · 위험 요소 없음"
        return "위험 요소 없음 (선호·성향·양자택일형 등)"

    d["최종 권장 문구"] = d.apply(final, axis=1)
    d["변경"] = d["최종 권장 문구"] != d["질문 문구"]
    d.loc[d["최종 권장 문구"] == "(비활성화)", "조치"] = "중복 통합(비활성화)"
    d["조치"] = d.apply(kind, axis=1)
    d.loc[d["최종 권장 문구"] == "(비활성화)", "조치"] = "중복 통합(비활성화)"
    d["이유"] = d.apply(why, axis=1)
    d["최종 문구 검수"] = d["최종 권장 문구"].map(lambda q: "" if q == "(비활성화)" else q).map(lambda q: " / ".join(m.split(" (")[0].split(".")[0] for l, m in lint(q) if l != "가이드"))
    cols = ["질문 ID", "관심사", "레벨", "분류", "노출", "답변", "교체", "질문 문구", "최종 권장 문구", "변경", "조치",
            "이유", "수정 원칙", "수정안 2 (대안)", "예측 변화(%p)", "우선순위", "기대 추가 답변(월)", "검토 상태", "최종 문구 검수"]
    d = d[cols].rename(columns={"질문 문구": "기존 문구", "수정안 2 (대안)": "대안"}).sort_values("질문 ID")
    # 관리자 DB id와 현재 가이드 3종, 자아분석 축 태그
    d.insert(1, "DB id", d["질문 ID"].map(stats_to_db))
    db = latest_db()
    if db is not None:
        g = db[["id", "empathy_guide", "hint_guide", "leading_question_guide"]].rename(columns={"id": "DB id"})
        d = d.merge(g, on="DB id", how="left")
        d["가이드 점검"] = d.apply(lambda r: "문구 변경 → 가이드 재작성 필요" if r["변경"] and r["최종 권장 문구"] != "(비활성화)" else "", axis=1)
    tags = Path(__file__).parent / "axis_tags.csv"
    if tags.exists():
        t = pd.read_csv(tags)[["질문 ID", "분석 축", "신호", "해석 가이드"]]
        d = d.merge(t, on="질문 ID", how="left")
    d.to_csv(OUT / "final_questions.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "final_questions.xlsx") as w:
        summ = d.groupby("조치").size().rename("질문 수").reset_index().sort_values("질문 수", ascending=False)
        summ.to_excel(w, sheet_name="요약", index=False)
        d.to_excel(w, sheet_name="전체 875개", index=False)
        d[d["변경"]].to_excel(w, sheet_name="변경 대상만", index=False)
        for ws in w.book.worksheets:
            ws.freeze_panes = "B2"
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col[:60])
                ws.column_dimensions[col[0].column_letter].width = min(max(8, width * 1.5), 55)
    print(d["조치"].value_counts().to_string())
    print("변경", int(d["변경"].sum()), "/ 유지", int((~d["변경"]).sum()))
    print("최종 문구 중 검수 경고 남은 것:", int((d["최종 문구 검수"] != "").sum()))


if __name__ == "__main__":
    main()
