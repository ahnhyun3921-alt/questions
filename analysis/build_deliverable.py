"""전달용 통합 엑셀: 수정본 + 중복 검토 + 신규 질문 + 나답투 가이드 + 디깅 원천.

사용법: python analysis/build_deliverable.py  →  output/나답_질문_수정본_신규.xlsx
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

TONE = [
    ("길이", "24~35자 (기존 중앙값 29자). 40자를 넘기면 조건을 줄인다."),
    ("어미", "해요체 의문형 '~인가요?'·'~나요?'. 양자택일은 구어 '뭐가 더 ~나요?'. '~시나요'·'~입니까' 같은 높임·격식체는 쓰지 않는다."),
    ("주어", "1인칭 '나·내'로 묻는다. '당신·본인·여러분'은 쓰지 않는다 (기존 875개 중 0건)."),
    ("한 번에 하나", "물음표 1개, 조건 1개. '~하고, ~한다면' 식 이중 조건을 피한다."),
    ("시점", "'오늘'을 쓰지 않는다 (답변 오즈 0.42배). 시점 없는 '평소·주로'가 기본. '오늘 밤' 질문은 저녁 노출 전용."),
    ("경험 전제", "'~한 적이 있다면'을 쓰지 않는다 (0.48~0.64배). 필요하면 '없다면 ~을 알려주세요'로 출구를 준다."),
    ("사람", "'누구인가요'로 특정인을 지목하지 않고 그 사람의 특징·이유를 묻는다."),
    ("무게", "상처·비밀·가족 갈등은 회복·선호 쪽으로 각도를 튼다. 깊은 성찰형은 레벨 2로만."),
    ("형식", "선호형(57%)·양자택일형(55%)·성향형(51%) 우선. 일화 회상형(40%)은 줄인다."),
    ("외부 질문 옮기기", "검사 문항(예: I get chores done right away)은 '~하는 편인가요, ~하는 편인가요?' 양자택일로. 저작권 있는 검사(MBTI·16Personalities)는 문장이 아니라 축 개념만 차용. '당신'→'나', 무거운 주제는 가벼운 각도로."),
    ("말투 예시", "가장 좋아하는 ___은 무엇인가요? / ___하는 편인가요, ___하는 편인가요? / A와 B 중 뭐가 더 ___나요?"),
]


def autosize(w):
    for ws in w.book.worksheets:
        ws.freeze_panes = "B2"
        for col in ws.columns:
            width = max(len(str(c.value or "")) for c in col[:80])
            ws.column_dimensions[col[0].column_letter].width = min(max(8, width * 1.4), 60)


def main():
    fin = pd.read_csv(OUT / "final_questions.csv")
    new = pd.read_csv(OUT / "new_questions.csv")
    ext = pd.read_csv(OUT / "new_questions_scraped.csv")
    love = pd.read_csv(OUT / "new_questions_love.csv")
    fun = pd.read_csv(OUT / "new_questions_fun.csv")
    selfq = pd.read_csv(OUT / "new_questions_self.csv")
    axes = (selfq.groupby(["분석 축"]).agg(질문_수=("문구", "size"), 근거_개념=("근거 개념", lambda x: " · ".join(sorted(set(x)))),
                                        예시=("문구", "first")).reset_index().sort_values("질문_수", ascending=False))
    src = (ext.groupby(["출처", "출처 URL", "인기·검증 근거"]).size().rename("채택 질문 수").reset_index()
           .sort_values("채택 질문 수", ascending=False))
    rw = pd.read_csv(Path(__file__).parent / "rewrites.csv").fillna("")
    card = pd.read_csv(OUT / "question_scorecard.csv")[["질문 ID", "질문 문구"]]
    dup = rw[rw["수정 원칙"].str.contains("중복") | rw["원인 진단"].str.contains("중복")].merge(card, on="질문 ID")
    dup = dup.assign(처리=lambda x: x["수정안 1 (권장)"].map(lambda t: "비활성화(통합)" if t.startswith("(통합") else "다른 각도로 재작성"))
    dup = dup[["질문 ID", "질문 문구", "수정 원칙", "원인 진단", "처리", "수정안 1 (권장)"]].rename(
        columns={"질문 문구": "기존 문구", "수정 원칙": "중복 상대·원칙", "수정안 1 (권장)": "최종 처리 문구"}).sort_values("질문 ID")
    dig = pd.read_csv(Path(__file__).parent / "new" / "drive_dig_candidates.csv")
    used = set(pd.read_csv(Path(__file__).parent / "new" / "candidates.csv")["참고 원문"].dropna())
    dig["채택"] = dig["문구"].isin(used)
    dig["판정"] = dig.apply(lambda r: "채택(나답투로 변환)" if r["채택"] else
                          ("기존 질문의 예전 문구" if r["유사도"] >= 0.5 else "미채택(기존과 의미 중복·무게 과다·추상적)"), axis=1)
    dig = dig[["출처", "분류", "문구", "판정", "최근접 기존", "유사도", "가이드"]]

    summ = pd.DataFrame([
        ("수정본", "전체 질문", len(fin)),
        ("수정본", "문구 변경", int((fin["변경"] & (fin["조치"] != "중복 통합(비활성화)")).sum())),
        ("수정본", "유지", int((~fin["변경"]).sum() - (fin["조치"] == "중복 통합(비활성화)").sum())),
        ("수정본", "중복 통합(비활성화)", int((fin["조치"] == "중복 통합(비활성화)").sum())),
        ("중복 검토", "중복 해소한 질문", len(dup)),
        ("신규(외부 디깅)", "커뮤니티·검사 원천에서 옮긴 질문", len(ext)),
        ("신규(외부 디깅)", "원천 수", ext["출처"].nunique()),
        ("신규(외부 디깅)", "검수 통과", int((ext["검수"] == "통과").sum())),
        ("신규(외부 디깅)", "예상 답변율 평균(결론 기준)", round(ext["예상 답변율(결론 기준)"].mean(), 3)),
        ("신규(내부 보강)", "부족한 형식·카테고리 보강 질문", len(new)),
        ("신규(내부 보강)", "예상 답변율 평균(결론 기준)", round(new["예상 답변율(결론 기준)"].mean(), 3)),
        ("신규(연애)", "재미 중심 연애 질문", len(love)),
        ("신규(연애)", "예상 답변율 평균(결론 기준)", round(love["예상 답변율(결론 기준)"].mean(), 3)),
        ("신규(재미 보강)", "전 카테고리 재미 질문", len(fun)),
        ("신규(재미 보강)", "예상 답변율 평균(결론 기준)", round(fun["예상 답변율(결론 기준)"].mean(), 3)),
        ("신규(자아분석)", "공감형 자아분석 질문", len(selfq)),
        ("신규(자아분석)", "분석 축 수", selfq["분석 축"].nunique()),
        ("신규(자아분석)", "예상 답변율 평균(결론 기준)", round(selfq["예상 답변율(결론 기준)"].mean(), 3)),
        ("신규 전체", "기존 풀 대비 최대 글자 유사도", max(x["글자 유사도"].max() for x in (new, ext, love, fun, selfq))),
        ("Drive 자료", "현재 풀에 없던 질문", len(dig)),
        ("Drive 자료", "채택(내부 보강에 포함)", int(dig["판정"].str.startswith("채택").sum())),
    ], columns=["구분", "항목", "값"])

    with pd.ExcelWriter(OUT / "나답_질문_수정본_신규.xlsx") as w:
        summ.to_excel(w, sheet_name="요약", index=False)
        fin.to_excel(w, sheet_name="수정본 전체", index=False)
        fin[fin["변경"]].to_excel(w, sheet_name="수정본 변경분", index=False)
        dup.to_excel(w, sheet_name="중복 검토", index=False)
        ext.to_excel(w, sheet_name="신규(외부 디깅)", index=False)
        love.to_excel(w, sheet_name="신규(연애)", index=False)
        fun.to_excel(w, sheet_name="신규(재미 보강)", index=False)
        selfq.sort_values(["분석 축", "신규 ID"]).to_excel(w, sheet_name="신규(자아분석)", index=False)
        axes.to_excel(w, sheet_name="자아분석 축", index=False)
        pd.read_csv(Path(__file__).parent / "axis_tags.csv").to_excel(w, sheet_name="기존 질문 축 태그", index=False)
        exp = OUT / "db_export"
        for fn, name in [("update_existing.csv", "DB 반영(수정)"), ("insert_new.csv", "DB 반영(신규)"),
                         ("update_level1_block.csv", "DB 반영(레벨1 661~684)"),
                         ("deactivate.csv", "DB 반영(비활성화)"), ("db_issues.csv", "DB 점검")]:
            if (exp / fn).exists():
                pd.read_csv(exp / fn).to_excel(w, sheet_name=name, index=False)
        src.to_excel(w, sheet_name="디깅 출처", index=False)
        new.to_excel(w, sheet_name="신규(내부 보강)", index=False)
        pd.DataFrame(TONE, columns=["항목", "나답투 규칙"]).to_excel(w, sheet_name="나답투 가이드", index=False)
        dig.to_excel(w, sheet_name="Drive 자료 검토", index=False)
        autosize(w)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
