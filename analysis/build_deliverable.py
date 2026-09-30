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
        ("신규", "신규 질문", len(new)),
        ("신규", "예상 답변율 평균(결론 기준)", round(new["예상 답변율(결론 기준)"].mean(), 3)),
        ("신규", "기존 질문과 최대 글자 유사도", new["글자 유사도"].max()),
        ("디깅", "Drive 자료 중 현재 풀에 없던 질문", len(dig)),
        ("디깅", "채택", int(dig["판정"].str.startswith("채택").sum())),
    ], columns=["구분", "항목", "값"])

    with pd.ExcelWriter(OUT / "나답_질문_수정본_신규.xlsx") as w:
        summ.to_excel(w, sheet_name="요약", index=False)
        fin.to_excel(w, sheet_name="수정본 전체", index=False)
        fin[fin["변경"]].to_excel(w, sheet_name="수정본 변경분", index=False)
        dup.to_excel(w, sheet_name="중복 검토", index=False)
        new.to_excel(w, sheet_name="신규 질문", index=False)
        pd.DataFrame(TONE, columns=["항목", "나답투 규칙"]).to_excel(w, sheet_name="나답투 가이드", index=False)
        dig.to_excel(w, sheet_name="디깅 원천(Drive)", index=False)
        autosize(w)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
