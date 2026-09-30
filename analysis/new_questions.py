"""신규 질문 후보 평가: 기존 풀과 중복 검사 + 검수 + v2 태깅 모델로 예상 답변율.

사용법: python analysis/new_questions.py
입력:  analysis/new/candidates.csv (출처, 참고 원문, 관심사, 레벨, 문구, T,P,E,W,S,F,C)
출력:  output/new_questions.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from analyze_v2 import F_LABEL, design, load  # noqa: E402
from dedupe import pairs  # noqa: E402
from lint_questions import lint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    c = pd.read_csv(Path(__file__).parent / "new" / "candidates.csv")
    d = load()
    s = d[d.res > 0]
    X = sm.add_constant(design(s, "tags"))
    m = sm.GLM(np.column_stack([s.ans, s.rep]), X, family=sm.families.Binomial()).fit()
    nd = c.rename(columns={"관심사": "cat", "레벨": "level"}).copy()
    Xn = sm.add_constant(design(nd, "tags"), has_constant="add").reindex(columns=X.columns, fill_value=0)
    c["예상 답변율(결론 기준)"] = m.predict(Xn).round(3)
    base = s.ans.sum() / s.res.sum()
    c["기존 평균 대비(%p)"] = ((c["예상 답변율(결론 기준)"] - base) * 100).round(1)
    c["형식"] = c.F.map(F_LABEL)

    f = pd.read_csv(ROOT / "output" / "final_questions.csv")
    f = f[~f["조치"].astype(str).str.startswith(("비활성", "중복 통합"))]
    cur = pd.read_csv(ROOT / "nadab_daily_question_stats.csv", encoding="utf-8-sig")
    pool = pd.concat([f[["질문 ID", "최종 권장 문구"]].rename(columns={"최종 권장 문구": "q"}),
                      cur[["질문 ID", "질문 문구"]].rename(columns={"질문 문구": "q"})]).drop_duplicates("q").reset_index(drop=True)
    pool["q"] = pool.q.str.strip()
    best = {}
    for a, b, cs, j in pairs(c["문구"].tolist(), pool.q.tolist(), char_thr=0.0, jac_thr=2):
        if a not in best or cs > best[a][2]:
            best[a] = (int(pool["질문 ID"][b]), pool.q[b], cs)
    c["가장 비슷한 기존 질문"] = [f"#{best[i][0]} {best[i][1]}" if i in best else "" for i in range(len(c))]
    c["글자 유사도"] = [round(best[i][2], 2) if i in best else 0 for i in range(len(c))]
    c["검수"] = c["문구"].map(lambda q: " / ".join(m.split(" (")[0].split(".")[0] for l, m in lint(q) if l != "가이드") or "통과")
    c.insert(0, "신규 ID", [f"N{i + 1:03d}" for i in range(len(c))])
    cols = ["신규 ID", "관심사", "레벨", "문구", "형식", "예상 답변율(결론 기준)", "기존 평균 대비(%p)", "출처", "참고 원문",
            "가장 비슷한 기존 질문", "글자 유사도", "검수", "T", "P", "E", "W", "S", "F", "C"]
    c = c[cols].sort_values(["관심사", "예상 답변율(결론 기준)"], ascending=[True, False])
    c.to_csv(ROOT / "output" / "new_questions.csv", index=False, encoding="utf-8-sig")
    print(f"기존 평균 {base:.3f} · 신규 평균 {c['예상 답변율(결론 기준)'].mean():.3f} · 최대 유사도 {c['글자 유사도'].max()}")
    print(c.groupby("관심사").size().to_dict(), c["형식"].value_counts().to_dict())
    print(c[["신규 ID", "문구", "예상 답변율(결론 기준)", "글자 유사도", "검수"]].to_string(index=False))


if __name__ == "__main__":
    main()
