"""나답 데일리 질문 성과 분석.

사용법: python analysis/analyze.py
입력:  nadab_daily_question_stats.csv
출력:  output/summary.json, output/question_scorecard.csv

방법
1. 퍼널: 노출 = 답변 + 교체 + 미응답 (합계 검증)
2. 세그먼트: 관심사·레벨·문구 특성별 합산 답변율 + Wilson 95% CI
3. 원인 모델: 이항 GLM(답변/노출) ~ 관심사 + 레벨 + 문구 특성 → 오즈비
4. 질문별 점수: GLM 예측값을 사전평균으로 쓰는 Beta-Binomial 회귀 축소
   (노출 2~3회짜리 질문의 우연한 0%/100%를 걸러냄)
5. 분류: 교체 우선 / 수정 권장 / 모니터링 / 우수 / 미노출
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, special, stats
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURES, RISK_FEATURES, extract  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

COLS = ["id", "q", "cat_code", "cat", "level", "status", "rev", "rev_at",
        "exp", "ans", "ar", "rep", "rr", "nr", "texp", "tans", "tar", "trep", "tnr"]


def load():
    d = pd.read_csv(ROOT / "nadab_daily_question_stats.csv", encoding="utf-8-sig")
    d.columns = COLS
    d["q"] = d["q"].str.strip()
    assert (d.exp == d.ans + d.rep + d.nr).all(), "노출 ≠ 답변+교체+미응답"
    assert (d.exp == d.texp).all(), "Revision이 1개가 아닌 질문이 있음"
    f = pd.DataFrame([extract(q) for q in d.q])
    return pd.concat([d, f], axis=1)


def rate_row(s, label, value):
    e, a, r, n = s.exp.sum(), s.ans.sum(), s.rep.sum(), s.nr.sum()
    lo, hi = proportion_confint(a, e, method="wilson") if e else (np.nan, np.nan)
    return {"group": label, "value": str(value), "questions": len(s), "exposures": int(e),
            "answer_rate": a / e if e else None, "ci_lo": lo, "ci_hi": hi,
            "replace_rate": r / e if e else None, "no_response_rate": n / e if e else None,
            "exp_per_q": e / len(s) if len(s) else None, "zero_exposure": int((s.exp == 0).sum())}


def segments(d):
    rows = [rate_row(d, "전체", "전체")]
    for col, label in [("cat", "관심사"), ("level", "레벨")]:
        for v, s in d.groupby(col):
            rows.append(rate_row(s, label, v))
    for k in FEATURES:
        rows.append(rate_row(d[d[k]], "문구 특성", k))
        rows.append(rate_row(d[~d[k]], "문구 특성(해당 없음)", k))
    d["길이 구간"] = pd.cut(d["문구 길이"], [0, 20, 30, 40, 100], labels=["~20자", "21~30자", "31~40자", "41자~"])
    for v, s in d.groupby("길이 구간", observed=True):
        rows.append(rate_row(s, "문구 길이", v))
    return pd.DataFrame(rows)


def design(d):
    X = pd.get_dummies(d["cat"], prefix="관심사", dtype=float).drop(columns="관심사_가치관")
    X["레벨2"] = (d.level == 2).astype(float)
    for k in FEATURES:
        X[k] = d[k].astype(float)
    X["문구 길이(10자)"] = d["문구 길이"] / 10
    return sm.add_constant(X)


def fit_glm(d):
    s = d[d.exp > 0]
    X = design(s)
    y = np.column_stack([s.ans, s.exp - s.ans])
    # scale=1 (보수적): 관측 분산이 이항보다 작아도 SE를 줄이지 않음
    m = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    ci = m.conf_int()
    coef = pd.DataFrame({"term": m.params.index, "odds_ratio": np.exp(m.params.values),
                         "or_lo": np.exp(ci[0].values), "or_hi": np.exp(ci[1].values),
                         "p_value": m.pvalues.values})
    return m, coef[coef.term != "const"]


def shrink(d, m):
    """GLM 예측값 mu_i를 사전평균으로 하는 Beta(k*mu, k*(1-mu)) 사전분포, k는 주변우도 최대화."""
    mu = m.predict(design(d))
    s = d.exp > 0

    def nll(logk):
        k = np.exp(logk)
        a, b = k * mu[s], k * (1 - mu[s])
        return -np.sum(special.betaln(d.ans[s] + a, d.exp[s] - d.ans[s] + b) - special.betaln(a, b))

    k = float(np.exp(optimize.minimize_scalar(nll, bounds=(-2, 8), method="bounded").x))
    a, b = k * mu + d.ans, k * (1 - mu) + d.exp - d.ans
    base = d.ans.sum() / d.exp.sum()
    d["예측 답변율(문구 모델)"] = mu
    d["추정 답변율(축소)"] = a / (a + b)
    d["P(평균 미만)"] = stats.beta.cdf(base, a, b)
    d["추정 하한(10%)"] = stats.beta.ppf(0.1, a, b)
    d["추정 상한(90%)"] = stats.beta.ppf(0.9, a, b)

    # 비교용: 문구 정보 없이 관측 데이터만 쓴 축소(전체 공통 Beta 사전분포)
    def nll0(p):
        a0, b0 = np.exp(p)
        return -np.sum(special.betaln(d.ans[s] + a0, d.exp[s] - d.ans[s] + b0) - special.betaln(a0, b0))

    a0, b0 = np.exp(optimize.minimize(nll0, [0.0, 0.0]).x)
    d["P(평균 미만|데이터만)"] = stats.beta.cdf(base, a0 + d.ans, b0 + d.exp - d.ans)
    return k, base


def classify(d, base):
    d["위험 특성"] = d[RISK_FEATURES].apply(lambda r: ", ".join(k for k in RISK_FEATURES if r[k]), axis=1)
    has_risk = d["위험 특성"] != ""
    p = d["P(평균 미만)"]
    grp = np.select(
        [d.status == "INACTIVE",
         # 문구 모델과 관측 데이터가 모두 '평균 미만'을 가리킬 때만 교체 우선
         (p >= 0.8) & (d["P(평균 미만|데이터만)"] >= 0.6) & (d.exp >= 3),
         has_risk & (d["추정 답변율(축소)"] < base),
         (p <= 0.2) & (d["P(평균 미만|데이터만)"] <= 0.4) & (d.exp >= 3),
         d.exp == 0],
        ["비활성", "A. 교체·재작성 우선", "B. 패턴 기반 수정 권장", "D. 우수(레퍼런스)", "E. 미노출(데이터 없음)"],
        default="C. 유지·모니터링")
    d["분류"] = grp
    return d


GUIDE = {
    "오늘 기반": "'오늘'→'요즘/평소'로 시점 중립화, 또는 저녁 시간대에만 노출",
    "경험 전제형(~있다면/~적 있)": "'~한 적이 있다면'→'~하는 편인가요/주로 무엇인가요' 성향형으로 전환하거나 '없다면 ~' 대안 제시",
    "최근·이번 주 회상": "기간 제한 제거('최근'→'지금까지/평소')",
    "특정 활동 전제": "대상 확장(예: 책→책·영화·노래 무엇이든) 또는 활동 비경험자용 대안 제시",
}


def guidance(risk: str) -> str:
    return " / ".join(GUIDE[k] for k in RISK_FEATURES if k in risk.split(", "))


def write_xlsx(card, seg, coef):
    base_cols = ["질문 ID", "질문 문구", "관심사", "레벨", "노출", "답변", "교체", "미응답",
                 "추정 답변율(축소)", "추정 하한(10%)", "추정 상한(90%)", "P(평균 미만)", "위험 특성", "분류"]
    a = card[card["분류"].str.startswith("A")][base_cols[:8] + ["추정 답변율(축소)", "원인 진단", "수정 원칙",
                                                             "수정안 1 (권장)", "수정안 2 (대안)"]]
    b = card[card["분류"].str.startswith("B")][base_cols + ["수정 방향(패턴)", "원인 진단", "수정 원칙",
                                                             "수정안 1 (권장)", "수정안 2 (대안)"]]
    dd = card[card["분류"].str.startswith("D")][base_cols]
    with pd.ExcelWriter(OUT / "question_review.xlsx") as w:
        pd.DataFrame({"분류": card["분류"].value_counts().index,
                      "질문 수": card["분류"].value_counts().values}).to_excel(w, sheet_name="분류 요약", index=False)
        a.to_excel(w, sheet_name="A 교체·재작성", index=False)
        b.to_excel(w, sheet_name="B 패턴 수정(수정안 포함)", index=False)
        dd.to_excel(w, sheet_name="D 우수 레퍼런스", index=False)
        card.to_excel(w, sheet_name="전체 스코어카드", index=False)
        seg.to_excel(w, sheet_name="세그먼트", index=False)
        coef.to_excel(w, sheet_name="원인 모델(오즈비)", index=False)
        for ws in w.book.worksheets:
            ws.freeze_panes = "B2" if ws.title != "분류 요약" else None
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col[:50])
                ws.column_dimensions[col[0].column_letter].width = min(max(8, width * 1.6), 60)


def main():
    d = load()
    seg = segments(d)
    m, coef = fit_glm(d)
    k, base = shrink(d, m)
    d = classify(d, base)

    keep = ["id", "q", "cat", "level", "status", "exp", "ans", "rep", "nr", "ar", "rr",
            "예측 답변율(문구 모델)", "추정 답변율(축소)", "추정 하한(10%)", "추정 상한(90%)",
            "P(평균 미만)", "P(평균 미만|데이터만)", "위험 특성", "분류"] + list(FEATURES) + ["문구 길이"]
    card = d[keep].rename(columns={"id": "질문 ID", "q": "질문 문구", "cat": "관심사", "level": "레벨",
                                   "status": "상태", "exp": "노출", "ans": "답변", "rep": "교체",
                                   "nr": "미응답", "ar": "답변율(%)", "rr": "교체율(%)"})
    card["수정 방향(패턴)"] = card["위험 특성"].fillna("").map(guidance)
    rw = pd.read_csv(Path(__file__).parent / "rewrites.csv")
    card = card.merge(rw, on="질문 ID", how="left")
    card = card.sort_values(["분류", "P(평균 미만)"], ascending=[True, False])
    card.to_csv(OUT / "question_scorecard.csv", index=False, encoding="utf-8-sig")
    write_xlsx(card, seg, coef)

    summary = {
        "base_answer_rate": base,
        "totals": {c: int(d[c].sum()) for c in ["exp", "ans", "rep", "nr"]},
        "n_questions": len(d),
        "exposure_hist": d.exp.value_counts().sort_index().to_dict(),
        "prior_concentration_k": k,
        "segments": seg.replace({np.nan: None}).to_dict("records"),
        "glm": coef.to_dict("records"),
        "class_counts": d["분류"].value_counts().to_dict(),
        "class_by_cat": pd.crosstab(d["cat"], d["분류"]).to_dict(),
        "risk_feature_counts": {k: int(d[k].sum()) for k in RISK_FEATURES},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    print(f"base={base:.3f} k={k:.1f}")
    print(d["분류"].value_counts())
    print(coef.round(3).to_string())


if __name__ == "__main__":
    main()
