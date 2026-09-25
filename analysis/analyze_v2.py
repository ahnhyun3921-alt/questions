"""v2 분석: 목표 변수 재정의 · AI 태깅 특성 · 교차검증 모델 비교 · FDR · 계층 베이지안 · 부트스트랩 · 기대 이득 우선순위.

사용법: python analysis/analyze_v2.py
입력:  nadab_daily_question_stats.csv, analysis/question_tags.csv, analysis/rewrites.csv
출력:  output/v2/*.csv, output/v2/summary.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent))
from analyze import COLS  # noqa: E402
from features import FEATURES, extract  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "v2"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260925)

TAG_LABEL = {
    "T": "시점 고정", "P": "경험 전제", "E": "인지 노력", "W": "감정 무게", "S": "자기노출", "C": "은유·모호",
}
F_LABEL = {"p": "선호(최애)형", "t": "성향·습관형", "c": "양자택일형", "r": "일화 회상형",
           "h": "상상형", "d": "개념·가치형", "n": "인물 지목형", "o": "기타"}


# ---------------------------------------------------------------- 데이터
def load():
    d = pd.read_csv(ROOT / "nadab_daily_question_stats.csv", encoding="utf-8-sig")
    d.columns = COLS
    d["q"] = d["q"].str.strip()
    tags = pd.read_csv(Path(__file__).parent / "question_tags.csv")
    d = d.merge(tags, on="id", how="left", validate="1:1")
    assert d[list("TPEWSFC")].notna().all().all(), "태그 누락"
    rx = pd.DataFrame([extract(q) for q in d.q])
    d = pd.concat([d, rx], axis=1)
    # 목표 변수: 결론이 난 노출(답변 or 교체) 중 답변 비율. 미응답(진행 중 포함)은 제외
    d["res"] = d.ans + d.rep
    return d


def design(d, kind):
    X = pd.get_dummies(d["cat"], prefix="관심사", dtype=float).drop(columns="관심사_가치관", errors="ignore")
    X["레벨2"] = (d.level == 2).astype(float)
    if kind in ("regex", "tags+regex"):
        for k in FEATURES:
            X["[규칙] " + k] = d[k].astype(float)
        X["[규칙] 길이(10자)"] = d["문구 길이"] / 10
    if kind in ("tags", "tags+regex"):
        for k in ["T", "P"]:
            X[f"{TAG_LABEL[k]}=1"] = (d[k] == 1).astype(float)
            X[f"{TAG_LABEL[k]}=2"] = (d[k] == 2).astype(float)
        for k in ["E", "W", "S"]:
            X[f"{TAG_LABEL[k]}(1~3)"] = d[k].astype(float) - 1
        X[TAG_LABEL["C"]] = d["C"].astype(float)
        for f, lab in F_LABEL.items():
            if f != "t":  # 기준: 성향·습관형
                X["형식: " + lab] = (d["F"] == f).astype(float)
    return X


# ---------------------------------------------------------------- 교차검증 모델 비교
def expand(X, a, r):
    """이항 카운트 → 가중 이진 행 (sklearn용)."""
    Xa = np.vstack([X, X])
    y = np.r_[np.ones(len(X)), np.zeros(len(X))]
    w = np.r_[a, r].astype(float)
    keep = w > 0
    return Xa[keep], y[keep], w[keep]


def binom_ll(p, a, r):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.sum(a * np.log(p) + r * np.log(1 - p)))


def cv_compare(d, k=10, reps=5):
    s = d[d.res > 0].reset_index(drop=True)
    a, r = s.ans.values, s.rep.values
    mats = {kind: design(s, kind).values for kind in ["base", "regex", "tags", "tags+regex"]}
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True)
    grid = [0.03, 0.1, 0.3, 1.0, 3.0]
    res = {m: {c: [] for c in grid} for m in list(mats) + ["text"]}
    null = []
    for rep in range(reps):
        folds = RNG.permutation(len(s)) % k
        for f in range(k):
            tr, te = folds != f, folds == f
            p0 = a[tr].sum() / (a[tr].sum() + r[tr].sum())
            null.append(binom_ll(np.full(te.sum(), p0), a[te], r[te]))
            Xt = tf.fit_transform(s.q[tr]); Xe = tf.transform(s.q[te])
            base_tr, base_te = mats["base"][tr], mats["base"][te]
            for m in list(mats) + ["text"]:
                if m == "text":
                    from scipy.sparse import hstack, csr_matrix
                    Atr = hstack([Xt, csr_matrix(base_tr)]).tocsr(); Ate = hstack([Xe, csr_matrix(base_te)]).tocsr()
                    Atr2 = __import__("scipy.sparse", fromlist=["vstack"]).vstack([Atr, Atr]).tocsr()
                    y = np.r_[np.ones(tr.sum()), np.zeros(tr.sum())]; w = np.r_[a[tr], r[tr]].astype(float)
                    kp = w > 0; Xtr_, ytr_, wtr_ = Atr2[kp], y[kp], w[kp]
                else:
                    Xtr_, ytr_, wtr_ = expand(mats[m][tr], a[tr], r[tr]); Ate = mats[m][te]
                for c in grid:
                    lr = LogisticRegression(C=c, max_iter=2000)
                    lr.fit(Xtr_, ytr_, sample_weight=wtr_)
                    res[m][c].append(binom_ll(lr.predict_proba(Ate)[:, 1], a[te], r[te]))
    n_total = (a.sum() + r.sum()) * reps
    ll0 = np.sum(null)
    rows = []
    for m, byc in res.items():
        best_c = max(byc, key=lambda c: np.sum(byc[c]))
        ll = np.sum(byc[best_c])
        rows.append({"model": m, "C": best_c, "cv_logloss": -ll / n_total,
                     "gain_vs_null_pct": (ll - ll0) / abs(ll0) * 100,
                     "pseudo_r2": 1 - ll / ll0})
    rows.append({"model": "null", "C": None, "cv_logloss": -ll0 / n_total, "gain_vs_null_pct": 0.0, "pseudo_r2": 0.0})
    return pd.DataFrame(rows).sort_values("cv_logloss")


# ---------------------------------------------------------------- 추론: GLM + FDR
def glm_effects(d, kind):
    s = d[d.res > 0]
    X = sm.add_constant(design(s, kind))
    m = sm.GLM(np.column_stack([s.ans, s.rep]), X, family=sm.families.Binomial()).fit()
    scale = max(1.0, m.pearson_chi2 / m.df_resid)  # 과산포면 SE 확대, 과소면 그대로(보수적)
    m = sm.GLM(np.column_stack([s.ans, s.rep]), X, family=sm.families.Binomial()).fit(scale=scale)
    ci = m.conf_int()
    t = pd.DataFrame({"term": m.params.index, "odds_ratio": np.exp(m.params), "or_lo": np.exp(ci[0]),
                      "or_hi": np.exp(ci[1]), "p": m.pvalues}).query("term != 'const'").reset_index(drop=True)
    t["q_fdr"] = multipletests(t.p, method="fdr_bh")[1]
    t["판정"] = np.select([t.q_fdr < 0.05, t.q_fdr < 0.2], ["유의(FDR<5%)", "경향(FDR<20%)"], "근거 부족")
    return m, t, scale


# ---------------------------------------------------------------- 계층 베이지안 (질문 랜덤효과)
def hier_bayes(d):
    s = d[d.res > 0].reset_index(drop=True)
    X = design(s, "tags")
    rows = np.repeat(np.arange(len(s)), s.res.values)
    y = np.concatenate([np.r_[np.ones(a), np.zeros(r)] for a, r in zip(s.ans, s.rep)])
    Xr = X.iloc[rows].reset_index(drop=True)
    ids = s.id.values[rows]
    exog = sm.add_constant(Xr).values
    Z = pd.get_dummies(pd.Series(ids).astype(str), dtype=float)
    model = BinomialBayesMixedGLM(y, exog, Z.values, ident=np.zeros(Z.shape[1], dtype=int),
                                  fe_p=2.0, vcp_p=1.0)
    fit = model.fit_vb()
    k = exog.shape[1]
    beta, beta_sd = fit.fe_mean, fit.fe_sd
    u = pd.Series(fit.vc_mean, index=Z.columns.astype(int))
    u_sd = pd.Series(fit.vc_sd, index=Z.columns.astype(int))
    tau = float(np.exp(fit.vcp_mean[0]))
    # 모든 질문(미노출 포함)에 대해 사후 로짓 = 고정효과 + 랜덤효과(없으면 0, sd=tau)
    Xall = sm.add_constant(design(d, "tags"), has_constant="add").values
    eta = Xall @ beta
    ui = d.id.map(u).fillna(0.0).values
    usd = d.id.map(u_sd).fillna(tau).values
    fe_sd = np.sqrt(np.einsum("ij,j,ij->i", Xall, beta_sd ** 2, Xall))
    sd = np.sqrt(usd ** 2 + fe_sd ** 2)
    return eta + ui, sd, tau


# ---------------------------------------------------------------- 부트스트랩: 수정안 효과 신뢰구간
def bootstrap_impact(d, B=400):
    rw = pd.read_csv(Path(__file__).parent / "rewrites.csv").fillna("")
    rw = rw[~rw["수정안 1 (권장)"].str.startswith("(")]
    new_text = rw.set_index("질문 ID")["수정안 1 (권장)"]
    tgt = d[d.id.isin(new_text.index)].copy()
    new = tgt.copy()
    nf = pd.DataFrame([extract(q) for q in new_text.loc[tgt.id]], index=tgt.index)
    for c in nf:
        new[c] = nf[c]
    s = d[d.res > 0]
    ids = s.id.values
    tot_res = s.res.sum()
    gains, rel = [], []
    Xo, Xn = sm.add_constant(design(tgt, "regex"), has_constant="add"), sm.add_constant(design(new, "regex"), has_constant="add")
    for _ in range(B):
        pick = RNG.choice(len(s), len(s), replace=True)
        bs = s.iloc[pick]
        Xb = sm.add_constant(design(bs, "regex"), has_constant="add")
        try:
            m = sm.GLM(np.column_stack([bs.ans, bs.rep]), Xb, family=sm.families.Binomial()).fit()
        except Exception:
            continue
        cols = Xb.columns
        po = m.predict(Xo.reindex(columns=cols, fill_value=0)); pn = m.predict(Xn.reindex(columns=cols, fill_value=0))
        w = tgt.res.values
        gains.append(float(np.sum(w * (pn - po)) / tot_res * 100))
        rel.append(float(np.mean(pn - po) * 100))
    g, r = np.array(gains), np.array(rel)
    return {"overall_pp_mean": float(g.mean()), "overall_pp_ci": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
            "per_question_pp_mean": float(r.mean()), "per_question_pp_ci": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
            "n_rewrites": int(len(tgt)), "B": int(len(g)),
            "note": "목표 변수는 결론 난 노출 중 답변 비율. 질문 단위 부트스트랩, 문구 규칙 모델 기준"}


# ---------------------------------------------------------------- 중복·유사 질문, 커버리지
def near_duplicates(d, thr=0.75):
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit(d.q)
    M = cosine_similarity(tf.transform(d.q))
    np.fill_diagonal(M, 0)
    i, j = np.where(np.triu(M) >= thr)
    rows = [{"id_a": int(d.id[a]), "id_b": int(d.id[b]), "sim": round(float(M[a, b]), 3),
             "q_a": d.q[a], "q_b": d.q[b], "exact": d.q[a] == d.q[b]} for a, b in zip(i, j)]
    return pd.DataFrame(rows).sort_values("sim", ascending=False)


def coverage(d):
    s = d.copy()
    s["형식"] = s.F.map(F_LABEL)
    g = s.groupby(["cat", "형식"]).agg(질문수=("id", "size"), 결론노출=("res", "sum"), 답변=("ans", "sum")).reset_index()
    g["답변율(결론 기준)"] = g["답변"] / g["결론노출"].replace(0, np.nan)
    f = s.groupby("형식").agg(질문수=("id", "size"), 결론노출=("res", "sum"), 답변=("ans", "sum"))
    f["답변율(결론 기준)"] = f["답변"] / f["결론노출"]
    f["비중(%)"] = f["질문수"] / len(s) * 100
    lo, hi = zip(*[proportion_ci(a, n) for a, n in zip(f["답변"], f["결론노출"])])
    f["ci_lo"], f["ci_hi"] = lo, hi
    return g, f.reset_index().sort_values("답변율(결론 기준)", ascending=False)


def proportion_ci(a, n):
    from statsmodels.stats.proportion import proportion_confint
    return proportion_confint(a, n, method="wilson") if n else (np.nan, np.nan)


# ---------------------------------------------------------------- main
def main():
    d = load()
    base_res = d.ans.sum() / d.res.sum()
    out = {"base_resolved_rate": float(base_res), "base_exposure_rate": float(d.ans.sum() / d.exp.sum()),
           "no_response_share": float(d.nr.sum() / d.exp.sum())}

    cv = cv_compare(d)
    cv.to_csv(OUT / "cv_models.csv", index=False, encoding="utf-8-sig")
    out["cv"] = cv.to_dict("records")

    effects = {}
    for kind in ["regex", "tags"]:
        _, t, sc = glm_effects(d, kind)
        t.to_csv(OUT / f"effects_{kind}.csv", index=False, encoding="utf-8-sig")
        effects[kind] = t
        out[f"effects_{kind}"] = t.to_dict("records")
        out[f"scale_{kind}"] = sc

    eta, sd, tau = hier_bayes(d)
    logit_base = np.log(base_res / (1 - base_res))
    d["사후 답변율"] = 1 / (1 + np.exp(-eta))
    d["사후 하한(10%)"] = 1 / (1 + np.exp(-(eta - 1.2816 * sd)))
    d["사후 상한(90%)"] = 1 / (1 + np.exp(-(eta + 1.2816 * sd)))
    d["P(평균 미만)"] = stats.norm.cdf((logit_base - eta) / sd)
    out["tau_question_sd_logit"] = tau

    # 기대 이득 우선순위: 다음 달 예상 노출 × (수정안 예측 - 현재 사후) , 수정안 예측은 규칙 모델 로짓 차이를 사후 로짓에 더함
    sc = pd.read_csv(ROOT / "output" / "rewrites_scored.csv") if (ROOT / "output" / "rewrites_scored.csv").exists() else None
    cat_exp = d.groupby("cat").exp.mean()
    d["예상 노출(월)"] = np.where(d.exp > 0, d.exp, d.cat.map(cat_exp))
    d["수정 후 예측"] = np.nan
    if sc is not None:
        lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        dl = (lg(sc["예측(수정안)"]) - lg(sc["예측(기존 문구)"])).values
        m = pd.Series(dl, index=sc["질문 ID"])
        delta = d.id.map(m)
        d.loc[delta.notna(), "수정 후 예측"] = 1 / (1 + np.exp(-(eta[delta.notna()] + delta[delta.notna()])))
    d["기대 추가 답변(월)"] = (d["예상 노출(월)"] * (1 - d.nr.sum() / d.exp.sum())
                           * (d["수정 후 예측"] - d["사후 답변율"])).clip(lower=0)
    pri = d[d["수정 후 예측"].notna()].sort_values("기대 추가 답변(월)", ascending=False)
    pri["우선순위"] = np.arange(1, len(pri) + 1)
    pri[["우선순위", "id", "q", "cat", "exp", "ans", "rep", "사후 답변율", "P(평균 미만)", "수정 후 예측",
         "예상 노출(월)", "기대 추가 답변(월)"]].to_csv(OUT / "priority.csv", index=False, encoding="utf-8-sig")
    top = pri.head(50)
    out["priority_top50_share"] = float(top["기대 추가 답변(월)"].sum() / pri["기대 추가 답변(월)"].sum())
    out["priority_total_gain"] = float(pri["기대 추가 답변(월)"].sum())
    out["priority_top10"] = top.head(10)[["id", "q", "기대 추가 답변(월)"]].to_dict("records")

    # v1(경험적 베이즈)과 순위 비교
    v1 = pd.read_csv(ROOT / "output" / "question_scorecard.csv")[["질문 ID", "P(평균 미만)"]].rename(
        columns={"질문 ID": "id", "P(평균 미만)": "v1"})
    cmp_ = d[["id", "P(평균 미만)", "exp"]].merge(v1, on="id")
    cmp_ = cmp_[cmp_.exp > 0]
    out["rank_corr_v1_v2"] = float(stats.spearmanr(cmp_["P(평균 미만)"], cmp_.v1).correlation)

    d[["id", "q", "cat", "level", "exp", "ans", "rep", "nr", "T", "P", "E", "W", "S", "F", "C",
       "사후 답변율", "사후 하한(10%)", "사후 상한(90%)", "P(평균 미만)"]].to_csv(OUT / "question_posterior.csv", index=False, encoding="utf-8-sig")

    out["impact"] = bootstrap_impact(d)

    dup = near_duplicates(d)
    dup.to_csv(OUT / "near_duplicates.csv", index=False, encoding="utf-8-sig")
    out["duplicates"] = dup.head(20).to_dict("records")
    out["n_exact_dup"] = int(dup.exact.sum())
    out["n_near_dup"] = int(len(dup))

    cov, fmt = coverage(d)
    cov.to_csv(OUT / "coverage_cat_format.csv", index=False, encoding="utf-8-sig")
    fmt.to_csv(OUT / "coverage_format.csv", index=False, encoding="utf-8-sig")
    out["format"] = fmt.replace({np.nan: None}).to_dict("records")

    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(cv.round(4).to_string(index=False))
    print(effects["tags"][["term", "odds_ratio", "p", "q_fdr", "판정"]].round(3).to_string(index=False))
    print(effects["regex"][["term", "odds_ratio", "p", "q_fdr", "판정"]].round(3).to_string(index=False))
    print(json.dumps({k: out[k] for k in ["base_resolved_rate", "no_response_share", "tau_question_sd_logit",
                                          "rank_corr_v1_v2", "impact", "n_exact_dup", "n_near_dup",
                                          "priority_top50_share", "priority_total_gain"]}, ensure_ascii=False, indent=1))
    print(fmt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
