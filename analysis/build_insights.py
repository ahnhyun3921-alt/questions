"""'나답 질문 개선함' 분석 탭용 인사이트(board/data/insights.json)를 만든다.

사용법: python analysis/build_insights.py   (build_board_data.py 다음에, 매달 8단계에서)
입력: 최신 통계(output/board_stats/<날짜>.json), 질문 태그(analysis/question_tags.csv),
      원본 DB, 개선본(board/data/questions.json)
결과(모두 '현재 Revision' 실적 기준, 결론 = 답변 + 교체):
  - uni: 요인별 단변량 비교(해당/비해당 답변율 차이와 95% CI)
  - model: 다변량 로지스틱 회귀(문항 단위 군집 표준오차) — 오즈비와 평균 답변율에서의 %p 환산
  - length: 글자 수 구간별 답변율(Wilson 95% CI)
  - axes: 자아분석 축별 문항 수(지금/개선 후)와 답변율
  - findings: 위 결과에서 뽑은 핵심 인사이트 문장
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from db_questions import latest_db, stats_to_db  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Z = 1.959964
FORM = {"p": "선호(최애)", "c": "양자택일", "t": "성향·습관", "h": "상상·가정", "r": "일화 회상", "d": "개념·가치", "n": "특정 인물", "o": "기타"}
CAT = {1: "취향", 2: "감정", 3: "루틴", 4: "관계", 5: "사랑", 6: "가치관"}
FACTORS = [  # (키, 이름, 설명, 조건)
    ("T", "시점 고정", "'오늘·이번 주'처럼 특정 날에 묶임", lambda d: d["T"] > 0),
    ("P", "경험 전제", "'~한 적 있다면'처럼 특정 경험이 있어야 답함", lambda d: d["P"] > 0),
    ("E", "즉답 가능", "생각할 필요 없이 바로 답할 수 있음", lambda d: d["E"] == 1),
    ("W", "감정 무게", "감정적으로 보통 이상 무거움", lambda d: d["W"] >= 2),
    ("S", "자기노출", "자기 이야기를 보통 이상 털어놓아야 함", lambda d: d["S"] >= 2),
    ("C", "은유·모호", "'감정의 온도'처럼 은유적이거나 모호함", lambda d: d["C"] == 1),
    ("L", "40자 초과", "문구가 40자를 넘음", lambda d: d["L"] > 40),
]


def wilson(a, n):
    if n == 0:
        return [None, None]
    p = a / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def load():
    snap = sorted((ROOT / "output" / "board_stats").glob("20*.json"))[-1]
    st = json.loads(snap.read_text())
    tags = pd.read_csv(Path(__file__).parent / "question_tags.csv").set_index("id")
    db = latest_db().astype({"id": str}).set_index("id")
    axis = pd.read_csv(Path(__file__).parent / "axis_tags.csv").set_index("질문 ID")
    rows = []
    for sid, t in tags.iterrows():
        k = str(stats_to_db(int(sid)))
        dl = db.loc[k, "deleted_at"] if k in db.index else None
        if k not in st["q"] or k not in db.index or (pd.notna(dl) and str(dl).strip()):
            continue
        v = st["q"][k]
        rows.append(dict(key=k, ans=v[1], rep=v[2], exp=v[0], L=len(str(db.loc[k, "question_text"])), level=int(db.loc[k, "question_level"]),
                         cat=CAT.get(int(db.loc[k, "interest_id"]), ""), axis=axis.loc[sid, "분석 축"] if sid in axis.index else "",
                         **{c: t[c] for c in "TPEWSFC"}))
    return snap.stem, pd.DataFrame(rows)


def univariate(d):
    out = []
    def one(key, name, desc, mask, group):
        a1, r1 = d[mask].ans.sum(), d[mask].rep.sum()
        a0, r0 = d[~mask].ans.sum(), d[~mask].rep.sum()
        if a1 + r1 == 0 or a0 + r0 == 0:
            return
        p1, p0 = a1 / (a1 + r1), a0 / (a0 + r0)
        se = np.sqrt(p1 * (1 - p1) / (a1 + r1) + p0 * (1 - p0) / (a0 + r0))
        out.append(dict(key=key, name=name, desc=desc, group=group, nq=int(mask.sum()), res=int(a1 + r1),
                        rate=round(p1, 4), rest=round(p0, 4), diff=round(p1 - p0, 4),
                        lo=round(p1 - p0 - Z * se, 4), hi=round(p1 - p0 + Z * se, 4)))
    for key, name, desc, f in FACTORS:
        one(key, name, desc, f(d), "문구 특성")
    for f, name in FORM.items():
        one("F" + f, name, "질문 형식", d["F"] == f, "질문 형식")
    for c in CAT.values():
        one("K" + c, c, "관심사", d["cat"] == c, "관심사")
    return out


# 평가 모델의 특성 정의 — 페이지(JS)의 modelPred도 같은 이름·조건을 쓴다
SPEC = [  # (이름, 무리, 조건)
    ("시점 고정", "문구 특성", lambda d: d["T"] > 0), ("특정 날 시점", "문구 특성", lambda d: d["T"] == 2),
    ("경험 전제", "문구 특성", lambda d: d["P"] > 0), ("특정 경험 전제", "문구 특성", lambda d: d["P"] == 2),
    ("즉답 가능", "문구 특성", lambda d: d["E"] == 1), ("깊은 성찰", "문구 특성", lambda d: d["E"] == 3),
    ("감정 무게", "문구 특성", lambda d: d["W"] >= 2), ("무거운 감정", "문구 특성", lambda d: d["W"] == 3),
    ("자기노출", "문구 특성", lambda d: d["S"] >= 2), ("높은 자기노출", "문구 특성", lambda d: d["S"] == 3),
    ("은유·모호", "문구 특성", lambda d: d["C"] == 1), ("40자 초과", "문구 특성", lambda d: d["L"] > 40),
    ("레벨 1", "노출 맥락", lambda d: d["level"] == 1),
] + [("형식: " + n, "질문 형식", (lambda f: lambda d: d["F"] == f)(f)) for f, n in FORM.items() if f != "t"] \
  + [("관심사: " + c, "관심사", (lambda c: lambda d: d["cat"] == c)(c)) for c in CAT.values() if c != "가치관"]


def design(d, mean_logl=None):
    X = pd.DataFrame({n: f(d).astype(float) for n, _, f in SPEC})
    ll = np.log(d["L"].astype(float))
    mean_logl = float(ll.mean()) if mean_logl is None else mean_logl
    X["문구 길이"] = ll - mean_logl     # 로그 글자 수(가운데 맞춤) — 효과는 '두 배 길어질 때'로 보고
    return X, mean_logl


def model(d):
    """릿지 규제 베타-이항 회귀(analysis/qmodel.py). λ는 문항 단위 5겹 교차검증으로 고른다.
    효과 구간은 문항 부트스트랩(200회) 백분위, p는 부호가 뒤집힌 비율의 두 배(양측)."""
    from qmodel import BetaBinomialRidge, kfold_eval, bb_loglik  # noqa: F401
    from scipy.stats import spearmanr
    d = d[(d.ans + d.rep) > 0].reset_index(drop=True)
    a, n = d.ans.values.astype(float), (d.ans + d.rep).values.astype(float)
    X, mean_logl = design(d)
    names = list(X.columns)
    Xv = X.values
    # λ 고르기
    cv = {lam: np.mean([kfold_eval(Xv, a, n, lambda: BetaBinomialRidge(lam), k=5, seed=s)["ll"] for s in range(2)]) for lam in [3, 10, 30]}
    lam = max(cv, key=cv.get)
    fit = BetaBinomialRidge(lam).fit(Xv, a, n)
    base = a.sum() / n.sum()
    lg = np.log(base / (1 - base))
    to_p = lambda x: float(1 / (1 + np.exp(-(lg + x))) - base)
    rng = np.random.default_rng(7)
    boots = []
    for _ in range(200):
        ix = rng.integers(0, len(a), len(a))
        boots.append(BetaBinomialRidge(lam).fit(Xv[ix], a[ix], n[ix]).coef_[1:])
    boots = np.array(boots)
    group = {nm: g for nm, g, _ in SPEC}
    group["문구 길이"] = "문구 특성"
    terms = []
    for j, nm in enumerate(names):
        scale = np.log(2) if nm == "문구 길이" else 1.0      # 길이는 '두 배'당 효과
        b, bs = fit.coef_[1 + j] * scale, boots[:, j] * scale
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = min(1.0, 2 * min((bs > 0).mean(), (bs < 0).mean()))
        terms.append(dict(name=nm.split(": ")[-1] + (" (2배)" if nm == "문구 길이" else ""), key=nm, group=group[nm], beta=round(float(b), 4),
                          or_=round(float(np.exp(b)), 3), or_lo=round(float(np.exp(lo)), 3), or_hi=round(float(np.exp(hi)), 3), p=round(float(p), 4),
                          pp=round(to_p(b), 4), pp_lo=round(to_p(lo), 4), pp_hi=round(to_p(hi), 4)))
    # 모델 평가: 기준 모델과 교차검증 로그우도·순위 상관·보정
    def cvll(cols):
        Xc = X[cols].values if cols else np.zeros((len(a), 0))
        return float(np.mean([kfold_eval(Xc, a, n, lambda: BetaBinomialRidge(lam), k=5, seed=s)["ll"] for s in range(2)]))
    old = [c for c in names if c in ("시점 고정", "경험 전제", "즉답 가능", "감정 무게", "자기노출", "은유·모호", "40자 초과") or c.startswith(("형식: ", "관심사: "))]
    m0 = cvll([])
    evals = [dict(name="평균만", ll=0.0), dict(name="관심사만", ll=round(cvll([c for c in names if c.startswith("관심사: ")]) - m0, 1)),
             dict(name="이전 모델 특성", ll=round(cvll(old) - m0, 1)), dict(name="지금 모델", ll=round(float(cv[lam]) - m0, 1))]
    ev_new = kfold_eval(Xv, a, n, lambda: BetaBinomialRidge(lam), k=5, seed=0)
    ev_old = kfold_eval(X[old].values, a, n, lambda: BetaBinomialRidge(lam), k=5, seed=0)
    big = n >= 6
    rho_new = float(spearmanr(ev_new["preds"][big], a[big] / n[big]).statistic)
    rho_old = float(spearmanr(ev_old["preds"][big], a[big] / n[big]).statistic)
    q = pd.qcut(ev_new["preds"], 5, labels=False)
    calib = [dict(pred=round(float(np.average(ev_new["preds"][q == b], weights=n[q == b])), 4), obs=round(float(a[q == b].sum() / n[q == b].sum()), 4),
                  n=int(n[q == b].sum())) for b in range(5)]
    # 문항별 사후: 특성 예측 μ 쪽으로 φ만큼 당긴 추정과, 평균±10%p 밖일 확률
    mu = fit.mu(Xv)
    pb = fit.prob_below(Xv, a, n, base - 0.10)
    pa = 1 - fit.prob_below(Xv, a, n, base + 0.10)
    post = {k: [round(float(m_), 4), round(float(pb_), 3), round(float(pa_), 3)] for k, m_, pb_, pa_ in zip(d.key, mu, pb, pa)}
    coef = {"const": round(float(fit.coef_[0]), 5), **{nm: round(float(fit.coef_[1 + j]), 5) for j, nm in enumerate(names)}}
    return dict(kind="betabinomial-ridge", coef=coef, phi=round(fit.phi_, 3), lam=lam, mean_logl=round(mean_logl, 5),
                n_q=int(len(a)), n_res=int(n.sum()), base=round(float(base), 4), ref="형식은 성향·습관, 관심사는 가치관 대비",
                terms=terms, evals=evals, rho=dict(new=round(rho_new, 3), old=round(rho_old, 3), n=int(big.sum())), calib=calib, post=post,
                policy=dict(fix_p=0.8, good_p=0.8, explore_n=20, gap=0.10), scale=1.0)


def length_bins(d):
    bins = [(0, 20, "20자 이하"), (21, 30, "21~30자"), (31, 40, "31~40자"), (41, 50, "41~50자"), (51, 999, "51자 이상")]
    out = []
    for lo, hi, name in bins:
        m = (d.L >= lo) & (d.L <= hi)
        a, r = int(d[m].ans.sum()), int(d[m].rep.sum())
        out.append(dict(name=name, nq=int(m.sum()), res=a + r, rate=round(a / (a + r), 4) if a + r else None, ci=wilson(a, a + r)))
    return out


def axes(d):
    rows = json.loads((ROOT / "board" / "data" / "questions.json").read_text())["rows"]
    names = sorted({r["axis"] for r in rows if r["axis"]})
    out = []
    for ax in names:
        now = sum(1 for r in rows if r["axis"] == ax and r["orig"] and not r["orig"][6])
        aft = sum(1 for r in rows if r["axis"] == ax and not r["deleted"])
        m = d.axis == ax
        a, rr = int(d[m].ans.sum()), int(d[m].rep.sum())
        out.append(dict(name=ax, now=now, aft=aft, res=a + rr, rate=round(a / (a + rr), 4) if a + rr else None, ci=wilson(a, a + rr)))
    return sorted(out, key=lambda x: -x["aft"])


ACTION = {
    "경험 전제": "'~한 적 있다면'을 빼고 누구나 답할 수 있는 상황으로 바꿔요.",
    "특정 경험 전제": "특정 경험(연애 중, 운동 등)이 있어야 답할 수 있는 질문은 '없다면 ~' 대안을 붙이거나 일반 상황으로 바꿔요.",
    "특정 날 시점": "'오늘·이번 주'처럼 특정 날에 묶지 말고 '요즘'이나 시점 없는 문장으로 바꿔요.",
    "레벨 1": "첫 질문(레벨 1) 자리는 답변이 잘 나와요. 가볍고 즉답 가능한 질문을 이 자리에 둬요.",
    "문구 길이 (2배)": "문구가 길수록 답변율이 떨어져요. 군더더기를 덜어 30자 안팎으로 줄여요.",
    "시점 고정": "'오늘·이번 주'를 '요즘'이나 시점 없는 문장으로 바꿔요.",
    "은유·모호": "은유를 걷어내고 구체적인 대상을 물어요.",
    "즉답 가능": "첫 질문과 신규 질문은 바로 답할 수 있는 형태를 기본으로 해요.",
    "기타": "예/아니오·청유형은 선호나 양자택일로 바꿔요.",
    "일화 회상": "기억을 꺼내야 하는 질문은 선호·성향형으로 바꾸거나 '없다면' 대안을 붙여요.",
    "관계": "관계 질문은 특정 인물 지목·경험 전제가 많아요. 가벼운 양자택일로 입구를 낮춰요.",
    "루틴": "루틴 질문은 '매일 하는 것' 전제가 많아요. 성향형('~하는 편인가요')으로 바꿔요.",
}


def findings(uni, mdl, ln, ax):
    """결과에서 의미 있는 것만 문장으로 만든다. '확실'은 95% 신뢰구간이 0(오즈비 1)을 넘지 않는 경우만."""
    terms = mdl["terms"]
    sig = [x for x in terms if x["p"] < 0.05]
    neg = sorted([x for x in sig if x["pp"] < 0], key=lambda x: x["pp"])
    pos = sorted([x for x in sig if x["pp"] > 0], key=lambda x: -x["pp"])
    def num(x):   # -0 같은 표기를 막는다
        v = round(x * 100)
        return "0" if v == 0 else f"{v:+d}"
    pct = lambda x: f"{num(x)}%p"
    ex = lambda x: f"'{x['name']}'({pct(x['pp'])}, 95% CI {num(x['pp_lo'])}~{num(x['pp_hi'])}%p)"
    out = []
    if neg:
        out.append(dict(kind="neg", title="답변율을 확실히 떨어뜨리는 요인",
                        body="다른 요인을 함께 고려해도 " + ", ".join(ex(x) for x in neg[:4]) + "은 답변율을 낮춰요.",
                        action=" ".join(ACTION[x["name"]] for x in neg[:3] if x["name"] in ACTION)))
    if pos:
        out.append(dict(kind="pos", title="답변율을 확실히 올리는 요인",
                        body=", ".join(ex(x) for x in pos[:3]) + "가 있는 질문은 답변율이 높아요.",
                        action=" ".join(ACTION[x["name"]] for x in pos[:3] if x["name"] in ACTION) or "이 특성을 신규 질문의 기본값으로 둬요."))
    # 단독으로는 크게 차이 나지만 다른 요인을 넣으면 사라지는 것 = 다른 특성과 겹쳐 있음
    u = {x["name"]: x for x in uni if x["group"] == "문구 특성"}
    tk = {x["name"]: x for x in terms}
    # 같은 특성의 '강한 단계'(예: 특정 경험 전제)가 효과를 가져간 경우는 겹침이 아니라 강도 문제로 따로 설명한다
    child = {"경험 전제": "특정 경험 전제", "시점 고정": "특정 날 시점", "즉답 가능": "깊은 성찰", "감정 무게": "무거운 감정", "자기노출": "높은 자기노출"}
    strength = [(par, ch) for par, ch in child.items() if ch in tk and tk[ch]["p"] < 0.05 and par in tk and tk[par]["p"] >= 0.05]
    for par, ch in strength:
        out.append(dict(kind="neg", title="전제의 강도가 갈라요",
                        body=f"'{par}'는 가벼운 수준(대부분이 답할 수 있는 정도)이면 {pct(tk[par]['pp'])}로 거의 차이가 없지만, "
                             f"'{ch}'(특정 경험이 있어야 답함)이면 {pct(tk[par]['pp'] + tk[ch]['pp'])}까지 떨어져요.",
                        action=ACTION.get(ch, "")))
    skip = {par for par, _ in strength}
    conf = [x for x in terms if x["group"] == "문구 특성" and x["p"] >= 0.05 and x["name"] not in skip and x["name"] in u and (u[x["name"]]["lo"] > 0 or u[x["name"]]["hi"] < 0)]
    if conf:
        out.append(dict(kind="neutral", title="겹쳐 있어서 커 보였던 요인",
                        body=", ".join(f"'{x['name']}'(단독 {pct(u[x['name']]['diff'])} → 함께 보면 {pct(x['pp'])}, 불확실)" for x in conf)
                             + "은 따로 보면 답변율이 확실히 다르지만, 경험 전제·형식 같은 다른 특성과 함께 붙어 다녀서 생긴 차이가 커요.",
                        action="문구를 고칠 때는 이 요인만 지우기보다, 같이 붙어 있는 경험 전제·형식을 함께 바꿔야 효과가 나요."))
    null = [x for x in terms if x["group"] == "문구 특성" and x["p"] >= 0.05 and x not in conf and x["name"] not in skip]
    if null:
        out.append(dict(kind="neutral", title="영향이 뚜렷하지 않은 요인",
                        body=", ".join(f"'{x['name']}'" for x in null) + "는 따로 봐도, 함께 봐도 답변율과 뚜렷한 관계가 없어요.",
                        action="이 특성 자체를 피할 필요는 없어요. 질문의 깊이를 지키는 데 써도 돼요."))
    good = [b for b in ln if b["rate"] is not None and b["res"] >= 50]
    if good:
        best, worst = max(good, key=lambda b: b["rate"]), min(good, key=lambda b: b["rate"])
        sure = best["ci"][0] > worst["ci"][1]
        out.append(dict(kind="neutral", title="문구 길이",
                        body=f"{best['name']} 답변율 {best['rate']:.0%}, {worst['name']} {worst['rate']:.0%}예요. "
                             + ("짧은 질문이 확실히 유리해요." if sure else "신뢰구간이 겹쳐서 길이만으로는 차이가 확실하지 않아요.")
                             + (lambda t: f" 다른 특성을 함께 고려해도 문구가 두 배 길어지면 답변율이 {pct(t['pp'])}(95% CI {num(t['pp_lo'])}~{num(t['pp_hi'])}%p) 달라져요." if t else "")(next((x for x in terms if x['name'] == '문구 길이 (2배)'), None)),
                        action="40자라는 경계보다 '짧을수록 낫다'가 맞아요. 군더더기를 덜어 30자 안팎으로 줄여요."))
    small = sorted(ax, key=lambda a: a["aft"])[:3]
    out.append(dict(kind="gap", title="자아분석 축 커버리지",
                    body="개선 후 가장 얇은 축은 " + ", ".join(f"{a['name']}({a['aft']}문항)" for a in small)
                         + f"이에요. 가장 두꺼운 {ax[0]['name']}({ax[0]['aft']}문항)의 1/3 수준이에요.",
                    action="다음 신규 질문은 얇은 축부터 채워야 자아분석 결과가 한쪽으로 치우치지 않아요."))
    lowax = [a for a in ax if a["rate"] is not None and a["res"] >= 80]
    if lowax:
        lo, hi = min(lowax, key=lambda a: a["rate"]), max(lowax, key=lambda a: a["rate"])
        out.append(dict(kind="neg", title="답이 잘 안 모이는 축",
                        body=f"{lo['name']} 축 답변율이 {lo['rate']:.0%}(95% CI {lo['ci'][0]:.0%}~{lo['ci'][1]:.0%})로 가장 낮고, "
                             f"가장 높은 축은 {hi['name']}({hi['rate']:.0%})예요.",
                        action=f"{lo['name']}처럼 감정을 직접 묻는 축은 '나는 ~하는 편인가요' 같은 성향·양자택일형으로 같은 신호를 받아요."))
    return out


LEVELS = [
    (1, "입문", "가볍고 바로 답하는 질문 — 선호·양자택일·상상, 시점·경험 전제·은유 없음", "처음 기록하는 날, 오랜만에 돌아온 날"),
    (2, "일상", "평소의 나를 관찰하는 질문 — 성향·습관, 가벼운 생각", "기록이 습관이 되는 시기의 기본"),
    (3, "성찰", "생각을 정리하거나 기억을 꺼내는 질문 — 개념·가치, 일화 회상, 특정 경험, 감정·자기노출 보통 이상", "기록이 쌓인 사용자에게 섞어서"),
    (4, "깊이", "무거운 감정이나 깊은 자기노출을 다루는 질문", "충분히 쌓인 뒤, 주 1회 이하 · 연속 금지"),
]
JOURNEY = [  # 기록 횟수 구간 → 레벨 비율(%)
    ("1~3회째", {1: 100}), ("4~14회째", {1: 40, 2: 60}), ("15~30회째", {1: 20, 2: 50, 3: 30}), ("31회째부터", {2: 45, 3: 40, 4: 15}),
]


def level_of(t, cur):
    """제안 레벨. 원래 레벨 1(첫 질문)은 그대로 1로 둔다."""
    if cur == 1:
        return 1
    if t["W"] == 3 or t["S"] == 3:
        return 4
    if t["E"] == 3 or (t["W"] >= 2 and t["S"] >= 2) or t["F"] in ("r", "d") or t["P"] == 2:
        return 3
    if t["T"] == 0 and t["P"] == 0 and t["E"] == 1 and t["W"] == 1 and t["S"] <= 1 and t["C"] == 0 and t["F"] in ("p", "c", "h"):
        return 1
    return 2


def levels(d, mdl):
    """4단계 레벨 구조 제안: 개선본 활성 질문마다 태그로 레벨을 매기고, 레벨별 실적·분포·여정별 기대 답변율을 낸다."""
    rows = json.loads((ROOT / "board" / "data" / "questions.json").read_text())["rows"]
    tags = pd.read_csv(Path(__file__).parent / "question_tags.csv").set_index("id")
    T = {str(stats_to_db(int(i))): r.to_dict() for i, r in tags.iterrows()}
    for f in ["new_questions.csv", "new_questions_scraped.csv", "new_questions_love.csv", "new_questions_fun.csv", "new_questions_self.csv"]:
        n = pd.read_csv(ROOT / "output" / f).set_index("신규 ID")
        for r in rows:
            if r["newId"] in n.index:
                T[r["key"]] = n.loc[r["newId"], list("TPEWSFC")].to_dict()
    stq = {k: (a, rp) for k, a, rp in zip(d.key, d.ans, d.rep)}
    post = mdl.get("post", {})
    live = [r for r in rows if not r["deleted"] and r["key"] in T]
    lv = {r["key"]: level_of(T[r["key"]], r["level"]) for r in live}
    out = []
    for L, name, desc, when in LEVELS:
        ks = [k for k, v in lv.items() if v == L]
        a = sum(stq[k][0] for k in ks if k in stq)
        n = sum(stq[k][0] + stq[k][1] for k in ks if k in stq)
        mus = [post[k][0] for k in ks if k in post]
        out.append(dict(level=L, name=name, desc=desc, when=when, n=len(ks), res=int(n), rate=round(a / n, 4) if n else None, ci=wilson(a, n),
                        mu=round(float(np.mean(mus)), 4) if mus else None,
                        from_cur={c: sum(1 for r in live if lv[r["key"]] == L and r["level"] == c) for c in (1, 2)},
                        by_cat={c: sum(1 for r in live if lv[r["key"]] == L and r["cat"] == c) for c in CAT.values()}))
    rate = {x["level"]: x["rate"] for x in out}
    journey = [dict(stage=s_, mix=mix, expect=round(sum(rate[l] * w for l, w in mix.items()) / 100, 4)) for s_, mix in JOURNEY]
    return dict(levels=out, journey=journey, n=len(live), lv=lv, untagged=sum(1 for r in rows if not r["deleted"]) - len(live))


def main():
    date, d = load()
    uni, mdl, ln, ax = univariate(d), model(d), length_bins(d), axes(d)
    out = dict(snapshot=date, uni=uni, model=mdl, length=ln, axes=ax, levels=levels(d, mdl), findings=findings(uni, mdl, ln, ax))
    f = ROOT / "board" / "data" / "insights.json"
    f.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    print(f"{date}: 문항 {mdl['n_q']} · 결론 {mdl['n_res']} · 평균 {mdl['base']:.1%} · 인사이트 {len(out['findings'])}개 → {f}")
    for x in mdl["terms"]:
        print(f"  {x['group']:6s} {x['name']:8s} OR {x['or_']:.2f} [{x['or_lo']:.2f},{x['or_hi']:.2f}] p={x['p']:.3f} ≈ {x['pp'] * 100:+.1f}%p")


if __name__ == "__main__":
    main()
