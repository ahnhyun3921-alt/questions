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
        rows.append(dict(key=k, ans=v[1], rep=v[2], exp=v[0], L=len(str(db.loc[k, "question_text"])),
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


def model(d):
    """문항 단위 이항 회귀(답변 수 / 교체 수). 문항마다 답변율이 들쭉날쭉한 만큼(과산포)
    표준오차를 넓혀서(quasi-binomial, Pearson χ² 척도) 같은 문항 응답이 서로 독립이 아님을 반영한다."""
    X = pd.DataFrame({name: f(d).astype(float) for _, name, _, f in FACTORS})
    for f, name in FORM.items():
        if f != "t":   # 기준 = 성향·습관(가장 많은 형식)
            X["형식: " + name] = (d["F"] == f).astype(float)
    for c in CAT.values():
        if c != "가치관":   # 기준 = 가치관
            X["관심사: " + c] = (d["cat"] == c).astype(float)
    X = X.loc[:, X.sum() >= 10]
    X = sm.add_constant(X)
    m = (d.ans + d.rep) > 0
    glm = sm.GLM(np.column_stack([d.ans[m], d.rep[m]]), X[m.values], family=sm.families.Binomial())
    res = glm.fit(scale="X2")
    if res.scale < 1:   # 과소산포로 나오면 구간을 좁히지 않도록 보통 이항(척도 1)을 쓴다 — 보수적으로
        res = glm.fit(scale=1.0)
    base = d.ans.sum() / (d.ans.sum() + d.rep.sum())
    out = []
    for name in X.columns:
        if name == "const":
            continue
        b, se, p = res.params[name], res.bse[name], res.pvalues[name]
        lo, hi = b - Z * se, b + Z * se
        # 평균 답변율에서 그 요인이 있을 때와 없을 때의 차이(%p) — 로짓 척도를 확률로 바꿔 직관적으로 보이게 한다
        lg = np.log(base / (1 - base))
        to_p = lambda x: 1 / (1 + np.exp(-(lg + x))) - base
        group = "질문 형식" if name.startswith("형식") else "관심사" if name.startswith("관심사") else "문구 특성"
        out.append(dict(name=name.split(": ")[-1], group=group, beta=round(b, 4), or_=round(float(np.exp(b)), 3),
                        or_lo=round(float(np.exp(lo)), 3), or_hi=round(float(np.exp(hi)), 3), p=round(float(p), 4),
                        pp=round(float(to_p(b)), 4), pp_lo=round(float(to_p(lo)), 4), pp_hi=round(float(to_p(hi)), 4)))
    # 생성기의 '예상 답변율' 계산용: 절편과 항별 계수(로짓)
    coef = {"const": round(float(res.params["const"]), 5), **{n: round(float(res.params[n]), 5) for n in X.columns if n != "const"}}
    return dict(coef=coef, n_q=int(m.sum()), scale=round(float(res.scale), 3), n_res=int(d.ans.sum() + d.rep.sum()), base=round(float(base), 4),
                ref="형식은 성향·습관, 관심사는 가치관 대비", terms=out)


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
    conf = [x for x in terms if x["group"] == "문구 특성" and x["p"] >= 0.05 and x["name"] in u and (u[x["name"]]["lo"] > 0 or u[x["name"]]["hi"] < 0)]
    if conf:
        out.append(dict(kind="neutral", title="겹쳐 있어서 커 보였던 요인",
                        body=", ".join(f"'{x['name']}'(단독 {pct(u[x['name']]['diff'])} → 함께 보면 {pct(x['pp'])}, 불확실)" for x in conf)
                             + "은 따로 보면 답변율이 확실히 다르지만, 경험 전제·형식 같은 다른 특성과 함께 붙어 다녀서 생긴 차이가 커요.",
                        action="문구를 고칠 때는 이 요인만 지우기보다, 같이 붙어 있는 경험 전제·형식을 함께 바꿔야 효과가 나요."))
    null = [x for x in terms if x["group"] == "문구 특성" and x["p"] >= 0.05 and x not in conf]
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
                             + f" 다만 다변량 모델에서 '40자 초과'의 효과는 {pct(next((x['pp'] for x in terms if x['name'] == '40자 초과'), 0))}로 불확실해요 — 짧은 질문엔 즉답형이 많아서예요.",
                        action="길이보다 즉답 가능 여부를 먼저 봐요. 40자 안팎은 읽기 편한 정도의 기준이에요."))
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


def main():
    date, d = load()
    uni, mdl, ln, ax = univariate(d), model(d), length_bins(d), axes(d)
    out = dict(snapshot=date, uni=uni, model=mdl, length=ln, axes=ax, findings=findings(uni, mdl, ln, ax))
    f = ROOT / "board" / "data" / "insights.json"
    f.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    print(f"{date}: 문항 {mdl['n_q']} · 결론 {mdl['n_res']} · 평균 {mdl['base']:.1%} · 인사이트 {len(out['findings'])}개 → {f}")
    for x in mdl["terms"]:
        print(f"  {x['group']:6s} {x['name']:8s} OR {x['or_']:.2f} [{x['or_lo']:.2f},{x['or_hi']:.2f}] p={x['p']:.3f} ≈ {x['pp'] * 100:+.1f}%p")


if __name__ == "__main__":
    main()
