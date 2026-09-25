"""수정안 검수 + 기대 효과 추정.

1. 수정안 1을 같은 특성 규칙으로 다시 검사해 위험 특성이 남았는지 확인 (자체 lint)
2. 원인 모델(GLM)에 수정안 문구를 넣어 예측 답변율을 계산하고,
   현재 노출량을 그대로 가정했을 때 전체 답변율이 얼마나 바뀌는지 추정
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze import OUT, design, fit_glm, load  # noqa: E402
from features import RISK_FEATURES, extract  # noqa: E402


def action_type(principle: str, text: str) -> str:
    if text.startswith("(유지"):
        return "유지(노출 시간 조정)" if "노출" in text else "유지"
    if "오류" in principle:
        return "오류 수정"
    if "노출" in principle or "(저녁 노출)" in text or "(아침 노출)" in text:
        return "노출 시간 조정 + 재작성"
    if "경미" in principle:
        return "경미 수정"
    return "재작성"


def main():
    d = load()
    m, _ = fit_glm(d)
    rw = pd.read_csv(Path(__file__).parent / "rewrites.csv").fillna("")
    card = pd.read_csv(OUT / "question_scorecard.csv")
    rw = rw.merge(card[["질문 ID", "분류", "추정 답변율(축소)"]], on="질문 ID", how="left")
    rw["조치 유형"] = [action_type(p, t) for p, t in zip(rw["수정 원칙"], rw["수정안 1 (권장)"])]

    # 1) 자체 lint: 수정안에 남은 위험 특성
    clean = rw["수정안 1 (권장)"].str.replace(r"^\((유지|저녁 노출|아침 노출)[^)]*\)\s*", "", regex=True)
    feats = pd.DataFrame([extract(q) for q in clean])
    rw["수정안 잔여 위험"] = feats[RISK_FEATURES].apply(
        lambda r: ", ".join(k for k in RISK_FEATURES if r[k]), axis=1)

    # 2) 기대 효과: 원 질문의 관심사·레벨은 유지하고 문구 특성만 교체
    base = d.set_index("id").loc[rw["질문 ID"]].reset_index()
    X_old = design(base)
    new = base.copy()
    for k, v in feats.items():
        new[k] = v.values
    X_new = design(new)
    rw["예측(기존 문구)"] = m.predict(X_old).values
    rw["예측(수정안)"] = m.predict(X_new).values
    # 유지·노출 조정 항목은 문구 효과 0으로 처리
    keep = rw["조치 유형"].str.startswith("유지")
    rw.loc[keep, "예측(수정안)"] = rw.loc[keep, "예측(기존 문구)"]
    rw["예측 변화(%p)"] = (rw["예측(수정안)"] - rw["예측(기존 문구)"]) * 100

    # 전체 답변율 영향: 현재 노출 수로 가중 (미노출 질문은 제외됨)
    exp = base["exp"].values
    gain = float(np.sum(exp * (rw["예측(수정안)"] - rw["예측(기존 문구)"])))
    total_exp, total_ans = d.exp.sum(), d.ans.sum()
    summary = {
        "n_rewrites": len(rw),
        "action_counts": rw["조치 유형"].value_counts().to_dict(),
        "residual_risk": int((rw["수정안 잔여 위험"] != "").sum()),
        "residual_examples": rw.loc[rw["수정안 잔여 위험"] != "", ["질문 ID", "수정안 1 (권장)", "수정안 잔여 위험"]].to_dict("records"),
        "mean_pred_old": float(rw.loc[~keep, "예측(기존 문구)"].mean()),
        "mean_pred_new": float(rw.loc[~keep, "예측(수정안)"].mean()),
        "affected_exposure_share": float(exp.sum() / total_exp),
        "overall_now": float(total_ans / total_exp),
        "overall_projected": float((total_ans + gain) / total_exp),
        "extra_answers_per_month": gain,
    }
    rw.to_csv(OUT / "rewrites_scored.csv", index=False, encoding="utf-8-sig")
    cols = ["질문 ID", "분류", "조치 유형", "원인 진단", "수정 원칙", "수정안 1 (권장)", "수정안 2 (대안)",
            "예측(기존 문구)", "예측(수정안)", "예측 변화(%p)", "수정안 잔여 위험"]
    with pd.ExcelWriter(OUT / "question_review.xlsx", mode="a", engine="openpyxl") as w:
        rw[cols].sort_values("예측 변화(%p)", ascending=False).to_excel(w, sheet_name="수정안 효과 추정", index=False)
        ws = w.book["수정안 효과 추정"]
        ws.freeze_panes = "B2"
        for col in ws.columns:
            width = max(len(str(c.value or "")) for c in col[:50])
            ws.column_dimensions[col[0].column_letter].width = min(max(8, width * 1.6), 60)
    (OUT / "improve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "residual_examples"}, ensure_ascii=False, indent=1))
    for r in summary["residual_examples"]:
        print(r)


if __name__ == "__main__":
    main()
