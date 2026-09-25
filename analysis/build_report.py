"""output/*.json + scorecard + 수정안 → output/report.html"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lint_questions import lint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

s = json.loads((OUT / "summary.json").read_text())
s["improve"] = json.loads((OUT / "improve_summary.json").read_text())
card = pd.read_csv(OUT / "question_scorecard.csv")[["질문 ID", "질문 문구", "관심사"]]
rw = pd.read_csv(OUT / "rewrites_scored.csv").fillna("").merge(card, on="질문 ID")
order = {"A": 0, "B": 1}
rw["_o"] = rw["분류"].str[0].map(order).fillna(2)
rw = rw.sort_values(["_o", "예측 변화(%p)"], ascending=[True, False])
s["rewrites"] = rw[["질문 ID", "질문 문구", "관심사", "분류", "조치 유형", "원인 진단",
                    "수정안 1 (권장)", "예측 변화(%p)"]].to_dict("records")
s["zero"] = int((pd.read_csv(OUT / "question_scorecard.csv")["노출"] == 0).sum())


def high(q):
    return any(level == "높음" for level, _ in lint(q))


s["high_before"] = int(sum(high(q) for q in rw["질문 문구"]))
s["resid_high"] = int(sum(high(q) for q in rw["수정안 1 (권장)"].str.replace(r"^\([^)]*\)\s*", "", regex=True)))
tpl = (Path(__file__).parent / "report_template.html").read_text()
(OUT / "report.html").write_text(tpl.replace("__DATA__", json.dumps(s, ensure_ascii=False, default=float)))
print("wrote", OUT / "report.html", s["high_before"], s["resid_high"])
