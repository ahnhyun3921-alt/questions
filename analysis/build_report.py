"""output/summary.json + scorecard → output/report.html"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

s = json.loads((OUT / "summary.json").read_text())
card = pd.read_csv(OUT / "question_scorecard.csv")
a = card[card["분류"].str.startswith("A")].sort_values("추정 답변율(축소)")
s["rewrites"] = a[["질문 ID", "질문 문구", "관심사", "노출", "답변", "원인 진단", "수정안 1 (권장)"]].to_dict("records")
s["zero"] = int((card["노출"] == 0).sum())
tpl = (Path(__file__).parent / "report_template.html").read_text()
(OUT / "report.html").write_text(tpl.replace("__DATA__", json.dumps(s, ensure_ascii=False, default=float)))
print("wrote", OUT / "report.html")
