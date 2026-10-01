"""개선함 페이지에서 바꾼 질문 분류(db `meta`)를 수동 분류표에 합친다.

사용법:
  1) ArtifactData `list` 로 개선함 db의 `meta` 컬렉션을 out_dir 에 받는다 (예: tmp/meta/meta/<DB id>.json)
  2) python analysis/import_page_meta.py tmp/meta/meta
  3) python analysis/tag_axes_v2.py → build_board_data → build_insights
meta 문서: {pt: s|r|k, ax: 축 키 또는 '', sig: A|B|'', subN: 0~6, by, at}
페이지 분류가 수동 분류표(axis_tags_v2_manual.csv)의 같은 행을 덮어쓴다. '관심사 제안'은 그대로 둔다.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def main(src):
    src = Path(src)
    files = sorted(src.glob("*.json")) if src.is_dir() else [src]
    path = HERE / "axis_tags_v2_manual.csv"
    man = pd.read_csv(path, dtype=str).fillna("").set_index("DB id")
    n = 0
    for f in files:
        j = json.loads(f.read_text())
        for d in (j if isinstance(j, list) else [j]):
            key = str(d.get("id") or d.get("doc_id") or f.stem)
            d = d.get("data", d)
            row = {"축 키": d.get("ax", ""), "목적": d.get("pt", ""), "소주제": str(d.get("subN") or ""), "신호": d.get("sig", "") if d.get("ax") else ""}
            if key in man.index:
                for c, v in row.items():
                    man.loc[key, c] = v
            else:
                man.loc[key] = {**row, "관심사 제안": ""}
            n += 1
    man.reset_index().to_csv(path, index=False)
    print(f"페이지 분류 {n}건을 {path.name}에 합쳤어요.")


if __name__ == "__main__":
    main(sys.argv[1])
