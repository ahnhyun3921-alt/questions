"""개선함 페이지에서 만든 신규 질문(db `added`)을 저장소로 가져온다.

사용법:
  1) ArtifactData `list` 로 개선함 db의 `added` 컬렉션을 out_dir 에 받는다
     (예: out_dir=tmp/added → tmp/added/added/<id>.json 파일들)
  2) python analysis/import_page_added.py tmp/added/added
결과:
  analysis/new/page_added.csv   (DB id·관심사·레벨·문구·가이드·형식·축·태그 T~C·F·태그 출처·만든 시각·근거)
  analysis/new/page_added_ids.txt (build_full_db.py 가 새 id를 매길 때 건너뛸 id)
그 뒤 build_full_db → build_board_data → build_insights 순서로 돌리면 전체본·중복 판정·모델 학습·레벨 판정에 들어간다.

태그가 저장돼 있지 않은 예전 문서는 페이지와 같은 규칙(문구로 자동 추정)으로 채우고 '태그 출처'에 표시한다.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

NEW = Path(__file__).parent / "new"
FORM_CODE = {"선호(최애)": "p", "양자택일": "c", "성향·습관": "t", "상상·가정": "h", "일화 회상": "r", "개념·가치": "d", "특정 인물": "n", "기타": "o"}
CAT = {1: "취향", 2: "감정", 3: "루틴", 4: "관계", 5: "사랑", 6: "가치관"}


def guess_form(t):   # 페이지 guessForm 과 같은 규칙
    if re.search(r"한다면|라면|있다면|만약", t) and "적이" not in t:
        return "h"
    if re.search(r"(인가요|나요|가요)[,\s].*(인가요|나요|가요)\?*$", t) or re.search(r"\s중\s.*(뭐가|어느|무엇이) 더", t) or re.search(r"호인가요|파인가요", t):
        return "c"
    if re.search(r"가장 좋아하는|최애|좋아하는 .*(무엇|뭔)", t):
        return "p"
    if re.search(r"편인가요|편이에요", t):
        return "t"
    if "누구" in t:
        return "n"
    if re.search(r"적이|기억에 남는|언제였|했던", t):
        return "r"
    if re.search(r"의미|정의|생각하는 .*(은|는) 무엇|이란", t):
        return "d"
    return "t"


def guess_tags(t, f):   # 페이지 guessTags 와 같은 규칙
    return dict(T=2 if re.search(r"오늘|어제|이번 ?주|내일|올해", t) else 1 if re.search(r"요즘|최근", t) else 0,
                P=2 if re.search(r"적이 있|있다면|했던|해 본|해봤", t) else 0,
                E=1 if f in ("p", "c") and len(t) <= 35 else 3 if re.search(r"의미|정의|가치|어떤 사람", t) else 2,
                W=2 if re.search(r"상처|죽음|실패|비밀|후회|외로|슬픔|불안|두려", t) else 1,
                S=2 if re.search(r"약점|비밀|부끄|고백|콤플렉스", t) else 1,
                C=1 if re.search(r"온도|날씨로|색깔로|모양|맛으로|계절로", t) else 0)


def main(src):
    src = Path(src)
    files = sorted(src.glob("*.json")) if src.is_dir() else [src]
    docs = []
    for f in files:
        j = json.loads(f.read_text())
        for d in (j if isinstance(j, list) else [j]):
            docs.append(d.get("data", d))
    rows = []
    for d in docs:
        text = str(d.get("text", "")).strip()
        if not text:
            continue
        f = d.get("formCode") or FORM_CODE.get(d.get("form", ""), "") or guess_form(text)
        tags = d.get("tags") or {}
        src_tag = "저장된 태그" if tags else "문구로 자동 추정"
        if not tags:
            tags = guess_tags(text, f)
        rows.append({"DB id": int(d["id"]), "interest_id": int(d["interest_id"]), "관심사": CAT.get(int(d["interest_id"]), ""),
                     "question_level": int(d.get("level", 2)), "문구": text,
                     "empathy_guide": d.get("g0", ""), "hint_guide": d.get("g1", ""), "leading_question_guide": d.get("g2", ""),
                     "형식": d.get("form", ""), "F": f, "분석 축": d.get("axis", ""),
                     "목적": d.get("pt", ""), "신호": d.get("sig", ""), "소주제": d.get("subN", "") or "", "패밀리": d.get("fam", ""),
                     **{k: int(tags.get(k, 0)) for k in "TPEWSC"}, "태그 출처": src_tag,
                     "예상 답변율": (d.get("pred") or {}).get("p", ""), "근거": d.get("why", ""), "만든 시각": d.get("at", "")})
    out = pd.DataFrame(rows).sort_values("DB id")
    NEW.mkdir(exist_ok=True)
    out.to_csv(NEW / "page_added.csv", index=False, encoding="utf-8-sig")
    (NEW / "page_added_ids.txt").write_text("\n".join(str(i) for i in out["DB id"]) + "\n")
    print(f"페이지 신규 {len(out)}개 → {NEW / 'page_added.csv'} (태그 저장 {sum(out['태그 출처'] == '저장된 태그')} · 자동 추정 {sum(out['태그 출처'] != '저장된 태그')})")


if __name__ == "__main__":
    main(sys.argv[1])
