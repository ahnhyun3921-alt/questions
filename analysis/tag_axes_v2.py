"""개선본의 모든 활성 질문에 자아분석 축 v2를 매긴다.

사용법: python analysis/tag_axes_v2.py   (build_board_data.py 전에)
입력: output/daily_questions_revised.csv(전체본), analysis/axis_tags.csv(v1), analysis/new/*.csv(신규 v1 축)
결과: analysis/axis_tags_v2.csv  (DB id, 관심사, 문구, 축 키, 축 이름, 목적, 소주제, 신호, 관심사 제안, 배정 방법, 근거)

수동 파일 열 (axis_tags_v2_manual.csv)
- 축 키: 그 질문이 읽는 축. 축은 '관심사 제안'이 있으면 그 관심사, 없으면 지금 관심사의 축이어야 해요.
- 목적: s 신호형(답이 축의 한쪽 극을 가리킴) / r 성찰형(생각을 꺼내게 함) / k 기록형(취향·일상 수집)
- 소주제: 관심사별 1~6 (axes_v2.SUBTOPICS)
- 신호: 첫 선택지(또는 '그렇다')가 가리키는 극 A/B. 서술형·척도형은 비움
- 관심사 제안: 다른 관심사에 더 맞으면 그 관심사(재배정 후보 큐로 감)

배정 순서
1) 수동 확정(axis_tags_v2_manual.csv 의 DB id → 축 키, 빈 값이면 '기록형'으로 확정)
2) 그 관심사 축들의 단서어 점수 — 가장 많이 맞은 축(동점이면 v1 대응 축 우선)
3) 단서어가 없으면 v1 축 → v2 대응표
4) 그래도 없으면 '기록형'(성향 신호보다 기록·취향 수집이 목적인 질문)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from axes_v2 import AXES_V2, V1_TO_V2, by_key  # noqa: E402
from db_questions import stats_to_db  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parent


def v1_axes():
    out = {}
    t = pd.read_csv(HERE / "axis_tags.csv")
    for sid, ax in zip(t["질문 ID"], t["분석 축"]):
        out[str(stats_to_db(int(sid)))] = ax
    full = pd.read_csv(ROOT / "output" / "daily_questions_revised.csv", dtype=str).fillna("")
    for i, ax in zip(full["id"], full["분석 축"]):
        if ax and i not in out:
            out[i] = ax
    pa = HERE / "new" / "page_added.csv"
    if pa.exists():
        for i, ax in pd.read_csv(pa, dtype={"DB id": str}).fillna("")[["DB id", "분석 축"]].values:
            if ax:
                out.setdefault(i, ax)
    return out


def main():
    full = pd.read_csv(ROOT / "output" / "daily_questions_revised.csv", dtype=str).fillna("")
    live = full[full["deleted_at"] == ""]
    cat_of = {1: "취향", 2: "감정", 3: "루틴", 4: "관계", 5: "사랑", 6: "가치관"}
    v1 = v1_axes()
    K = by_key()
    man_path = HERE / "axis_tags_v2_manual.csv"
    man = pd.read_csv(man_path, dtype=str).fillna("").set_index("DB id") if man_path.exists() else pd.DataFrame()
    manual = man["축 키"].to_dict() if len(man) else {}
    # 페이지에서 만든 질문은 만들 때 고른 v2 축·목적·신호·소주제를 그대로 쓴다(수동 파일에 없을 때)
    pa = HERE / "new" / "page_added.csv"
    if pa.exists():
        for _, r in pd.read_csv(pa, dtype=str).fillna("").iterrows():
            i = r["DB id"]
            if i in manual or (r.get("분석 축", "") not in K and not r.get("목적", "")):
                continue
            manual[i] = r["분석 축"] if r["분석 축"] in K else ""
            man.loc[i] = {"축 키": manual[i], "목적": r.get("목적", "") or ("s" if manual[i] else "k"), "소주제": r.get("소주제", ""),
                          "신호": r.get("신호", ""), "관심사 제안": ""}
    rows = []
    for _, r in live.iterrows():
        i, text = r["id"], r["question_text"]
        cat = cat_of.get(int(float(r["interest_id"])), "")
        axes = AXES_V2.get(cat, [])
        if i in manual:
            key, how, hit = manual[i], "수동 확정", ""
        else:
            scores = {ax["key"]: [w for w in ax["kw"] if w in text] for ax in axes}
            prior = V1_TO_V2.get((v1.get(i, ""), cat), "")
            best = max(scores, key=lambda k: (len(scores[k]), k == prior)) if scores else ""
            if best and scores[best]:
                key, how, hit = best, "단서어", " ".join(scores[best])
            elif prior:
                key, how, hit = prior, "v1 축 대응", v1.get(i, "")
            else:
                key, how, hit = "", "기록형", ""
        mr = man.loc[i] if i in manual else {}
        rows.append({"DB id": i, "관심사": cat, "문구": text, "축 키": key, "축 이름": K[key]["name"] if key in K else "",
                     "목적": mr.get("목적", "") if len(mr) else ("s" if key else "k"),
                     "소주제": mr.get("소주제", "") if len(mr) else "", "신호": mr.get("신호", "") if len(mr) else "",
                     "관심사 제안": mr.get("관심사 제안", "") if len(mr) else "",
                     "배정 방법": how, "근거": hit})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "axis_tags_v2.csv", index=False, encoding="utf-8-sig")
    print(f"활성 {len(out)}개 · 축 배정 {int((out['축 키'] != '').sum())}개 ({(out['축 키'] != '').mean():.0%})")
    print(out.groupby(["관심사", "축 이름"]).size().to_string())
    print(out["배정 방법"].value_counts().to_string())
    print(out["목적"].value_counts().to_string())
    bad = [(r["DB id"], r["축 키"]) for _, r in out.iterrows()
           if r["축 키"] in K and K[r["축 키"]]["cat"] != (r["관심사 제안"] or r["관심사"])]
    assert not bad, f"축이 관심사와 맞지 않음: {bad[:10]}"


if __name__ == "__main__":
    main()
