"""'나답 질문 개선함' 페이지용 데이터(board/data/questions.json)를 만든다.

사용법: python analysis/build_full_db.py && python analysis/build_board_data.py
입력: output/daily_questions_revised.csv(DB 형식 전체본)와 분석 산출물
각 행: DB 열 + 목적(purpose) + 바뀐 이유 + 출처 + 자아분석 축 + 성과 수치
"""
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from db_questions import GAP_LEN, GAP_START, INTEREST_ID, original_first_ids  # noqa: E402
from axes_v2 import AXES_V2, SUBTOPICS, by_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
NEW = Path(__file__).parent / "new"
CAT = {v: k for k, v in INTEREST_ID.items()}

PURPOSE_CAT = {
    "취향": "좋아하는 것을 꺼내 나의 취향 지도를 그리게 해요.",
    "감정": "감정을 알아차리고 다루는 방식을 돌아보게 해요.",
    "루틴": "생활 습관과 시간을 쓰는 방식을 돌아보게 해요.",
    "관계": "사람들 사이에서의 나를 돌아보게 해요.",
    "사랑": "연애와 좋아하는 사람 앞의 나를 돌아보게 해요.",
    "가치관": "삶의 기준과 선택을 정리하게 해요.",
}
PURPOSE_FORM = {
    "c": ("양자택일", "둘 중 고르게 해서 바로 답할 수 있어요."),
    "p": ("선호(최애)", "좋아하는 것 하나를 떠올리게 해서 부담이 적어요."),
    "h": ("상상·가정", "상상으로 답하게 해서 경험이 없어도 답할 수 있어요."),
    "t": ("성향·습관", "평소 패턴을 관찰하게 해요."),
    "r": ("일화 회상", "기억 속 장면을 꺼내게 해요."),
    "d": ("개념·가치", "생각을 한 문장으로 정리하게 해요."),
    "n": ("특정 인물", "떠오르는 사람을 통해 관계를 돌아보게 해요."),
    "o": ("기타", ""),
}


def stats_id_of(db_id):
    try:
        i = int(db_id)
    except (TypeError, ValueError):
        return None
    if i < GAP_START:
        return i
    if i < GAP_START + GAP_LEN:
        return None
    return i - GAP_LEN


def s(x):
    return "" if pd.isna(x) else str(x)


def duplicate_pairs(rows):
    """활성 질문끼리 중복 후보(dedupe.pairs)를 뽑고 analysis/dup_labels.csv 판정을 붙인다.
    판정 diff(형식만 비슷)는 빼고, 판정이 없는 새 후보는 v='new'로 남긴다(월간 루틴에서 판정)."""
    from dedupe import pairs
    live = [r for r in rows if not r["deleted"]]
    lab_path = Path(__file__).parent / "dup_labels.csv"
    lab = {}
    if lab_path.exists():
        for _, x in pd.read_csv(lab_path, dtype=str).fillna("").iterrows():
            lab[frozenset((x["a"], x["b"]))] = (x["판정"], x["메모"])
    out = []
    for a, b, c, _ in pairs([r["text"] for r in live]):
        ka, kb = live[a]["key"], live[b]["key"]
        v, note = lab.get(frozenset((ka, kb)), ("new", ""))
        if v != "diff":
            out.append({"a": ka, "b": kb, "score": round(c, 2), "v": v, "note": note})
    return sorted(out, key=lambda x: ({"dup": 0, "similar": 1, "new": 2}[x["v"]], -x["score"]))


def main():
    orig_db = pd.read_csv(sorted((ROOT / "data").glob("daily_questions_*.csv"))[-1], dtype=str).fillna("").set_index("id")
    full = pd.read_csv(OUT / "daily_questions_revised.csv", dtype=str).fillna("")
    card = pd.read_csv(OUT / "question_scorecard.csv").set_index("질문 ID")
    fin = pd.read_csv(OUT / "final_questions.csv").set_index("질문 ID")
    tags = pd.read_csv(Path(__file__).parent / "question_tags.csv").set_index("id")
    axis = pd.read_csv(Path(__file__).parent / "axis_tags.csv").set_index("질문 ID")
    block = pd.read_csv(NEW / "level1_block.csv").set_index("DB id")
    # 자아분석 축 v2(관심사별 고유 축) + 목적 유형·소주제·신호 방향·관심사 제안 — tag_axes_v2.py 결과
    v2 = pd.read_csv(Path(__file__).parent / "axis_tags_v2.csv", dtype=str).fillna("").set_index("DB id")
    AX2 = by_key()
    newsets = pd.concat([pd.read_csv(OUT / f) for f in
                         ["new_questions.csv", "new_questions_scraped.csv", "new_questions_love.csv",
                          "new_questions_fun.csv", "new_questions_self.csv"]]).set_index("신규 ID")

    rw = pd.read_csv(Path(__file__).parent / "rewrites.csv").fillna("").set_index("질문 ID")
    pa_path = NEW / "page_added.csv"
    page_added = pd.read_csv(pa_path, dtype={"DB id": str}).fillna("").set_index("DB id") if pa_path.exists() else pd.DataFrame()
    dec_path = ROOT / "data" / "decisions.csv"
    dec = pd.read_csv(dec_path, dtype=str).fillna("").set_index("id") if dec_path.exists() else pd.DataFrame()
    clean = lambda t: re.sub(r"^\([^)]*\)\s*", "", t).strip()

    keep1 = {str(i) for i in original_first_ids()}

    def alternatives(r, sid, cur):
        """추천 수정안 목록: [라벨, 문구]. 지금 개선본 문구와 같은 건 표시만 하고 남긴다."""
        alts = []
        if (r["id"] in keep1 and not r["신규 ID"]) or r["변경 구분"].startswith("레벨1 묶음 원본 유지"):   # 첫 질문은 원본 그대로 — 수정안을 띄우지 않는다
            return []
        def add(label, text):
            text = (text or "").strip()
            if text and not text.startswith("(") and "중복만 제거" not in text and all(text != t for _, t in alts):
                alts.append([label, text])
        if r["기존 question_text"]:
            add("지금 DB 문구", r["기존 question_text"])
        if sid is not None and sid in rw.index:
            add("수정안 1 (권장)", rw.loc[sid, "수정안 1 (권장)"])
            add("수정안 2 (대안)", clean(rw.loc[sid, "수정안 2 (대안)"]))
        dk = str(sid) if sid is not None else (f"D{r['id']}" if r["id"] and 661 <= int(r["id"]) <= 684 else "")
        if dk and len(dec) and dk in dec.index and dec.loc[dk, "status"] in ("chosen", "applied"):
            add("검토에서 고른 문구", dec.loc[dk, "text"])
        if r["id"] and r["id"].isdigit() and int(r["id"]) in block.index:
            add("수정안 (권장)", block.loc[int(r["id"]), "권장 문구"])
        return [[l, t, t == cur] for l, t in alts]

    def decision(r, sid):
        """검토 페이지 결정 상태(chosen/applied/skipped/pending) — 없으면 ''."""
        dk = str(sid) if sid is not None else (r["신규 ID"] or (f"D{r['id']}" if r["id"] and 661 <= int(r["id"]) <= 684 else ""))
        return dec.loc[dk, "status"] if dk and len(dec) and dk in dec.index else ""

    rows = []
    for _, r in full.iterrows():
        key = r["id"] or r["신규 ID"]
        sid = stats_id_of(r["id"]) if r["id"] and not r["신규 ID"] else None
        cat = CAT.get(int(float(r["interest_id"])), "")
        level = int(float(r["question_level"]))
        form, ax, axnote, why, source, src_url, pred = "", "", "", "", "", "", None
        exp = ans = rep = None
        if r["신규 ID"].startswith("P") and r["신규 ID"][1:] in page_added.index:   # 개선함 페이지에서 만든 질문
            pa = page_added.loc[r["신규 ID"][1:]]
            form, ax, source = s(pa["F"]), s(pa["분석 축"]), "개선함 '새 질문 만들기'"
            why = " · ".join(x for x in [s(pa["근거"]), f"태그: {s(pa['태그 출처'])}"] if x)
            pred = float(pa["예상 답변율"]) if s(pa["예상 답변율"]) else None
        elif r["신규 ID"]:
            n = newsets.loc[r["신규 ID"]]
            form = s(n.get("F"))
            ax, axnote = s(n.get("분석 축")), s(n.get("해석 가이드"))
            source = s(n.get("출처"))
            src_url = s(n.get("출처 URL"))
            orig = s(n.get("원문")) or s(n.get("참고 원문"))
            why = " · ".join(x for x in [f"원문: {orig}" if orig else "", s(n.get("인기·검증 근거"))] if x)
            pred = None if pd.isna(n.get("예상 답변율(결론 기준)")) else round(float(n["예상 답변율(결론 기준)"]), 3)
        elif r["id"] and int(r["id"]) in block.index:
            b = block.loc[int(r["id"])]
            form, why, source = s(b["F"]), s(b["판정"]), "레벨 1 첫 질문 묶음(DB 661~684)"
            if r["변경 구분"].startswith("레벨1 묶음 원본 유지"):
                why = "원본 문구 그대로 · 비어 있던 가이드만 채움"
        if sid is not None and sid in card.index:
            c = card.loc[sid]
            exp, ans, rep = int(c["노출"]), int(c["답변"]), int(c["교체"])
            if sid in tags.index and not form:
                form = s(tags.loc[sid, "F"])
            if sid in axis.index:
                ax, axnote = axis.loc[sid, "분석 축"], axis.loc[sid, "해석 가이드"]
            if sid in fin.index and r["변경 구분"].startswith("문구 수정"):
                why = " → ".join(x for x in [s(fin.loc[sid, "이유"]), s(fin.loc[sid, "수정 원칙"])] if x)
        ax1 = ax                                    # v1 공통 축(기록용)
        t2 = v2.loc[r["id"]] if r["id"] in v2.index else None
        ak = t2["축 키"] if t2 is not None else ""
        pt = t2["목적"] if t2 is not None else ""
        recat = t2["관심사 제안"] if t2 is not None else ""
        sub_cat = recat or cat
        subn = int(t2["소주제"]) if t2 is not None and t2["소주제"] else 0
        sub = SUBTOPICS.get(sub_cat, [""] * 6)[subn - 1] if subn else ""
        sig = t2["신호"] if t2 is not None else ""
        ax = AX2[ak]["name"] if ak in AX2 else ""
        axnote = AX2[ak]["reads"] if ak in AX2 else ""
        fname, fnote = PURPOSE_FORM.get(form, ("", ""))
        purpose = []
        if level == 1:
            purpose.append("첫 질문용이에요. 가볍고 바로 답할 수 있어 기록을 시작하게 해요.")
        purpose.append(PURPOSE_CAT.get(cat, ""))
        if fnote:
            purpose.append(fnote)
        if ak in AX2 and pt == "s":
            A = AX2[ak]
            purpose.append(f"신호형이에요. 답으로 '{A['name']}' 축에서 어느 쪽인지 읽어요"
                           + (f"(첫 선택지 → {A['a'] if sig == 'A' else A['b']})." if sig else "."))
        elif pt == "r":
            purpose.append("성찰형이에요. 생각과 이유를 꺼내는 것 자체가 기록이 돼요." + (f" '{ax}' 축의 맥락도 담겨요." if ax else ""))
        elif pt == "k":
            purpose.append("기록형이에요. 취향·일상을 모아 '나의 목록'을 쌓아요.")
        rows.append({
            "key": key, "id": r["id"], "newId": r["신규 ID"], "statsId": sid,
            "interest_id": int(float(r["interest_id"])), "cat": cat, "level": level,
            "text": r["question_text"], "oldText": r["기존 question_text"],
            "g": [r["empathy_guide"], r["hint_guide"], r["leading_question_guide"]],
            "deleted": bool(r["deleted_at"]), "deletedAt": r["deleted_at"],
            "created_at": r["created_at"], "updated_at": r["updated_at"],
            "change": r["변경 구분"], "catChange": r.get("관심사 변경", ""), "set": r["세트"],
            "form": fname, "axis": ax, "axisNote": axnote, "ax": ak, "axis1": ax1,
            "pt": pt, "sub": sub, "subN": subn, "sig": sig, "recat": recat, "purpose": " ".join(p for p in purpose if p),
            "why": why, "source": source, "srcUrl": src_url, "pred": pred,
            "exp": exp, "ans": ans, "rep": rep,
            "alts": alternatives(r, sid, r["question_text"]), "decided": decision(r, sid),
            # 원본 DB 값 [interest_id, 문구, 레벨, 가이드 3종, 삭제 여부] — CSV '수정 여부' 열 계산용(신규는 null)
            "orig": None if r["신규 ID"] or r["id"] not in orig_db.index else [
                int(orig_db.loc[r["id"], "interest_id"]), orig_db.loc[r["id"], "question_text"], int(orig_db.loc[r["id"], "question_level"]),
                orig_db.loc[r["id"], "empathy_guide"], orig_db.loc[r["id"], "hint_guide"], orig_db.loc[r["id"], "leading_question_guide"],
                orig_db.loc[r["id"], "deleted_at"] != ""],
        })
    dups = duplicate_pairs(rows)
    built = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")
    out = ROOT / "board" / "data" / "questions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    axes = [{k: v for k, v in A.items() if k != "kw"} | {"cat": c} for c, lst in AXES_V2.items() for A in lst]
    out.write_text(json.dumps({"builtAt": built, "rows": rows, "dups": dups, "axes": axes, "subtopics": SUBTOPICS},
                              ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    live = [x for x in rows if not x["deleted"]]
    print(f"중복 후보 {len(dups)}쌍 (같은 질문 {sum(x['v'] == 'dup' for x in dups)} · 비슷 {sum(x['v'] == 'similar' for x in dups)} · 판정 전 {sum(x['v'] == 'new' for x in dups)})")
    print(f"{len(rows)}행 (활성 {len(live)}) · 목적 빈 칸 {sum(not x['purpose'] for x in rows)} · {out.stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
