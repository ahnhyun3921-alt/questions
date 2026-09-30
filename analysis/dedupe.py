"""질문 중복 검사: 글자 n-gram 코사인 + 핵심어 겹침(조사·어미 제거 후 어간 Jaccard).

사용법:
  python analysis/dedupe.py                       # 최종 권장 문구(output/final_questions.csv) 내부 중복
  python analysis/dedupe.py --new 파일.csv        # 신규 후보(열: 문구) vs 기존 최종 문구
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
STOP = set("나 내 나의 나를 나에게 나는 내가 무엇 무엇인가요 뭐 뭘까요 어떤 어떻게 언제 가장 주로 편인가요 것 건가요 있나요 "
           "하나요 인가요 중 더 는 은 이 가 을 를 에 에서 의 와 과 도 요 좋아요 무엇이든 수 있다면 한다면 때 순간 사람".split())
SUFFIX = re.compile(r"(으로|에게|에서|까지|부터|처럼|보다|하고|이나|이랑|랑|이란|라는|이라는|은|는|이|가|을|를|에|의|와|과|도|로|만|요|나요|가요|까요|죠)$")


def stems(q):
    toks = re.findall(r"[가-힣A-Za-z0-9]+", q)
    out = set()
    for t in toks:
        t = SUFFIX.sub("", t)
        if len(t) < 2 or t in STOP:
            continue
        out.add(t[:2])  # 어간 근사: 앞 2글자
    return out


def pairs(a_text, b_text=None, char_thr=0.55, jac_thr=0.5):
    same = b_text is None
    b_text = a_text if same else b_text
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit(list(a_text) + list(b_text))
    M = cosine_similarity(tf.transform(a_text), tf.transform(b_text))
    sa, sb = [stems(q) for q in a_text], [stems(q) for q in b_text]
    J = np.array([[len(x & y) / len(x | y) if x | y else 0 for y in sb] for x in sa])
    if same:
        np.fill_diagonal(M, 0); np.fill_diagonal(J, 0)
        M, J = np.triu(M), np.triu(J)
    hit = (M >= char_thr) | ((J >= jac_thr) & (M >= 0.3))
    i, j = np.where(hit)
    return [(a, b, float(M[a, b]), float(J[a, b])) for a, b in zip(i, j)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new")
    a = ap.parse_args()
    f = pd.read_csv(ROOT / "output" / "final_questions.csv")
    f = f[~f["조치"].astype(str).str.startswith(("비활성", "중복 통합"))].reset_index(drop=True)
    base = f["최종 권장 문구"].str.strip().tolist()
    if a.new:
        n = pd.read_csv(a.new)
        res = pairs(n["문구"].tolist(), base)
        rows = [{"신규": n["문구"][x], "기존 ID": int(f["질문 ID"][y]), "기존": base[y], "글자유사": round(c, 2), "핵심어겹침": round(j, 2)} for x, y, c, j in res]
    else:
        res = pairs(base)
        rows = [{"ID_A": int(f["질문 ID"][x]), "A": base[x], "ID_B": int(f["질문 ID"][y]), "B": base[y],
                 "A변경": bool(f["변경"][x]), "B변경": bool(f["변경"][y]), "글자유사": round(c, 2), "핵심어겹침": round(j, 2)} for x, y, c, j in res]
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["글자유사", "핵심어겹침"], ascending=False)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
