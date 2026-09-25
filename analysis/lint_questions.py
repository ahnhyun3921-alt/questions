"""새 질문 문구 검수 도구.

사용법:
  python analysis/lint_questions.py "오늘 기억에 남는 소리가 있다면, 어떤 건가요?"
  python analysis/lint_questions.py -f new_questions.txt   # 한 줄에 질문 하나

데이터로 확인된 위험 패턴과 작성 가이드 위반을 표시한다.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from features import extract  # noqa: E402

WH = re.compile(r"무엇|뭐|언제|어떤|어떻게|누구|어디|왜|몇")

# (조건, 등급, 메시지) - 등급: 높음 = 통제 후 유의, 보통 = 방향성, 가이드 = 작성 규칙
CHECKS = [
    (lambda q, f: f["오늘 기반"], "높음",
     "'오늘' 기반 (답변 오즈 0.45배). '평소/주로'로 바꾸거나 저녁 노출 전용으로 지정하세요."),
    (lambda q, f: f["경험 전제형(~있다면/~적 있)"] and "없다면" not in q, "높음",
     "경험 전제형 (0.64배). '~하는 편인가요/주로 무엇인가요'로 바꾸거나 '없다면 ~' 대안을 붙이세요."),
    (lambda q, f: f["최근·이번 주 회상"], "보통",
     "기간 제한 회상 (0.70배, 유의하지 않음). 기간을 빼도 뜻이 통하면 빼세요."),
    (lambda q, f: f["특정 활동 전제"] and "무엇이든" not in q, "보통",
     "특정 활동 전제 (0.76배, 유의하지 않음). 해당 활동을 안 하는 사람도 답할 수 있게 대상을 넓히세요."),
    (lambda q, f: bool(re.search(r"누구(인가요|였나요|예요|일까요)", q)), "보통",
     "특정 인물 지목. 관계 카테고리의 '누구' 질문 답변율은 28%입니다. 그 사람의 특징을 물어보세요."),
    (lambda q, f: bool(re.search(r"[았었였했냈났웠쳤졌갔왔봤눴줬]나요\?$", q)) and not WH.search(q), "가이드",
     "예/아니오로 끝나는 닫힌 질문. '무엇/언제/어떤'으로 열어주세요."),
    (lambda q, f: q.count("?") >= 2, "가이드", "질문이 두 개입니다. 하나만 물어보세요."),
    (lambda q, f: f["문구 길이"] > 40, "가이드", "40자 초과. 조건을 줄여보세요."),
    (lambda q, f: len(re.findall(r"다면", q)) >= 2, "가이드", "'~다면' 조건이 두 번 겹칩니다."),
]


def lint(q: str):
    q = q.strip()
    f = extract(q)
    return [(lvl, msg) for cond, lvl, msg in CHECKS if cond(q, f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="*")
    ap.add_argument("-f", "--file")
    a = ap.parse_args()
    qs = list(a.questions)
    if a.file:
        qs += [l for l in Path(a.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    bad = 0
    for q in qs:
        issues = lint(q)
        bad += any(l == "높음" for l, _ in issues)
        print(("✗ " if issues else "✓ ") + q)
        for lvl, msg in issues:
            print(f"    [{lvl}] {msg}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
