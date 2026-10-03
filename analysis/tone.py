"""화면 문구를 보고서체(간결한 명사·'~함/~임' 종결)로 바꾼다. 줄표(—)·청유·친절체 종결을 없앤다.
질문 문장(물음표로 끝나는 '~인가요?' 등)과 정규식은 건드리지 않는다."""
import re

PHRASES = [
    ("칸을 골라 주세요", "칸 선택 필요"), ("골라 주세요", "수정안 선택"),
    ("잠시 후 다시 해 주세요", "잠시 후 다시 시도"), ("잠시 후 다시 시도하세요", "잠시 후 다시 시도"),
    ("다시 눌러 주세요", "다시 시도"), ("새로고침해 주세요", "새로고침 필요"),
    ("'중복 아니에요'를 눌러 주세요", "'중복 아님' 선택"), ("중복 아니에요", "중복 아님"),
    ("둘 다 둘게요", "둘 다 유지"), ("지금 문구 그대로 둘게요", "지금 문구 유지"), ("그대로 둘게요", "그대로 두기"),
    ("문구를 먼저 써 주세요", "문구 먼저 입력"), ("직접 써 주세요", "직접 입력"), ("확인하고 저장해 주세요", "확인 후 저장"),
    ("확인해 주세요", "확인 필요"), ("채워 주세요", "채우기 필요"), ("문구를 넣어 보세요", "문구 입력 시 진단"),
    ("좁혀 보세요", "좁히기 권장"), ("구체적으로 물어보세요", "구체적으로 묻기 권장"),
    ("다시 재요", "다시 측정"), ("를 재요", "를 측정"), ("매달 다시 재요", "매달 재측정"),
    (" — ", ": "), ("✏️ ", ""), ("⬆ ", ""), ("↻ ", ""), (" ✓", ""),
]
SKIP = ("필", "중", "주", "개", "수", "요")   # 필요·중요·주요·개요·수요


def _batchim(ch):
    c = ord(ch) - 0xAC00
    return c % 28 if 0 <= c < 11172 else None


def _with(ch, jong):
    c = ord(ch) - 0xAC00
    return chr(0xAC00 + c - c % 28 + jong)


M, LM, L = 16, 10, 8   # ㅁ, ㄻ, ㄹ 종성 번호


def _parts(ch):
    c = ord(ch) - 0xAC00
    return c // 588, (c % 588) // 28, c % 28


def _mk(i, m, f):
    return chr(0xAC00 + i * 588 + m * 28 + f)


QUESTION_END = ("인가", "나", "가", "까", "지")   # '~인가요,'처럼 질문 문장 안이면 그대로


def _end(w):
    if w.endswith(SKIP):
        return None
    for k, v in {"모아": "모음", "주어": "줌", "해": "함", "써": "씀", "돼": "됨", "커": "큼", "꺼": "끔", "떠": "뜸"}.items():
        if w.endswith(k):
            return w[:-len(k)] + v
    last, prev = w[-1], (w[-2] if len(w) > 1 else "")
    if w.endswith("이에"):
        return w[:-2] + "임"
    if last == "예":
        return w[:-1] + "임"
    if last == "에":
        return w[:-2] + _with(prev, M) if prev and _batchim(prev) == 0 else w[:-1] + "임"
    if last in "라러" and prev and _batchim(prev) == L:          # 르 불규칙: 달라→다름, 눌러→누름
        return w[:-2] + _with(prev, 0) + "름"
    if last in "어아" and prev:
        b = _batchim(prev)
        if b == L:
            return w[:-2] + _with(prev, LM)                        # 만들어→만듦
        if b:
            return w[:-1] + "음"                                   # 있어→있음, 같아→같음
        return w[:-2] + _with(prev, M)                             # 바뀌어→바뀜
    i, m, f = _parts(last)
    if f:
        return None
    vow = {14: 13, 9: 8, 6: 20, 10: 11}                            # ㅝ→ㅜ(바꿔→바꿈) ㅘ→ㅗ(봐→봄) ㅕ→ㅣ(걸려→걸림) ㅙ→ㅚ
    if m in vow:
        return w[:-1] + _mk(i, vow[m], M)
    if m in (0, 1):                                                # ㅏ·ㅐ: 가→감, 내→냄
        return w[:-1] + _mk(i, m, M)
    return None


def convert(t):
    for a, b in PHRASES:
        t = t.replace(a, b)
    def rep(m):
        w, nxt = m.group(1), m.group(2)
        if nxt == "?" or (nxt and nxt in ",·" and w.endswith(QUESTION_END)):
            return m.group(0)
        e = _end(w)
        return (e + nxt) if e else m.group(0)
    return re.sub(r"([가-힣]+)요([^가-힣]|$)", rep, t)


def walk(x):
    if isinstance(x, str):
        return convert(x)
    if isinstance(x, list):
        return [walk(v) for v in x]
    if isinstance(x, dict):
        return {k: walk(v) for k, v in x.items()}
    return x
