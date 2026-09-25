# 월간 질문 개선 루틴

매월 1일 자동 실행되는 세션이 이 순서대로 진행한다. 사람이 직접 돌릴 때도 같다.

검토 페이지: https://claude.ai/artifact/LHDQVB4zuBeX8JkWjs79DJ
작업 브랜치: `claude/nadab-daily-question-stats-4o327c`

## 1. 준비
```bash
git fetch origin claude/nadab-daily-question-stats-4o327c
git checkout claude/nadab-daily-question-stats-4o327c && git pull
pip install -q pandas statsmodels scipy openpyxl
```

## 2. 데이터 수집과 분석
```bash
python analysis/fetch_data.py      # 통계 사이트 CSV → data/snapshots/, nadab_daily_question_stats.csv
python analysis/analyze.py
python analysis/improve.py
```
fetch가 실패하면(로그인 필요, 컬럼 변경) 멈추고 사용자에게 알린다.

## 3. 선택 결과 기록
ArtifactData `list`로 검토 페이지의 `decisions` 컬렉션을 `out_dir: output/decisions_dump`에 내려받고:
```bash
python analysis/record_decisions.py output/decisions_dump
```
'반영 누락 의심' 질문이 있으면 요약에 적는다.

## 4. 새 수정안 작성
```bash
python analysis/sync_review.py     # "needs rewrite:" 줄이 수정안이 필요한 질문
```
각 질문마다 `analysis/rewrites.csv`에 한 줄을 추가한다(이미 있는 ID면 그 줄을 고친다).
- 열: 질문 ID, 원인 진단, 수정 원칙, 수정안 1 (권장), 수정안 2 (대안)
- 규칙: 리포트의 '질문 작성 가이드'. 오늘 → 평소/주로, 경험 전제 → 성향형 또는 '없다면' 대안, 특정 활동 → 대상 확장, '누구' → 특징, 닫힌 질문 → 열린 질문, 40자 이내. 시점 없는 성향형·최애형을 우선한다.
- 원래 주제와 관심사는 유지한다. 관측 성과가 평균 이상이면 '(유지) 원문'으로 둔다.
- 작성 후 `python analysis/lint_questions.py "수정안1" "수정안2"`에서 [높음]이 없어야 한다.
- CSV는 pandas로 기록한다(쉼표 따옴표 처리).

그다음 `python analysis/improve.py && python analysis/sync_review.py`를 다시 돌려 `needsRewrite`가 0인지 확인한다.

## 5. 검토 페이지 갱신
`output/review_sync/batch.json`의 writes 배열을 그대로 ArtifactData `batch`로 보낸다.
chunks 수가 지난번보다 줄었으면 남는 `qchunks/c{n}` 문서는 delete로 지운다.
`decisions` 컬렉션은 절대 쓰지 않는다(사용자 선택).

## 6. 리포트·기록
```bash
python analysis/build_report.py
git add -A && git commit -m "Monthly question review: <날짜>" && git push -u origin claude/nadab-daily-question-stats-4o327c
```
리포트 아티팩트(https://claude.ai/artifact/QYjtnbNpKA6hMZ219aUX7J)는 `output/report.html`로 url을 지정해 다시 게시한다.

## 7. 요약 보고 (5줄 이내)
- 새로 A·B에 들어온 질문 수와 새 수정안 수
- 선택 완료·반영함·보류 개수, 반영 누락 의심
- 수정한 질문(Revision 2 이상)의 전후 답변율 중 표본 20회 이상인 것
- 전체 답변율·교체율 변화
