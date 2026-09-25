# 나답 데일리 질문 성과 분석

`nadab_daily_question_stats.csv`(Revision 1, 2026-08-27 적용분)를 분석해 수정이 필요한 질문을 찾고 수정안을 제시합니다.

## 실행
```bash
pip install pandas statsmodels scipy openpyxl
python analysis/analyze.py       # 스코어카드·엑셀·요약 생성
python analysis/improve.py       # 수정안 검수·효과 추정 (엑셀에 시트 추가)
python analysis/build_report.py  # HTML 리포트 생성

# 새 질문 검수
python analysis/lint_questions.py "새 질문 문구"
python analysis/lint_questions.py -f new_questions.txt
```

## 결과물 (`output/`)
- `question_review.xlsx`: 분류 요약 / A 교체·재작성 / B 패턴 수정(수정안 포함) / D 우수 레퍼런스 / 전체 스코어카드 / 세그먼트 / 원인 모델 / 수정안 효과 추정
- `rewrites_scored.csv`: 수정안별 조치 유형, 예측 답변율 변화, 잔여 위험
- `question_scorecard.csv`: 875개 질문별 추정 답변율, P(평균 미만), 위험 특성, 분류
- `report.html`: 분석 리포트

## 분석 파일 (`analysis/`)
- `features.py`: 문구 특성 추출 규칙
- `analyze.py`: 세그먼트 분석, 이항 GLM, Beta-Binomial 축소, 분류
- `rewrites.csv`: A·B그룹과 오류 질문 264개의 진단과 수정안 2개(직접 작성). `rw/`는 작성 원본
- `improve.py`: 수정안 자체 검수 + 원인 모델로 기대 효과 추정
- `lint_questions.py`: 새 질문 검수 도구
