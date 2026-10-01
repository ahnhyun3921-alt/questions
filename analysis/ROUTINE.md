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
python analysis/analyze_v2.py      # AI 태깅 모델·FDR·계층 베이즈·우선순위 (output/v2/)
```
`analyze_v2.py`가 "태그 누락"으로 멈추면 새 질문이 생긴 것이다. `analysis/TAGGING.md` 기준으로
`analysis/question_tags.csv`에 새 ID의 태그를 추가하고 다시 돌린다.
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

그다음 `python analysis/improve.py && python analysis/analyze_v2.py && python analysis/sync_review.py`를 다시 돌려 `needsRewrite`가 0인지 확인한다.

## 질문 ID 주의
통계 사이트의 `질문 ID`와 관리자 DB `id`는 다르다(통계 ID 661 이상은 DB id가 24 크다).
사용자가 `daily_questions_*.csv`(DB 원본)를 새로 주면 `data/`에 넣고, `analysis/db_questions.py`의
`check_mapping`이 빈 결과인지 확인한다. 어긋나면 규칙이 바뀐 것이니 문구로 다시 맞춘다.
문구가 바뀌는 질문은 가이드 3종(empathy·hint·leading)도 새 문구에 맞게 `analysis/guides_rewrite.csv`에 적는다.

## 5. 검토 페이지 갱신
`output/review_sync/batch.json`의 writes 배열을 그대로 ArtifactData `batch`로 보낸다.
chunks 수가 지난번보다 줄었으면 남는 `qchunks/c{n}` 문서는 delete로 지운다.
`decisions` 컬렉션은 절대 쓰지 않는다(사용자 선택).

## 6. 리포트·기록
```bash
python analysis/build_final_list.py
python analysis/build_db_export.py   # output/db_export/: DB id 기준 수정·신규·비활성화 파일
python analysis/build_full_db.py     # output/daily_questions_revised.csv: DB 원본 형식 전체본
python analysis/build_report.py
git add -A && git commit -m "Monthly question review: <날짜>" && git push -u origin claude/nadab-daily-question-stats-4o327c
```
리포트 아티팩트(https://claude.ai/artifact/QYjtnbNpKA6hMZ219aUX7J)는 `output/report.html`로 url을 지정해 다시 게시한다.

## 7. 요약 보고 (5줄 이내)
- 새로 A·B에 들어온 질문 수와 새 수정안 수
- 선택 완료·반영함·보류 개수, 반영 누락 의심
- 수정한 질문(Revision 2 이상)의 전후 답변율 중 표본 20회 이상인 것
- 전체 답변율·교체율 변화, v2 교차검증에서 가장 좋은 모델과 FDR 통과 특성이 바뀌었는지

## 8. 질문 개선함 페이지 (BE 전달용)
https://claude.ai/artifact/1F4W6W5WHQsnM6YsXUm4ea
```bash
python analysis/record_decisions.py <decisions 덤프 폴더>   # 검토 페이지 선택 먼저 반영
python analysis/build_final_list.py && python analysis/build_db_export.py && python analysis/build_full_db.py
python analysis/build_board_stats.py && python analysis/build_board_data.py && python analysis/build_insights.py && python analysis/build_board_page.py
```
`board/dist/index.html`(나눔스퀘어를 넣어 만든 페이지)을 `url`과 `files: {"data/questions.json": "board/data/questions.json", "data/insights.json": "board/data/insights.json"}`로 다시 게시한다(capabilities는 생략해 유지).
검토 페이지에서 수정안 1이 아닌 문구를 고른 질문은 `analysis/guides_decided.csv`에 그 문구에 맞는 가이드를 먼저 쓴다(문구가 다르면 적용되지 않는다).
신규 질문 DB id는 `analysis/new/new_db_ids.csv`에 고정돼 있고, 새 신규 질문만 다음 번호를 받는다.
AI 추천은 db `ai/{id}`에 쌓인다.
페이지에서 고친 내용은 db `edits/{key}`(key = DB id 또는 신규 ID)에, 다운로드 기록은 `downloads`에 쌓인다. 둘 다 사용자 데이터라 루틴이 쓰지 않는다.
`edits`는 ArtifactData `list`로 읽어 다음 개선본에 반영할지 사용자에게 보고한다.

## 9. 월간 통계 올리기 (개선함 페이지의 '데이터 근거')

- 매달 한 번, 2단계에서 통계를 받은 뒤에 해요. 따로 매일 돌리는 루틴은 없어요.
- `python analysis/build_board_stats.py`를 돌린 다음, ArtifactData로 개선함 db에 이번 달 스냅숏 날짜의 `stats/<날짜>` 문서 하나만 set 해요. 파일은 `output/board_stats/<날짜>.json`이에요.
- 스냅숏(`data/snapshots`)과 `output/board_stats`는 함께 커밋해요.
- 페이지는 `stats` 컬렉션 최근 90개를 읽어 KPI, 추이, 교체 많은 질문, 질문별 근거를 그려요.
- 보기 탭:
  - **고쳐야 할 것**: 문구를 그대로 두는 질문 가운데 노출이 5회 이상이고, 추정 답변율이 평균보다 10%p 넘게 낮은데 아직 수정안이 없는 질문. 매달 10단계에서 '골라 주세요'로 올라가요.
  - **신규 질문**, **변경된 질문**, **비활성화**.
- 맨 위 '골라 주세요' 탭은 두 가지를 모아 보여 줘요.
  - 월간 점검 추천(`picks`).
  - 수정안이 여러 개인데 아직 안 고른 질문.
- 루틴은 `edits`, `downloads`, `ai`, `decisions`를 쓰지 않아요.

## 10. 월간 '골라 주세요' 올리기 (개선함 페이지)

매달 1~9단계 다음에 해요.

1. ArtifactData `list`로 개선함 db의 `keeps` 컬렉션을 읽어 `output/monthly_picks/keeps.json`에 저장해요. 문서마다 doc_id와 data를 남겨요.
2. `python analysis/monthly_picks.py candidates --month <YYYY-MM> --keeps output/monthly_picks/keeps.json`
   - 노출 5회 이상이고, 추정 답변율이 평균보다 10%p 넘게 낮은 질문을 뽑아요.
   - 개선본에 이미 수정안이 있는 질문은 빼요.
   - 90일 안에 '그대로 두기'로 한 질문도 빼요.
3. `output/monthly_picks/<YYYY-MM>.json`의 각 item `opts`에 수정안을 2~3개 직접 써요.
   - 형식은 `{text, empathy, hint, leading, why}`예요.
   - 4단계 수정 원칙을 따르고, `lint_questions.py`에서 [높음]이 없어야 해요.
   - 가이드는 기존 DB 말투로 써요.
4. `python analysis/monthly_picks.py batch --month <YYYY-MM>`를 돌린 다음, ArtifactData `batch`로 `output/monthly_picks/<YYYY-MM>_batch.json`을 개선함 db에 써요.
   - 쓰는 곳은 `picks/<YYYY-MM>-<DB id>`예요.
5. 쓰는 컬렉션은 `picks`뿐이에요. `keeps`는 읽기만 해요. `edits`, `downloads`, `ai`, `stats`(9단계 몫)는 이 단계에서 건드리지 않아요.

페이지에서는 이렇게 처리돼요.
- 질문을 저장하거나 '지금 문구 그대로 둘게요'를 누르면 그 달 추천은 끝난 것으로 쳐요.
  - 저장은 `edits`에, 그대로 두기는 `keeps`에 기록돼요.
- 같은 질문의 추천은 가장 최근 달 것만 보여요.

## 11. 중복 후보 판정 (개선함 '중복 검토' 탭)

- `build_board_data.py`가 활성 질문끼리 중복 후보를 뽑아요. 기준은 `dedupe.pairs`(글자 n-gram과 핵심어 겹침)이고, 결과는 `questions.json`의 `dups`에 들어가요.
- 아직 판정하지 않은 쌍은 실행 로그에 '판정 전 N'으로 나와요.
- 판정 전인 쌍은 `analysis/dup_labels.csv`에 판정을 적고 다시 빌드해요.
  - 열: `a, b, 판정, 메모, a 문구, b 문구`
  - `dup`: 같은 질문이라 하나만 남기길 권함
  - `similar`: 답이 겹치기 쉬움
  - `diff`: 형식만 비슷하고 묻는 게 다름. 페이지에 보이지 않아요.
- 페이지 쪽 처리:
  - '중복 아니에요'를 누르면 `dupok/<작은id>-<큰id>`에 기록돼요.
  - 비활성화는 `edits`에 `deleted`로 기록돼요.
  - 루틴은 이 두 컬렉션을 쓰지 않아요.
- 신규 질문을 페이지에서 끄면 '삭제'로 쳐서 다운로드 파일에 넣지 않아요. 단, 이미 다운로드 기록에 들어간 신규 질문이면 `deleted_at`을 채워서 남겨요.

## 12. BE가 올린 최신본 (개선함 '다운로드' 탭)

- BE가 CSV를 올리면 개선함 db에 저장돼요.
  - 파일 내용: `uploads/<id>_<n>`. 250행씩 나눠 담아요.
  - 기록: `uplog/<id>`
  - 지금 최신본 표시: `upmeta/current`
- 페이지는 `questions.json`(개선본) 위에 이 파일 내용을 덮어써서 보여 줘요. 그 위에 올린 뒤에 고친 `edits`만 더해요.
- 그래서 8단계에서 개선함을 다시 게시해도 BE의 최신본은 그대로 유지돼요.
- 루틴은 `uploads`, `uplog`, `upmeta`를 쓰지 않아요.
- 다음 달 분석의 기준이 될 최신 DB가 필요하면 ArtifactData로 `upmeta/current`와 `uploads`를 읽어서 쓰면 돼요.

## 13. 분석 탭 (`build_insights.py`)

- 최신 통계로 다음을 다시 계산해요.
  - 요인별 단변량 비교
  - 다변량 로지스틱 회귀: 문항 단위 이항 모델이에요. 과산포가 있으면 신뢰구간을 넓히고, 척도가 1보다 작으면 1로 둬요.
  - 길이 구간별 답변율 (Wilson 95% CI)
  - 축별 문항 수와 답변율
- 결과에서 '핵심 인사이트' 문장을 만들어요. '확실'이라고 쓰는 건 95% 신뢰구간이 0을 넘지 않을 때만이에요.
- 7단계 요약에 인사이트 중 바뀐 것을 한 줄 넣어요. 예: "경험 전제 효과 -7%p → -5%p".
