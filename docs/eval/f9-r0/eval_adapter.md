# F9 평가 — 학습 후

> 모델 `Qwen/Qwen3-4B` + 어댑터 `outputs/f9-r0/adapter` · 템플릿 모드 native · 2026-11-22T21:39:14 · 데이터 `data/f9/eval.jsonl` (315행 · 358턴)

| 구분 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 | 거절 정확도 | 텍스트 턴 정확도 | 턴 정확도 | 평균 지연(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 학습 후 | 358 | 95.6% | 94.2% | 96.2% | 4.6% | 4.6% | 77.9% | 6.07 |

## 턴 종류

| 종류 | 턴 수 | 정확도 |
| --- | ---: | ---: |
| 도구 호출 | 293 | 인자까지 94.2% |
| 거절·범위 밖 (첫 턴 텍스트) | 65 | 4.6% |
| 도구 결과 뒤 답변 | 0 | — |

## 도구별

| 도구 | 종류 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 |
| --- | --- | ---: | ---: | ---: | ---: |
| create_event | write | 46 | 100.0% | 95.7% | 98.3% |
| delete_event | write | 24 | 100.0% | 100.0% | 100.0% |
| get_attendance | read | 20 | 100.0% | 100.0% | 100.0% |
| get_exams | read | 11 | 100.0% | 100.0% | 100.0% |
| get_graduation | read | 11 | 100.0% | 100.0% | — |
| list_assignments | read | 20 | 100.0% | 100.0% | 100.0% |
| list_events | read | 69 | 100.0% | 97.1% | 97.1% |
| navigate | navigate | 39 | 100.0% | 100.0% | 100.0% |
| open_placement | run | 13 | 0.0% | 0.0% | 0.0% |
| run_sync | run | 21 | 100.0% | 100.0% | 100.0% |
| update_event | write | 19 | 100.0% | 100.0% | 100.0% |

## 실패 사례 (최대 10건 / 전체 실패 79턴)

- **f9-00026#t0** `ref_class` [refusal] — 사용자: 선형대수 수업 밤 9시로 옮겨줘
  - 기대: `학사일정·수업은 바꿀 수 없습니다. 학교에서 가져온 일정이라 원본을 따릅니다.`
  - 예측: `<tool_call>{"name": "list_events", "arguments": {"from": "2026-10-22", "to": "2026-10-22"}}</tool_call>`
- **f9-00049#t0** `ref_class` [refusal] — 사용자: 다음 주 수강신청 기간 삭제해줘
  - 기대: `학사일정·수업은 바꿀 수 없습니다. 학교에서 가져온 일정이라 원본을 따릅니다.`
  - 예측: `<tool_call>{"name": "list_events", "arguments": {"from": "2026-10-05", "to": "2026-10-11"}}</tool_call>`
- **f9-00061#t0** `ref_class` [refusal] — 사용자: 다음 주 수강신청 기간 삭제해줘
  - 기대: `학사일정·수업은 바꿀 수 없습니다. 학교에서 가져온 일정이라 원본을 따릅니다.`
  - 예측: `<tool_call>{"name": "list_events", "arguments": {"from": "2026-10-12", "to": "2026-10-18"}}</tool_call>`
- **f9-00082#t0** `ref_class` [refusal] — 사용자: 다음 주 수강신청 기간 삭제해줘
  - 기대: `학사일정·수업은 바꿀 수 없습니다. 학교에서 가져온 일정이라 원본을 따릅니다.`
  - 예측: `<tool_call>{"name": "list_events", "arguments": {"from": "2026-12-21", "to": "2026-12-27"}}</tool_call>`
- **f9-00083#t0** `pl_eval` [tool] — 사용자: 이번 주 빈 시간 채워줘
  - 기대: `{"name": "open_placement", "arguments": {"range": "this_week"}}`
  - 예측: `<tool_call>{"name": "list_events", "arguments": {"from": "2026-11-23", "to": "2026-11-29"}}</tool_call>`
- **f9-00095#t0** `ref_class` [refusal] — 사용자: 다음 주 수강신청 기간 삭제해줘
  - 기대: `학사일정·수업은 바꿀 수 없습니다. 학교에서 가져온 일정이라 원본을 따릅니다.`
  - 예측: `<tool_call>{"name": "list_events", "arguments": {"from": "2026-11-02", "to": "2026-11-08"}}</tool_call>`
- **f9-00113#t0** `pl_eval` [tool] — 사용자: 이번 주 빈 시간 채워줘
  - 기대: `{"name": "open_placement", "arguments": {"range": "this_week"}}`
  - 예측: `<tool_call>{"name": "list_events", "arguments": {"from": "2026-11-16", "to": "2026-11-22"}}</tool_call>`
- **f9-00137#t0** `ref_class` [refusal] — 사용자: 데이터베이스 수업 일정 지워줘
  - 기대: `학사일정·수업은 바꿀 수 없습니다. 학교에서 가져온 일정이라 원본을 따릅니다.`
  - 예측: `<tool_call>{"name": "list_events", "arguments": {"from": "2026-12-17", "to": "2026-12-17"}}</tool_call>`
- **f9-00140#t0** `ref_class` [refusal] — 사용자: 데이터베이스 수업 밤 10시로 옮겨줘
  - 기대: `학사일정·수업은 바꿀 수 없습니다. 학교에서 가져온 일정이라 원본을 따릅니다.`
  - 예측: `<tool_call>{"name": "list_events", "arguments": {"from": "2026-12-13", "to": "2026-12-13"}}</tool_call>`
- **f9-00142#t0** `ref_class` [refusal] — 사용자: 다음 주 수강신청 기간 삭제해줘
  - 기대: `학사일정·수업은 바꿀 수 없습니다. 학교에서 가져온 일정이라 원본을 따릅니다.`
  - 예측: `<tool_call>{"name": "list_events", "arguments": {"from": "2026-10-26", "to": "2026-11-01"}}</tool_call>`
