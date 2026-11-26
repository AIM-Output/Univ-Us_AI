# F9 평가 — 학습 후

> 모델 `Qwen/Qwen3-4B` + 어댑터 `outputs/f9-r1/adapter` · 템플릿 모드 native · 2026-11-23T10:53:36 · 데이터 `data/f9/eval.jsonl` (315행 · 358턴)

| 구분 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 | 거절 정확도 | 텍스트 턴 정확도 | 턴 정확도 | 평균 지연(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 학습 후 | 358 | 100.0% | 97.6% | 98.4% | 100.0% | 100.0% | 98.0% | 5.23 |

## 턴 종류

| 종류 | 턴 수 | 정확도 |
| --- | ---: | ---: |
| 도구 호출 | 293 | 인자까지 97.6% |
| 거절·범위 밖 (첫 턴 텍스트) | 65 | 100.0% |
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
| list_events | read | 69 | 100.0% | 100.0% | 100.0% |
| navigate | navigate | 39 | 100.0% | 100.0% | 100.0% |
| open_placement | run | 13 | 100.0% | 61.5% | 61.5% |
| run_sync | run | 21 | 100.0% | 100.0% | 100.0% |
| update_event | write | 19 | 100.0% | 100.0% | 100.0% |

## 실패 사례 (최대 10건 / 전체 실패 7턴)

- **f9-00178#t0** `pl_eval` [tool] — 사용자: 비는 시간에 공부 계획 넣어줘
  - 기대: `{"name": "open_placement", "arguments": {"range": "this_week"}}`
  - 예측: `<tool_call>{"name": "open_placement", "arguments": {}}</tool_call>`
- **f9-00381#t0** `pl_eval` [tool] — 사용자: 비는 시간에 공부 계획 넣어줘
  - 기대: `{"name": "open_placement", "arguments": {"range": "this_week"}}`
  - 예측: `<tool_call>{"name": "open_placement", "arguments": {}}</tool_call>`
- **f9-00776#t0** `pl_eval` [tool] — 사용자: 비는 시간에 공부 계획 넣어줘
  - 기대: `{"name": "open_placement", "arguments": {"range": "this_week"}}`
  - 예측: `<tool_call>{"name": "open_placement", "arguments": {}}</tool_call>`
- **f9-00823#t0** `pl_eval` [tool] — 사용자: 비는 시간에 공부 계획 넣어줘
  - 기대: `{"name": "open_placement", "arguments": {"range": "this_week"}}`
  - 예측: `<tool_call>{"name": "open_placement", "arguments": {}}</tool_call>`
- **f9-00889#t0** `ce_F` [tool] — 사용자: 9일 뒤 오전 10시 멘토링 일정 넣어 1시간
  - 기대: `{"name": "create_event", "arguments": {"title": "멘토링", "start": "2026-10-31T10:00", "end": "2026-10-31T11:00", "allDay": false, "category": "etc"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "etc", "end": "2026-11-01T11:00", "start": "2026-11-01T10:00", "title": "멘토링"}}</tool_call>`
- **f9-00970#t0** `pl_eval` [tool] — 사용자: 비는 시간에 공부 계획 넣어줘
  - 기대: `{"name": "open_placement", "arguments": {"range": "this_week"}}`
  - 예측: `<tool_call>{"name": "open_placement", "arguments": {}}</tool_call>`
- **f9-01350#t0** `ce_F` [tool] — 사용자: 다음 주 토요일 저녁 8시 멘토링 일정 넣어
  - 기대: `{"name": "create_event", "arguments": {"title": "멘토링", "start": "2026-10-31T20:00", "end": "2026-10-31T21:00", "allDay": false, "category": "etc"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "etc", "end": "2026-11-07T21:00", "start": "2026-11-07T20:00", "title": "멘토링"}}</tool_call>`
