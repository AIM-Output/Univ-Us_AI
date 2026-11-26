# F9 평가 — 학습 후

> 모델 `Qwen/Qwen3-4B` + 어댑터 `outputs/f9-r1-date/adapter` · 템플릿 모드 native · 2026-11-23T18:01:17 · 데이터 `data/f9/eval.jsonl` (315행 · 358턴)

| 구분 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 | 거절 정확도 | 텍스트 턴 정확도 | 턴 정확도 | 평균 지연(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 학습 후 | 358 | 100.0% | 95.9% | 96.9% | 100.0% | 100.0% | 96.6% | 6.30 |

## 턴 종류

| 종류 | 턴 수 | 정확도 |
| --- | ---: | ---: |
| 도구 호출 | 293 | 인자까지 95.9% |
| 거절·범위 밖 (첫 턴 텍스트) | 65 | 100.0% |
| 도구 결과 뒤 답변 | 0 | — |

## 도구별

| 도구 | 종류 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 |
| --- | --- | ---: | ---: | ---: | ---: |
| create_event | write | 46 | 100.0% | 93.5% | 97.4% |
| delete_event | write | 24 | 100.0% | 100.0% | 100.0% |
| get_attendance | read | 20 | 100.0% | 100.0% | 100.0% |
| get_exams | read | 11 | 100.0% | 100.0% | 100.0% |
| get_graduation | read | 11 | 100.0% | 100.0% | — |
| list_assignments | read | 20 | 100.0% | 100.0% | 100.0% |
| list_events | read | 69 | 100.0% | 97.1% | 97.1% |
| navigate | navigate | 39 | 100.0% | 100.0% | 100.0% |
| open_placement | run | 13 | 100.0% | 61.5% | 61.5% |
| run_sync | run | 21 | 100.0% | 100.0% | 100.0% |
| update_event | write | 19 | 100.0% | 89.5% | 94.7% |

## 실패 사례 (최대 10건 / 전체 실패 12턴)

- **f9-00145#t0** `up_B` [tool] — 사용자: 이번 주 토요일 팀 회의 시간을 오후 5시로 바꿔줘
  - 기대: `{"name": "list_events", "arguments": {"from": "2026-12-12", "to": "2026-12-12"}}`
  - 예측: `<tool_call>{"name": "list_events", "arguments": {"from": "2026-12-11", "to": "2026-12-11"}}</tool_call>`
- **f9-00178#t0** `pl_eval` [tool] — 사용자: 비는 시간에 공부 계획 넣어줘
  - 기대: `{"name": "open_placement", "arguments": {"range": "this_week"}}`
  - 예측: `<tool_call>{"name": "open_placement", "arguments": {}}</tool_call>`
- **f9-00381#t0** `pl_eval` [tool] — 사용자: 비는 시간에 공부 계획 넣어줘
  - 기대: `{"name": "open_placement", "arguments": {"range": "this_week"}}`
  - 예측: `<tool_call>{"name": "open_placement", "arguments": {}}</tool_call>`
- **f9-00482#t0** `ce_F` [tool] — 사용자: 4일 뒤 오전 11시 알바 일정 넣어
  - 기대: `{"name": "create_event", "arguments": {"title": "알바", "start": "2026-12-01T11:00", "end": "2026-12-01T12:00", "allDay": false, "category": "etc"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "etc", "end": "2026-11-31T12:00", "start": "2026-11-31T11:00", "title": "알바"}}</tool_call>`
- **f9-00776#t0** `pl_eval` [tool] — 사용자: 비는 시간에 공부 계획 넣어줘
  - 기대: `{"name": "open_placement", "arguments": {"range": "this_week"}}`
  - 예측: `<tool_call>{"name": "open_placement", "arguments": {}}</tool_call>`
- **f9-00823#t0** `pl_eval` [tool] — 사용자: 비는 시간에 공부 계획 넣어줘
  - 기대: `{"name": "open_placement", "arguments": {"range": "this_week"}}`
  - 예측: `<tool_call>{"name": "open_placement", "arguments": {}}</tool_call>`
- **f9-00844#t0** `del_B` [tool] — 사용자: 2일 뒤 친구 약속 일정 지워줘
  - 기대: `{"name": "list_events", "arguments": {"from": "2026-12-01", "to": "2026-12-01"}}`
  - 예측: `<tool_call>{"name": "list_events", "arguments": {"from": "2026-11-31", "to": "2026-11-31"}}</tool_call>`
- **f9-00970#t0** `pl_eval` [tool] — 사용자: 비는 시간에 공부 계획 넣어줘
  - 기대: `{"name": "open_placement", "arguments": {"range": "this_week"}}`
  - 예측: `<tool_call>{"name": "open_placement", "arguments": {}}</tool_call>`
- **f9-01177#t1** `up_B` [tool] — 사용자: 모레 동아리 모임 시간을 저녁 8시로 바꿔줘
  - 기대: `{"name": "update_event", "arguments": {"id": "ev:7773", "patch": {"start": "2026-10-04T20:00", "end": "2026-10-04T21:00"}}}`
  - 예측: `<tool_call>{"name": "update_event", "arguments": {"id": "ev:7773", "patch": {"end": "2026-10-04T21:00", "start": "2026-10-04T20:00", "title": "동아리 모임"}}}</tool_call>`
- **f9-01246#t1** `up_B` [tool] — 사용자: 오늘 스터디 시간을 저녁 7시 반로 바꿔줘
  - 기대: `{"name": "update_event", "arguments": {"id": "ev:6132", "patch": {"start": "2026-11-14T19:30", "end": "2026-11-14T20:30"}}}`
  - 예측: `<tool_call>{"name": "update_event", "arguments": {"id": "ev:6132", "patch": {"end": "2026-11-14T20:30", "start": "2026-11-14T19:30", "title": "스터디"}}}</tool_call>`
