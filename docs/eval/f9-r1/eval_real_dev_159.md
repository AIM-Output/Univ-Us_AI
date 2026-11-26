# F9 평가 — f9-r1 real_dev

> 모델 `Qwen/Qwen3-4B` + 어댑터 `outputs/f9-r1/adapter` · 템플릿 모드 native · 2026-11-27T00:15:59 · 데이터 `data/f9/real_dev.jsonl` (159행 · 159턴)

| 구분 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 | 거절 정확도 | 텍스트 턴 정확도 | 턴 정확도 | 평균 지연(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f9-r1 real_dev | 159 | 99.3% | 92.8% | 96.1% | 100.0% | 100.0% | 93.7% | 9.03 |

## 턴 종류

| 종류 | 턴 수 | 정확도 |
| --- | ---: | ---: |
| 도구 호출 | 139 | 인자까지 92.8% |
| 거절·범위 밖 (첫 턴 텍스트) | 20 | 100.0% |
| 도구 결과 뒤 답변 | 0 | — |

## 도구별

| 도구 | 종류 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 |
| --- | --- | ---: | ---: | ---: | ---: |
| create_event | write | 39 | 100.0% | 82.1% | 95.5% |
| get_attendance | read | 9 | 100.0% | 100.0% | 100.0% |
| get_exams | read | 11 | 90.9% | 90.9% | 100.0% |
| get_graduation | read | 11 | 100.0% | 100.0% | — |
| list_assignments | read | 7 | 100.0% | 100.0% | 100.0% |
| list_events | read | 39 | 100.0% | 94.9% | 95.6% |
| navigate | navigate | 11 | 100.0% | 100.0% | 100.0% |
| open_placement | run | 8 | 100.0% | 100.0% | 100.0% |
| run_sync | run | 4 | 100.0% | 100.0% | 100.0% |
| (평가셋에 없음) delete_event, update_event | — | — | — | — | — |

## 실패 사례 (최대 10건 / 전체 실패 10턴)

- **ju-003#t0** `real_ju` [tool] — 사용자: 매주 월요일 아침 9시 반 알고리즘 스터디 반복으로 넣어줘
  - 기대: `{"name": "create_event", "arguments": {"title": "알고리즘 스터디", "start": "2026-11-23T09:30", "end": "2026-11-23T10:30", "allDay": false, "category": "study", "repeat": "weekly:MO"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "study", "end": "2026-11-16T10:30", "repeat": "weekly:MO", "start": "2026-11-16T09:30", "title": "알고리즘 스터디"}}</tool_call>`
- **ju-042#t0** `real_ju` [tool] — 사용자: 다다음 주 화요일 아침 열 시에 학과 세미나 있어
  - 기대: `{"name": "create_event", "arguments": {"title": "학과 세미나", "start": "2026-12-01T10:00", "end": "2026-12-01T11:00", "allDay": false, "category": "study"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "study", "end": "2026-11-30T11:00", "start": "2026-11-30T10:00", "title": "학과 세미나"}}</tool_call>`
- **ju-045#t0** `real_ju` [tool] — 사용자: 다다음 주 시험 뭐 있어?
  - 기대: `{"name": "list_events", "arguments": {"from": "2026-11-30", "to": "2026-12-06", "kinds": ["exam"]}}`
  - 예측: `<tool_call>{"name": "list_events", "arguments": {"from": "2026-12-07", "kinds": ["exam"], "to": "2026-12-13"}}</tool_call>`
- **ju-051#t0** `real_ju` [tool] — 사용자: 담주 화요일 점심 12시에 동아리 선배랑 밥 약속
  - 기대: `{"name": "create_event", "arguments": {"title": "밥 약속", "start": "2026-11-24T12:00", "end": "2026-11-24T13:00", "allDay": false, "category": "personal"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "personal", "end": "2026-11-24T13:00", "start": "2026-11-24T12:00", "title": "동아리 선배랑 밥 약속"}}</tool_call>`
- **ju-053#t0** `real_ju` [tool] — 사용자: 11/27 하루 종일 전공 박람회 간다
  - 기대: `{"name": "create_event", "arguments": {"title": "전공 박람회", "start": "2026-11-27", "end": "2026-11-28", "allDay": true, "category": "etc"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": true, "category": "study", "end": "2026-11-28", "start": "2026-11-27", "title": "전공 박람회"}}</tool_call>`
- **ju-063#t0** `real_ju` [tool] — 사용자: 다음 주 토요일은 종일 학과 MT야
  - 기대: `{"name": "create_event", "arguments": {"title": "학과 MT", "start": "2026-11-28", "end": "2026-11-29", "allDay": true, "category": "personal"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": true, "category": "study", "end": "2026-11-29", "start": "2026-11-28", "title": "학과 MT"}}</tool_call>`
- **ju-127#t0** `real_ju` [tool] — 사용자: 다다음 주 마감 뭐 있어?
  - 기대: `{"name": "list_events", "arguments": {"from": "2026-11-30", "to": "2026-12-06", "kinds": ["deadline"]}}`
  - 예측: `<tool_call>{"name": "list_events", "arguments": {"from": "2026-12-07", "kinds": ["deadline"], "to": "2026-12-13"}}</tool_call>`
- **ju-134#t0** `real_ju` [tool] — 사용자: 다음 주 월요일부터 수요일까지 종일 학회 출장
  - 기대: `{"name": "create_event", "arguments": {"title": "학회 출장", "start": "2026-11-23", "end": "2026-11-26", "allDay": true, "category": "study"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": true, "category": "study", "end": "2026-11-26", "start": "2026-11-24", "title": "학회 출장"}}</tool_call>`
- **ju-151#t0** `real_ju` [tool] — 사용자: 내일 점심 12시 학과 사무실 방문
  - 기대: `{"name": "create_event", "arguments": {"title": "학과 사무실 방문", "start": "2026-11-23T12:00", "end": "2026-11-23T13:00", "allDay": false, "category": "personal"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "study", "end": "2026-11-23T13:00", "start": "2026-11-23T12:00", "title": "학과 사무실 방문"}}</tool_call>`
- **sb-063#t0** `real_sb` [tool] — 사용자: 시험 공부 플랜 어떻게 되고 있어?
  - 기대: `{"name": "get_exams", "arguments": {}}`
  - 예측: `<tool_call>{"name": "open_placement", "arguments": {}}</tool_call>`
