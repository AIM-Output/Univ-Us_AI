# F9 평가 — f9-r1 real_dev new30

> 모델 `Qwen/Qwen3-4B` + 어댑터 `outputs/f9-r1/adapter` · 템플릿 모드 native · 2026-11-27T01:19:43 · 데이터 `outputs/f9-r1/real_dev_new30.jsonl` (30행 · 30턴)

| 구분 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 | 거절 정확도 | 텍스트 턴 정확도 | 턴 정확도 | 평균 지연(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f9-r1 real_dev new30 | 30 | 100.0% | 96.3% | 98.9% | 100.0% | 100.0% | 96.7% | 15.56 |

## 턴 종류

| 종류 | 턴 수 | 정확도 |
| --- | ---: | ---: |
| 도구 호출 | 27 | 인자까지 96.3% |
| 거절·범위 밖 (첫 턴 텍스트) | 3 | 100.0% |
| 도구 결과 뒤 답변 | 0 | — |

## 도구별

| 도구 | 종류 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 |
| --- | --- | ---: | ---: | ---: | ---: |
| create_event | write | 14 | 100.0% | 92.9% | 98.6% |
| get_attendance | read | 2 | 100.0% | 100.0% | 100.0% |
| get_exams | read | 1 | 100.0% | 100.0% | 100.0% |
| list_assignments | read | 1 | 100.0% | 100.0% | 100.0% |
| list_events | read | 7 | 100.0% | 100.0% | 100.0% |
| navigate | navigate | 1 | 100.0% | 100.0% | 100.0% |
| run_sync | run | 1 | 100.0% | 100.0% | 100.0% |
| (평가셋에 없음) delete_event, get_graduation, open_placement, update_event | — | — | — | — | — |

## 실패 사례 (최대 10건 / 전체 실패 1턴)

- **hr-003#t0** `real_hr` [tool] — 사용자: 담주 화요일 19시에 토익 스터디 2시간짜리로 잡아줘
  - 기대: `{"name": "create_event", "arguments": {"title": "토익 스터디", "start": "2026-11-24T19:00", "end": "2026-11-24T21:00", "allDay": false, "category": "study"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "study", "end": "2026-11-24T23:00", "start": "2026-11-24T19:00", "title": "토익 스터디"}}</tool_call>`
