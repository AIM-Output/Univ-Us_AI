# F9 평가 — f9-r1-date probe

> 모델 `Qwen/Qwen3-4B` + 어댑터 `outputs/f9-r1-date/adapter` · 템플릿 모드 native · 2026-11-23T18:29:48 · 데이터 `outputs/date_probe/probe_dates.jsonl` (180행 · 180턴)

| 구분 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 | 거절 정확도 | 텍스트 턴 정확도 | 턴 정확도 | 평균 지연(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f9-r1-date probe | 180 | 100.0% | 72.8% | 89.1% | — | — | 72.8% | 9.41 |

## 턴 종류

| 종류 | 턴 수 | 정확도 |
| --- | ---: | ---: |
| 도구 호출 | 180 | 인자까지 72.8% |
| 거절·범위 밖 (첫 턴 텍스트) | 0 | — |
| 도구 결과 뒤 답변 | 0 | — |

## 도구별

| 도구 | 종류 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 |
| --- | --- | ---: | ---: | ---: | ---: |
| create_event | write | 180 | 100.0% | 72.8% | 89.1% |
| (평가셋에 없음) delete_event, get_attendance, get_exams, get_graduation, list_assignments, list_events, navigate, open_placement, run_sync, update_event | — | — | — | — | — |

## 실패 사례 (최대 10건 / 전체 실패 49턴)

- **probe-0009#t0** `n_days|d31|wd3|+13` [tool] — 사용자: 컴퓨터구조 예습 13일 뒤 오후 1시 반 있음 저장해줘
  - 기대: `{"name": "create_event", "arguments": {"title": "컴퓨터구조 예습", "start": "2027-03-31T13:30", "end": "2027-03-31T14:30", "allDay": false, "category": "study"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "study", "end": "2027-04-01T14:30", "start": "2027-04-01T13:30", "title": "컴퓨터구조 예습"}}</tool_call>`
- **probe-0011#t0** `n_days|cross|wd1|+5` [tool] — 사용자: 5일 뒤 일정 추가할게 오전 10시 영어회화 시험 공부
  - 기대: `{"name": "create_event", "arguments": {"title": "영어회화 시험 공부", "start": "2027-01-03T10:00", "end": "2027-01-03T11:00", "allDay": false, "category": "study"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "study", "end": "2026-01-03T11:00", "start": "2026-01-03T10:00", "title": "영어회화 시험 공부"}}</tool_call>`
- **probe-0012#t0** `n_days|d31|wd6|+7` [tool] — 사용자: 7일 후 저녁 여덟 시 반 인턴 면접 잡아줘 3시간
  - 기대: `{"name": "create_event", "arguments": {"title": "인턴 면접", "start": "2027-01-31T20:30", "end": "2027-01-31T23:30", "allDay": false, "category": "etc"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "etc", "end": "2027-02-01T23:30", "start": "2027-02-01T20:30", "title": "인턴 면접"}}</tool_call>`
- **probe-0019#t0** `n_days|d31|wd6|+6` [tool] — 사용자: 6일 뒤 정오에 미시경제학 팀플 회의 좀 넣어주실래요?
  - 기대: `{"name": "create_event", "arguments": {"title": "미시경제학 팀플 회의", "start": "2027-07-31T12:00", "end": "2027-07-31T13:00", "allDay": false, "category": "team"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "team", "end": "2027-08-01T13:00", "start": "2027-08-01T12:00", "title": "미시경제학 팀플 회의"}}</tool_call>`
- **probe-0020#t0** `next_wd|d31|wd5|+7` [tool] — 사용자: 담주 토요일 오전 9시 반 축구 스케줄 등록
  - 기대: `{"name": "create_event", "arguments": {"title": "축구", "start": "2027-07-31T09:30", "end": "2027-07-31T10:30", "allDay": false, "category": "personal"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "personal", "end": "2027-08-01T10:30", "start": "2027-08-01T09:30", "title": "축구"}}</tool_call>`
- **probe-0021#t0** `next_wd|d31|wd4|+5` [tool] — 사용자: 프로그래밍언어론 예습 넣어줘. 날짜는 다음 주 수요일이고 시간은 오후 5시 반
  - 기대: `{"name": "create_event", "arguments": {"title": "프로그래밍언어론 예습", "start": "2027-03-31T17:30", "end": "2027-03-31T18:30", "allDay": false, "category": "study"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "study", "end": "2027-04-01T18:30", "start": "2027-04-01T17:30", "title": "프로그래밍언어론 예습"}}</tool_call>`
- **probe-0022#t0** `next_wd|d31|wd5|+4` [tool] — 사용자: 다음 주 수요일 오전 10시 반 축구, 3시간
  - 기대: `{"name": "create_event", "arguments": {"title": "축구", "start": "2027-03-31T10:30", "end": "2027-03-31T13:30", "allDay": false, "category": "personal"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "personal", "end": "2027-04-01T13:30", "start": "2027-04-01T10:30", "title": "축구"}}</tool_call>`
- **probe-0031#t0** `next_wd|d31|wd5|+4` [tool] — 사용자: 스터디 추가해줘 다음 주 수요일 오전 9시
  - 기대: `{"name": "create_event", "arguments": {"title": "스터디", "start": "2027-03-31T09:00", "end": "2027-03-31T10:00", "allDay": false, "category": "study"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "study", "end": "2027-04-01T10:00", "start": "2027-04-01T09:00", "title": "스터디"}}</tool_call>`
- **probe-0032#t0** `n_days|cross|wd0|+14` [tool] — 사용자: 코딩 테스트 준비 14일 뒤 저녁 6시 반에 있음 저장해줘
  - 기대: `{"name": "create_event", "arguments": {"title": "코딩 테스트 준비", "start": "2027-08-09T18:30", "end": "2027-08-09T19:30", "allDay": false, "category": "study"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "study", "end": "2027-08-10T17:30", "start": "2027-08-10T16:30", "title": "코딩 테스트 준비"}}</tool_call>`
- **probe-0034#t0** `n_days|cross|wd3|+5` [tool] — 사용자: 5일 뒤 노트 정리 밤 9시 시작 2시간 반
  - 기대: `{"name": "create_event", "arguments": {"title": "노트 정리", "start": "2027-03-02T21:00", "end": "2027-03-02T23:30", "allDay": false, "category": "study"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "study", "end": "2027-03-01T23:30", "start": "2027-03-01T21:00", "title": "노트 정리"}}</tool_call>`
