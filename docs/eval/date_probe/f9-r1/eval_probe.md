# F9 평가 — r1 probe

> 모델 `Qwen/Qwen3-4B` + 어댑터 `outputs/f9-r1/adapter` · 템플릿 모드 native · 2026-11-23T11:58:08 · 데이터 `outputs/date_probe/probe_dates.jsonl` (180행 · 180턴)

| 구분 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 | 거절 정확도 | 텍스트 턴 정확도 | 턴 정확도 | 평균 지연(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 probe | 180 | 100.0% | 76.7% | 90.8% | — | — | 76.7% | 9.59 |

## 턴 종류

| 종류 | 턴 수 | 정확도 |
| --- | ---: | ---: |
| 도구 호출 | 180 | 인자까지 76.7% |
| 거절·범위 밖 (첫 턴 텍스트) | 0 | — |
| 도구 결과 뒤 답변 | 0 | — |

## 도구별

| 도구 | 종류 | 턴 n | 도구 선택 | 인자 일치 | 필드 정확도 |
| --- | --- | ---: | ---: | ---: | ---: |
| create_event | write | 180 | 100.0% | 76.7% | 90.8% |
| (평가셋에 없음) delete_event, get_attendance, get_exams, get_graduation, list_assignments, list_events, navigate, open_placement, run_sync, update_event | — | — | — | — | — |

## 실패 사례 (최대 10건 / 전체 실패 42턴)

- **probe-0006#t0** `next_wd|d01|wd5|+8` [tool] — 사용자: 일정 하나 추가: 담주 일요일 15시 시험 공부
  - 기대: `{"name": "create_event", "arguments": {"title": "시험 공부", "start": "2026-11-01T15:00", "end": "2026-11-01T16:00", "allDay": false, "category": "study"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "study", "end": "2026-10-25T16:00", "start": "2026-10-25T15:00", "title": "시험 공부"}}</tool_call>`
- **probe-0012#t0** `n_days|d31|wd6|+7` [tool] — 사용자: 7일 후 저녁 여덟 시 반 인턴 면접 잡아줘 3시간
  - 기대: `{"name": "create_event", "arguments": {"title": "인턴 면접", "start": "2027-01-31T20:30", "end": "2027-01-31T23:30", "allDay": false, "category": "etc"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "etc", "end": "2027-02-01T23:30", "start": "2027-02-01T20:30", "title": "인턴 면접"}}</tool_call>`
- **probe-0013#t0** `n_days|d30of31|wd6|+6` [tool] — 사용자: 나 6일 후 오후 5시 30분에 영화 보기 있어. 추가해줘
  - 기대: `{"name": "create_event", "arguments": {"title": "영화 보기", "start": "2027-01-30T17:30", "end": "2027-01-30T18:30", "allDay": false, "category": "personal"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "personal", "end": "2027-02-01T18:30", "start": "2027-02-01T17:30", "title": "영화 보기"}}</tool_call>`
- **probe-0021#t0** `next_wd|d31|wd4|+5` [tool] — 사용자: 프로그래밍언어론 예습 넣어줘. 날짜는 다음 주 수요일이고 시간은 오후 5시 반
  - 기대: `{"name": "create_event", "arguments": {"title": "프로그래밍언어론 예습", "start": "2027-03-31T17:30", "end": "2027-03-31T18:30", "allDay": false, "category": "study"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "study", "end": "2027-04-07T18:30", "start": "2027-04-07T17:30", "title": "프로그래밍언어론 예습"}}</tool_call>`
- **probe-0022#t0** `next_wd|d31|wd5|+4` [tool] — 사용자: 다음 주 수요일 오전 10시 반 축구, 3시간
  - 기대: `{"name": "create_event", "arguments": {"title": "축구", "start": "2027-03-31T10:30", "end": "2027-03-31T13:30", "allDay": false, "category": "personal"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "personal", "end": "2027-04-07T13:30", "start": "2027-04-07T10:30", "title": "축구"}}</tool_call>`
- **probe-0026#t0** `n_days|end30|wd4|+5` [tool] — 사용자: 음 5일 뒤 아침 7시에 데이터마이닝 스터디 박아줘
  - 기대: `{"name": "create_event", "arguments": {"title": "데이터마이닝 스터디", "start": "2027-06-30T07:00", "end": "2027-06-30T08:00", "allDay": false, "category": "study"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "study", "end": "2027-07-01T08:00", "start": "2027-07-01T07:00", "title": "데이터마이닝 스터디"}}</tool_call>`
- **probe-0028#t0** `n_days|d30of31|wd5|+4` [tool] — 사용자: 4일 뒤 낮 2시에 콘서트가 생겼어 일정에 넣어줘
  - 기대: `{"name": "create_event", "arguments": {"title": "콘서트", "start": "2026-12-30T14:00", "end": "2026-12-30T15:00", "allDay": false, "category": "personal"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "personal", "end": "2027-01-01T15:00", "start": "2027-01-01T14:00", "title": "콘서트"}}</tool_call>`
- **probe-0029#t0** `n_days|end30|wd0|+7` [tool] — 사용자: 7일 뒤에 생일 파티 있는데 낮 2시야. 등록 좀
  - 기대: `{"name": "create_event", "arguments": {"title": "생일 파티", "start": "2026-11-30T14:00", "end": "2026-11-30T15:00", "allDay": false, "category": "personal"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "personal", "end": "2026-12-01T15:00", "start": "2026-12-01T14:00", "title": "생일 파티"}}</tool_call>`
- **probe-0030#t0** `next_wd|d01|wd2|+10` [tool] — 사용자: 내주 토요일 밤 10시 반 심리학개론 과제 하기 스케줄 등록..
  - 기대: `{"name": "create_event", "arguments": {"title": "심리학개론 과제 하기", "start": "2027-05-01T22:30", "end": "2027-05-01T23:30", "allDay": false, "category": "study"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "study", "end": "2027-04-30T23:30", "start": "2027-04-30T22:30", "title": "심리학개론 과제 하기"}}</tool_call>`
- **probe-0031#t0** `next_wd|d31|wd5|+4` [tool] — 사용자: 스터디 추가해줘 다음 주 수요일 오전 9시
  - 기대: `{"name": "create_event", "arguments": {"title": "스터디", "start": "2027-03-31T09:00", "end": "2027-03-31T10:00", "allDay": false, "category": "study"}}`
  - 예측: `<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "study", "end": "2027-04-07T10:00", "start": "2027-04-07T09:00", "title": "스터디"}}</tool_call>`
