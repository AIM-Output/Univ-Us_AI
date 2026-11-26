# data/f9-r1-date — 이어서 학습용 데이터 (날짜 보강 + 복습 행)

> 생성: 2026-11-23 14:06 · seed 23 · 3000행 · 생성기 `src/prepare_f9_dates.py`

다른 run 의 어댑터에서 시작하는 학습(`train.py --init_adapter`)에 쓴다. 날짜 문장만 넣으면 거절을 잊을 수 있어 기존 학습 행을 함께 넣었다.

```
python src/prepare_f9_dates.py --n 1000 --seed 23 --mix 2000 --mix_text 500 --mix_out data/f9-r1-date
```

| 구성 | 행 |
| --- | ---: |
| 날짜 보강 (`synthetic-dates`) | 1000 |
| 복습 — 글로 답하는 행 (거절 · 인사) | 500 |
| 복습 — 도구로 답하는 행 | 1500 |
| **합계** | **3000** |

- 전체에서 글로 답하는 행의 비율: 500/3000 = 16.7% (원래 학습 데이터 1861/11185 = 16.6%)
- 복습 행은 기존 train.jsonl 에서 (출처 · 종류 · 틀 묶음)별 비율대로 뽑았다. 출처: synthetic 200 · synthetic-v2 1800
- 복습 행의 첫 도구: none 500 · list_events 432 · create_event 402 · navigate 143 · open_placement 104 · list_assignments 99 · get_attendance 91 · run_sync 91 · get_exams 69 · get_graduation 69

| 파일 | sha256 앞 12자 |
| --- | --- |
| data/f9-r1-date/train.jsonl | eaef20d494b1 |
| data/f9/extra_dates.jsonl | 9d3b3ecb8856 |
| data/f9/train.jsonl | aa603286c6c6 |
