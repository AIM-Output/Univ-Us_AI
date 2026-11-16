"""F9 자연어 대화(Function Calling) — Univ-Us. 구조는 ../README.md.

전부 표준 라이브러리 + fastapi — config · tools · llm · store · service · api
LLM 은 도구를 고르기만 하고, 읽기는 각 기능의 계산 결과를 그대로 옮기며, 쓰기는 미리보기만 만든다(F9 D2·D4).
실제 변경은 사용자가 [확인]을 누른 뒤 C1 캘린더에 서버가 한다.

이 파일은 import 해도 아무 일도 하지 않는다(사이드카 --self-check 가 모든 모듈을 불러 본다).
"""
