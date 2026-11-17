# Univ-Us_Train — Claude Code 작업 지침

**먼저 [HANDOFF.md](HANDOFF.md) 를 읽는다.** 거기에 프로젝트 맥락 · 환경 사실 · 데이터 흐름 · 현재 상태 · 다음 할 일이 있다.

## 지킬 것
- **시킨 것만 한다.** 제안이나 다음 단계를 덧붙이지 않는다.
- 비밀값은 커밋하지 않는다: `config.local.env` · `scripts/server.env` · `private.pem` · HF 토큰. (`.gitignore` 에 줄 끝 주석을 쓰지 않는다.)
- 모델 이름은 `config.env` 의 `MODEL_ID` 에만 쓴다. `src/` 와 `gpu_server/` 안에는 모델 이름을 넣지 않는다.
- 공통 코드는 `common` 브랜치에서 고치고 모델 브랜치(`gpu/*`, `kaggle/*`)에 `git merge common` 한다. `main` 은 문서만.
- 서버(pod)에서 무언가를 지울 때는 목록을 보여주고 확인받은 뒤에만 지운다.
- 서버에서 코드를 고쳤으면 로컬 레포에도 같은 수정을 반영한다(서버 사본은 배포 때 덮어써진다).
- 완료라고 말하기 전에 실제 실행 결과(명령 출력·HTTP 응답·로그)를 확인한다.

## 이 폴더가 어디인지에 따라
- **로컬 PC** (`C:\Project\Univ-Us_AI\Univ-Us_Train`, `.git` 있음): 코드 수정·커밋·배포·터널·로컬 데이터 서버를 여기서 한다.
- **GPU 서버** (`~/univus-train`, `.git` 없음, `.deployed_branch` 있음): 환경 설치 · `gpu_server` 기동 · 학습 상태 확인 · 병합을 여기서 한다. 영속 산출물은 `~/univus-train-out`.

## 자주 쓰는 확인
```bash
cat .deployed_branch                      # 서버에 올라온 브랜치
bash gpu_server/run_server.sh status      # 서버 프로세스
tail -n 40 ~/univus-train-out/logs/server.log
df -h ~ ; du -sh ~/univus-train-out/*     # 홈 10GB 예산
```
