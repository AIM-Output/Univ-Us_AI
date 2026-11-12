# gpu/donut-ocr — Donut · SynthDoG-ko 한국어 문서 읽기(OCR) · 학교 GPU 서버 · 전체 미세조정

| 항목 | 값 |
| --- | --- |
| 무엇을 | 문서 이미지 → 한국어 텍스트. Donut(OCR 엔진 없는 이미지→텍스트 Transformer)을 SynthDoG-ko 로 전체 미세조정 |
| 모델 | `naver-clova-ix/donut-base` (Swin 인코더 + mBART 4층 디코더, 약 200M) — `config.env` 의 `MODEL_ID` 에만 |
| 데이터 | `naver-clova-ix/synthdog-ko` — train 23,532행(parquet 4개) · validation 500행. 열 `image`={bytes,path} · `ground_truth`(JSON) |
| 환경 | A100 MIG `1g.10gb` — **가용 9.5GiB**, bf16 · `HF_HUB_OFFLINE=1` (가중치·데이터는 서버에 미리 받아 둠) |
| 방식 | 해상도 2560x1920 → **1280x960** · MAX_LEN 768 · fp32 가중치 + bf16 autocast · gradient checkpointing · batch 2 × accum 8 · LR 3e-5 · 8,000행 1 epoch(≈500 step) |

## 왜 이 모델인가
- **논문·데이터가 한 세트** — Donut 논문이 SynthDoG(합성 문서 생성기)와 함께 나왔고, 공개된 SynthDoG-ko 가 바로 그 데이터다.
- **200M 이라 9.5GiB 에서 전체 미세조정이 된다** (LoRA·양자화 없이).
- **학습 전 기준선이 정의된다** — donut-base 는 `<s_synthdog>` 태스크(문서 글자 읽기)로 사전학습돼 있어, 같은 프롬프트로 학습 전/후를 같은 지표로 비교할 수 있다.

## 실행 순서 (서버, `~/univus-ocr`)

```bash
bash train_ocr.sh --prepare   # 스키마·길이·이미지 크기 → $OCR_OUT_DIR/data_stats.md (선택)
bash train_ocr.sh --smoke     # 16행·4 step — s/step·GPU 메모리를 먼저 잰다 (산출물은 OCR_CKPT_DIR/smoke-*)
setsid nohup bash train_ocr.sh > ~/univus-train-out/outputs/donut-ocr-ko/nohup.out 2>&1 < /dev/null &
bash eval_ocr.sh              # 학습 전(base) → 학습 후(finetuned) → compare.md
```
OOM 이면 `BATCH=1 GRAD_ACCUM=16 bash train_ocr.sh`(유효 배치 유지) → 그래도면 `IMAGE_H=960 IMAGE_W=720`(평가도 같은 값). 평가 OOM 은 `bash eval_ocr.sh --batch 2`.
시간이 모자라면 `TIME_BUDGET_MIN`(기본 420) 이 지나면 그 step 에서 멈추고 저장한다. 끊겼으면 `bash train_ocr.sh --resume $OCR_CKPT_DIR/step-<n>`.

## 산출물

| 경로 | 내용 |
| --- | --- |
| `~/univus-train-out/outputs/donut-ocr-ko/model/` | 최종 모델 bf16 safetensors(≈400MB) + processor + tokenizer |
| 같은 곳 `run.json` · `train.log` · `train_console.log` | 설정·step 수·소요·최종 loss·s/step·peak GiB·버전 / step 로그 |
| 같은 곳 `eval_base.{md,json}` · `eval_finetuned.{md,json}` · `compare.md` | 학습 전/후 채점 · 비교표 |
| `/home/jovyan_venv/.venv/univus-train/ocr_ckpt/step-<n>/` | 중간 체크포인트(최신 1개, overlay — pod 재시작 시 사라짐) |

## 지표 (모두 `normalize` 뒤: 양끝 공백 제거·연속 공백 1개. SynthDoG 의 글자 사이 공백은 그대로)
- **CER** = 편집거리(ref, hyp) / max(1, len(ref)) — 행별 평균. 낮을수록 좋다.
- **1-NED** = 1 − 편집거리 / max(len(ref), len(hyp)) — 행별 평균(Donut 논문의 정규화 편집거리 기반 정확도). 높을수록 좋다.
- **정확 일치율** · **문자 가중 CER** = Σ편집거리 / Σlen(ref) · **평균 지연**(s/장, greedy, batch 4, bf16).

## 인용
- Geewook Kim et al., "OCR-free Document Understanding Transformer", ECCV 2022. arXiv:2111.15664
- 데이터: SynthDoG-ko — Hugging Face `naver-clova-ix/synthdog-ko`
