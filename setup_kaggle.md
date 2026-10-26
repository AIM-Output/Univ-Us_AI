# 캐글 준비 — kaggle/qwen3-8b

## 1. 계정·노트북 설정 (1회)

| 순서 | 어디서 | 무엇을 |
| --- | --- | --- |
| 1 | kaggle.com → Settings → Phone verification | 전화 인증 — **인증 없이는 GPU·인터넷을 못 켠다** |
| 2 | 노트북 우측 Settings → Accelerator | **GPU T4 x2** — T4 기준으로만 검증한다. 노트북은 한 장만 쓴다(`CUDA_VISIBLE_DEVICES=0`). P100 은 최근 torch cu12x 휠이 sm_60 을 빼서 **미검증** |
| 3 | Settings → Internet | **On** (HF 다운로드·업로드) |
| 4 | Add-ons → Secrets | `HF_TOKEN` = Hugging Face 토큰(**write** 권한 — 어댑터를 올려야 한다) |
| 5 | Settings → Persistence | Files only 로 둬도 **세션이 끝나면 /kaggle/working 은 사라진다**고 전제한다 |

주간 GPU 할당 **30시간**, 세션 최대 **12시간**. 할당은 Settings 화면 오른쪽에 남은 시간이 보인다.

## 2. 데이터 올리기 (학습 전 1회, 데이터 바뀔 때마다)

평가셋은 팀이 **같은 파일**을 써야 비교가 된다. 내 PC 에서 만든 `data/f9/*.jsonl` 을 HF private dataset 으로 올린다.

```bash
# 내 PC (common 또는 아무 모델 브랜치) — huggingface_hub 2.x 의 CLI 이름은 `hf` 다 (huggingface-cli 는 없다)
python src/prepare_f9.py --n 1500 --seed 42                 # data/f9/train.jsonl · eval.jsonl · README.md
hf auth login
hf repos create your-hf-id/univus-f9-data --repo-type dataset --private
hf upload your-hf-id/univus-f9-data data/f9 . --repo-type dataset
```

`config.env` 의 `HF_DATA_REPO=your-hf-id/univus-f9-data` 로 맞춘다. 캐글 데이터셋으로 올려도 되지만 그러면 노트북의 다운로드 셀을 `/kaggle/input/...` 경로로 고쳐야 한다.

## 3. 코드 올리기

노트북은 저장소 코드를 **git clone** 한다. 이 저장소는 아직 로컬 전용이므로 둘 중 하나:

| 방법 | 어떻게 |
| --- | --- |
| A. GitHub 에 올린 뒤 clone | 원격을 만들고 push → 노트북 1번 셀 `REPO_URL` 에 주소, `BRANCH=kaggle/qwen3-8b` |
| B. 코드를 캐글 Dataset 으로 | 브랜치 체크아웃 상태의 폴더(`.git` `data` `outputs` `.venv` 제외)를 zip → Kaggle Datasets 에 **`univus-train-code`** 로 업로드 → 노트북에 Add Data (CLI 면 `kernel-metadata.json` 의 `dataset_sources` 에 이미 `<kaggle-id>/univus-train-code` 가 있으니 id 만 바꾼다). `REPO_URL` 을 비우면 노트북이 `/kaggle/input/univus-train-code` 아래에서 `config.env` 를 **재귀로 찾아** 그 폴더를 쓴다 — zip 이 한 단계 더 안에 풀려도 된다 |

## 4. 노트북 실행

- 캐글 웹에서 `kaggle/univus_f9_train.ipynb` 를 **Import Notebook** 으로 올리고 위 설정을 켠 뒤 **Run All**
- 또는 CLI: `pip install kaggle` · `~/.kaggle/kaggle.json` 준비 후

```bash
cd kaggle
# kernel-metadata.json 의 "id" 를 <본인 kaggle id>/univus-f9-qwen3-8b 로 바꾼 뒤
kaggle kernels push -p .
kaggle kernels status <본인id>/univus-f9-qwen3-8b
kaggle kernels output <본인id>/univus-f9-qwen3-8b -p ../outputs/kaggle-run   # 로그·compare.md 내려받기
```

## 5. 끝났는지 확인

HF 의 `HF_PUSH_REPO` 에 다음이 있어야 한다. **없으면 결과가 없는 것이다** (세션 디스크는 이미 지워졌다).

```
adapter_config.json  adapter_model.safetensors  tokenizer*  run.json  eval_base.md  eval_adapter.md  compare.md
```

서버에서 쓰려면: `hf download your-hf-id/univus-f9-qwen3-8b-lora --local-dir ~/univus-train-out/outputs/qwen3-8b-f9/adapter`
(8B 는 서버 9.5GiB 에서 **4-bit 추론만** 가능하고 컨텍스트가 길면 OOM — GPU 문서 1장 예산표.)
학습이 중간에 끊겼으면: `hf download <repo> --include "last-checkpoint/*" --local-dir outputs/qwen3-8b-f9` 로 받아 `checkpoint-N` 으로 이름을 바꾼 뒤 다음 세션에서 `--resume`.
