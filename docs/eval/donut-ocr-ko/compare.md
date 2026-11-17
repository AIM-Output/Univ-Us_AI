# Donut 한국어 OCR — 학습 전/후 (SynthDoG-ko validation)

- 학습 전: `naver-clova-ix/donut-base` · 학습 후: `/home/jovyan/univus-train-out/outputs/donut-ocr-ko/model`
- 행 수 500 / 500 · image 1280x960 · max_len 768 · greedy

| 지표 | 학습 전 | 학습 후 | 변화 |
| --- | --- | --- | --- |
| CER 평균 ↓ | 60.64% | 5.19% | -55.44%p |
| 1-NED 평균 ↑ | 68.64% | 95.25% | +26.61%p |
| 정확 일치율 ↑ | 5.40% | 14.60% | +9.20%p |
| 문자 가중 CER ↓ | 67.93% | 7.51% | -60.42%p |
| 평균 지연 s/장 ↓ | 0.611 | 0.451 | -0.159s |
