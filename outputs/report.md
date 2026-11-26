# 손끝 도슨트 AI 해설 평가 결과

## 1. 사람 평가자 간 일치도 (평균 Spearman)

| criterion        |   spearman |   qwk |
|:-----------------|-----------:|------:|
| C1_accuracy      |      0.598 | 0.509 |
| C2_structure     |      0.603 | 0.476 |
| C3_specificity   |      0.638 | 0.57  |
| C4_objectivity   |      0.345 | 0.346 |
| C5_spatial       |      0.625 | 0.39  |
| C6_listenability |      0.269 | 0.295 |
| overall          |      0.618 | 0.535 |


## 2. 사람 평균 ↔ LLM 판정 일치도 (Spearman)

| criterion        |   gemini-judge |   gemma-judge |
|:-----------------|---------------:|--------------:|
| C1_accuracy      |          0.617 |         0.346 |
| C2_structure     |          0.854 |         0.674 |
| C3_specificity   |          0.63  |         0.396 |
| C4_objectivity   |          0.692 |         0.362 |
| C5_spatial       |          0.691 |         0.392 |
| C6_listenability |          0.586 |         0.484 |
| overall          |          0.695 |         0.521 |


## 3. 환각 탐지 성능 (사람 과반 기준)

| judge        |   human_positive |   precision |   recall |    f1 |
|:-------------|-----------------:|------------:|---------:|------:|
| gemini-judge |               31 |        0.76 |    0.613 | 0.679 |
| gemma-judge  |               31 |        0    |    0     | 0     |


## 4. 모델 × 프롬프트별 사람 평균 점수

|                                  |   C1_accuracy |   C2_structure |   C3_specificity |   C4_objectivity |   C5_spatial |   C6_listenability |   overall |   hallucination_count |
|:---------------------------------|--------------:|---------------:|-----------------:|-----------------:|-------------:|-------------------:|----------:|----------------------:|
| ('gemini', 'generate_v2_guided') |          4.33 |           4.64 |             4.42 |             4.56 |         4.69 |               4.53 |      4.29 |                  0.91 |
| ('gemma', 'generate_v2_guided')  |          3.4  |           4.2  |             4.2  |             3.53 |         4.36 |               3.89 |      3.31 |                  2.31 |
| ('qwen', 'generate_v2_guided')   |          2.73 |           3.22 |             2.79 |             3.42 |         3.42 |               3.64 |      2.6  |                  7.82 |


## 5. 자기 선호 편향 (판정 점수 − 사람 점수, overall)

`True` = 같은 회사 모델이 쓴 해설, `False` = 다른 회사 모델이 쓴 해설


| judge        |   False |   True |
|:-------------|--------:|-------:|
| gemini-judge |  -0.622 |  0.178 |
| gemma-judge  |   0.656 |  0.689 |

