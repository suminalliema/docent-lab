# 손끝 도슨트 AI 해설 랩 (docent-lab)

시각장애인 관람객을 위한 **미술 작품 음성해설을 VLM으로 생성하고, 그 품질을 측정하는** 실험 레포입니다.
결과물 중 팀이 검수한 해설은 [손끝 도슨트](https://sonkkeut-docent.github.io) 사이트의 '웹 도슨트' 메뉴에 게시합니다.

> 롯데 밸유 for ESG 5기 **썬덕걸즈**의 배리어프리 미술관 프로젝트에서 출발했습니다.

## 연구 질문

1. **어떤 모델과 프롬프트가** 시각장애인용 해설을 가장 잘 쓰는가?
2. **LLM 판정(LLM-as-a-judge)은 사람 평가와 얼마나 일치하는가?** 어떤 항목에서 어긋나는가?
3. 판정 모델은 **환각**(이미지에 없는 내용)을 얼마나 잡아내는가?
4. 판정 모델은 **같은 회사 모델이 쓴 해설에 더 후한가?** (자기 선호 편향)

## 파이프라인

```
작품 이미지 ─▶ ① 해설 생성 ─▶ ② LLM 판정 ─┐
 (data/)       (모델 × 프롬프트)   (루브릭 채점)  ├─▶ ④ 일치도 분석 ─▶ report.md
               └──────────▶ ③ 사람 평가 ───┘   (블라인드 채점표)
                                  │
                                  ▼
                    ⑤ 사람 검수 → 승인된 해설만 사이트 게시
```

| 단계 | 스크립트 | 결과 |
|---|---|---|
| ① 생성 | `src/generate.py` (+ 선택: `src/import_manual.py`) | `outputs/descriptions.jsonl` |
| ② 판정 | `src/judge.py` | `outputs/judge_scores.jsonl` |
| ③ 사람 평가 시트 | `src/make_human_sheet.py` | `outputs/human_sheet_*.csv` |
| ④ 분석 | `src/agreement.py` | `outputs/report.md` |
| ⑤ 사이트 내보내기 | `src/export_site.py` | `<사이트>/docent/artworks.json` |

평가 기준: [`rubric.md`](rubric.md) (환각 + 6개 항목 + 종합)

## 모델 구성 (무료)

| 이름 | 종류 | 실행 위치 | 비용 |
|---|---|---|---|
| Gemini | 상용 (Google) | Google AI Studio 무료 등급 API | 무료 (한도 있음) |
| Gemma 3 | 오픈소스 (Google) | 내 맥, Ollama | 무료 |
| Qwen2.5-VL | 오픈소스 (Alibaba) | 내 맥, Ollama | 무료 |
| (선택) ChatGPT·Claude | 상용 | 웹 채팅에서 직접 수집 → `src/import_manual.py` | 무료 플랜 |

상용 모델과 내 노트북에서 돌아가는 오픈소스 모델이 시각장애인용 해설에서 얼마나 차이 나는지도 함께 봅니다.
유료 API(OpenAI·Anthropic)는 `config.yaml`의 주석을 풀면 바로 추가됩니다.

## 실행

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # GEMINI_API_KEY 입력
ollama pull gemma3:12b            # 오픈소스 모델 내려받기 (Ollama 앱 설치 후)
ollama pull qwen2.5vl:7b
# data/artworks.csv 에 작품 정보, data/images/ 에 이미지 넣기

python src/generate.py --limit 1  # 먼저 1개로 시험
python src/generate.py
python src/import_manual.py       # (선택) 웹 채팅으로 받은 해설 합치기
python src/judge.py
python src/make_human_sheet.py --raters R1 R2 R3        # 실명 대신 코드 사용
# 채점이 끝난 시트를 data/human_scores/<이름>.csv 로 저장
python src/agreement.py
# data/reviewed.csv 에 검수한 최종본 입력 (status=approved)
python src/export_site.py --site ../sonkkeut-docent.github.io
```

## 원칙

- **AI 초안을 그대로 게시하지 않습니다.** 시각장애인 관람객은 해설이 틀려도 확인할 방법이 없으므로, 모든 해설은 팀원이 이미지와 대조해 고친 뒤 게시합니다.
- **작품 이미지**는 퍼블릭 도메인·공공누리 등 이용이 허락된 것만 씁니다. `artworks.csv`에 출처와 라이선스를 반드시 적습니다.
- **사람 평가 원본**(`data/human_scores/`)은 평가자 보호를 위해 커밋하지 않습니다. 당사자 평가는 동의를 받은 경우에만 진행합니다.
- 루브릭이나 프롬프트를 바꾸면 버전을 올리고, 이전 버전 결과와 섞지 않습니다.

## 결과

_(실험 후 `outputs/report.md`의 핵심 수치와 발견을 여기에 요약합니다.)_

- 최고 조합:
- 사람-판정 일치도(overall Spearman):
- 판정이 가장 많이 틀리는 항목:
- 환각 탐지 recall:
- 자기 선호 편향:

## 만든 사람

마수민 ([@suminalliema](https://github.com/suminalliema)) · 동덕여자대학교 데이터사이언스전공
