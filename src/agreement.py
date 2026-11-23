"""4단계: 사람 평가 ↔ LLM 판정 일치도, 모델·프롬프트 비교, 자기 선호 편향을 분석합니다.

채점이 끝난 시트를 data/human_scores/ 에 넣고 실행합니다 (파일 이름 = 평가자 이름.csv).

    python src/agreement.py

결과: outputs/report.md, outputs/agreement_*.csv
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

from common import CRITERIA, DATA, OUT, read_jsonl


def load_human() -> pd.DataFrame:
    key = pd.read_csv(OUT / "human_key.csv")
    frames = []
    for f in sorted((DATA / "human_scores").glob("*.csv")):
        df = pd.read_csv(f).merge(key, on="blind_id")
        df["rater"] = f.stem
        frames.append(df)
    if not frames:
        raise SystemExit("data/human_scores/ 에 채점한 CSV가 없습니다.")
    human = pd.concat(frames, ignore_index=True)
    for c in CRITERIA + ["hallucination_count"]:
        human[c] = pd.to_numeric(human[c], errors="coerce")
    return human


def load_judge() -> pd.DataFrame:
    j = pd.DataFrame(read_jsonl(OUT / "judge_scores.jsonl"))
    if "parse_error" in j:
        j = j[j["parse_error"] != True]  # noqa: E712
    for c in CRITERIA + ["hallucination_count"]:
        j[c] = pd.to_numeric(j[c], errors="coerce")
    return j


def qwk(a, b) -> float:
    a, b = np.rint(a).astype(int), np.rint(b).astype(int)
    if len(set(a)) < 2 and len(set(b)) < 2:
        return float("nan")
    return cohen_kappa_score(a, b, weights="quadratic", labels=[1, 2, 3, 4, 5])


def corr_row(x, y) -> dict:
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3:
        return {"n": len(x), "spearman": np.nan, "kendall": np.nan, "qwk": np.nan, "mae": np.nan}
    return {"n": len(x), "spearman": spearmanr(x, y).statistic, "kendall": kendalltau(x, y).statistic,
            "qwk": qwk(x, y), "mae": float(np.mean(np.abs(x - y)))}


def main():
    human, judge = load_human(), load_judge()
    lines = ["# 손끝 도슨트 AI 해설 평가 결과\n"]

    # 1) 사람끼리 얼마나 일치하나 (평가 기준이 믿을 만한지의 상한선)
    rows = []
    for a, b in combinations(sorted(human["rater"].unique()), 2):
        ha = human[human.rater == a].set_index("desc_id")
        hb = human[human.rater == b].set_index("desc_id").reindex(ha.index)
        for c in CRITERIA:
            rows.append({"pair": f"{a}-{b}", "criterion": c, **corr_row(ha[c].values, hb[c].values)})
    inter = pd.DataFrame(rows)
    inter.to_csv(OUT / "agreement_human_human.csv", index=False, encoding="utf-8-sig")
    if len(inter):
        lines += ["## 1. 사람 평가자 간 일치도 (평균 Spearman)\n",
                  inter.groupby("criterion")[["spearman", "qwk"]].mean().round(3).to_markdown(), "\n"]

    # 2) 사람 평균 vs 판정 모델
    hmean = human.groupby("desc_id")[CRITERIA + ["hallucination_count"]].mean()
    rows = []
    for jname, jd in judge.groupby("judge"):
        jd = jd.set_index("desc_id").reindex(hmean.index)
        for c in CRITERIA:
            rows.append({"judge": jname, "criterion": c, **corr_row(hmean[c].values, jd[c].values)})
    hj = pd.DataFrame(rows)
    hj.to_csv(OUT / "agreement_human_judge.csv", index=False, encoding="utf-8-sig")
    lines += ["## 2. 사람 평균 ↔ LLM 판정 일치도 (Spearman)\n",
              hj.pivot(index="criterion", columns="judge", values="spearman").round(3).to_markdown(), "\n"]

    # 3) 환각 탐지: 사람 과반이 '환각 있음'으로 본 해설을 판정 모델이 잡아내나
    h_flag = (human.assign(f=human.hallucination_count > 0).groupby("desc_id")["f"].mean() >= 0.5)
    rows = []
    for jname, jd in judge.groupby("judge"):
        j_flag = (jd.set_index("desc_id")["hallucination_count"] > 0).reindex(h_flag.index).fillna(False)
        p, r, f1, _ = precision_recall_fscore_support(h_flag, j_flag, average="binary", zero_division=0)
        rows.append({"judge": jname, "human_positive": int(h_flag.sum()), "precision": p, "recall": r, "f1": f1})
    hall = pd.DataFrame(rows)
    hall.to_csv(OUT / "agreement_hallucination.csv", index=False, encoding="utf-8-sig")
    lines += ["## 3. 환각 탐지 성능 (사람 과반 기준)\n", hall.round(3).to_markdown(index=False), "\n"]

    # 4) 모델 × 프롬프트 비교 (사람 점수 기준)
    meta = judge[["desc_id", "generator", "prompt"]].drop_duplicates().set_index("desc_id")
    comp = hmean.join(meta).groupby(["generator", "prompt"])[CRITERIA + ["hallucination_count"]].mean()
    comp.to_csv(OUT / "compare_generators.csv", encoding="utf-8-sig")
    lines += ["## 4. 모델 × 프롬프트별 사람 평균 점수\n", comp.round(2).to_markdown(), "\n"]

    # 5) 자기 선호 편향: 판정 모델이 같은 회사 모델의 해설에 더 후한가
    #    (judge 이름의 앞부분이 generator 이름과 같으면 '자기 것'으로 봄. 예: claude-judge ↔ claude)
    j2 = judge.set_index("desc_id").join(hmean["overall"].rename("human_overall"))
    j2["own"] = j2.apply(lambda r: r["judge"].split("-")[0] == r["generator"], axis=1)
    j2["gap"] = j2["overall"] - j2["human_overall"]
    bias = j2.groupby(["judge", "own"])["gap"].mean().unstack()
    bias.to_csv(OUT / "self_preference.csv", encoding="utf-8-sig")
    lines += ["## 5. 자기 선호 편향 (판정 점수 − 사람 점수, overall)\n",
              "`True` = 같은 회사 모델이 쓴 해설, `False` = 다른 회사 모델이 쓴 해설\n\n",
              bias.round(3).to_markdown(), "\n"]

    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"완료 → {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
