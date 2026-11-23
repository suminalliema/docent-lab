"""2단계: 생성된 해설을 판정 모델(LLM-as-a-judge)로 루브릭 채점합니다.

    python src/judge.py
"""
import pandas as pd

from common import (CRITERIA, DATA, OUT, ROOT, append_jsonl, call_model, load_config,
                    load_prompt, parse_json, read_image, read_jsonl)


def main():
    cfg = load_config()
    rubric = (ROOT / cfg["rubric_file"]).read_text(encoding="utf-8")
    template = load_prompt(cfg["judge_prompt"])
    artworks = pd.read_csv(DATA / "artworks.csv").fillna("").set_index("id")
    descs = read_jsonl(OUT / "descriptions.jsonl")

    out_path = OUT / "judge_scores.jsonl"
    done = {(r["desc_id"], r["judge"]) for r in read_jsonl(out_path)}

    for d in descs:
        art = artworks.loc[d["artwork_id"]]
        image, mime = read_image(art["image_path"])
        prompt = template.format(title=art["title"], artist=art["artist"] or "작가 미상",
                                 year=art["year"] or "연도 미상",
                                 description=d["description"], rubric=rubric)
        for j in cfg["judges"]:
            if (d["desc_id"], j["name"]) in done:
                continue
            print(f"[채점] {d['desc_id']} ← {j['name']}")
            raw = call_model(j["provider"], j["model"], prompt, image, mime, 0.0, cfg["max_tokens"])
            try:
                scores = parse_json(raw)
            except ValueError as e:
                print(f"  ! 파싱 실패, raw 저장: {e}")
                scores = {"parse_error": True, "raw": raw}
            row = {"desc_id": d["desc_id"], "artwork_id": d["artwork_id"],
                   "generator": d["generator"], "prompt": d["prompt"],
                   "judge": j["name"], "judge_model": j["model"]}
            row.update({k: scores.get(k) for k in CRITERIA})
            row["hallucination_count"] = scores.get("hallucination_count")
            row["hallucination_spans"] = scores.get("hallucination_spans")
            row["comment"] = scores.get("comment")
            if scores.get("parse_error"):
                row["raw"] = scores["raw"]
            append_jsonl(out_path, row)
    print(f"완료 → {out_path}")


if __name__ == "__main__":
    main()
