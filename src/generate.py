"""1단계: 작품마다 (모델 × 프롬프트) 조합으로 해설을 생성합니다.

    python src/generate.py            # 전체 실행 (이미 만든 조합은 건너뜀)
    python src/generate.py --limit 2  # 작품 2개만 시험 실행
"""
import argparse

import pandas as pd

from common import DATA, OUT, append_jsonl, call_model, load_config, load_prompt, read_image, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    artworks = pd.read_csv(DATA / "artworks.csv").fillna("")
    if args.limit:
        artworks = artworks.head(args.limit)

    out_path = OUT / "descriptions.jsonl"
    done = {(r["artwork_id"], r["generator"], r["prompt"]) for r in read_jsonl(out_path)}

    for _, art in artworks.iterrows():
        image, mime = read_image(art["image_path"])
        for prompt_name in cfg["prompts"]:
            prompt = load_prompt(prompt_name).format(
                title=art["title"], artist=art["artist"] or "작가 미상", year=art["year"] or "연도 미상")
            for g in cfg["generators"]:
                key = (art["id"], g["name"], prompt_name)
                if key in done:
                    continue
                print(f"[생성] {art['id']} / {g['name']} / {prompt_name}")
                text = call_model(g["provider"], g["model"], prompt, image, mime,
                                  cfg["temperature"], cfg["max_tokens"])
                append_jsonl(out_path, {
                    "desc_id": f"{art['id']}__{g['name']}__{prompt_name}",
                    "artwork_id": art["id"],
                    "generator": g["name"],
                    "model": g["model"],
                    "prompt": prompt_name,
                    "description": text,
                    "n_chars": len(text),
                })
    print(f"완료 → {out_path}")


if __name__ == "__main__":
    main()
