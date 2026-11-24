"""3단계: 사람 평가용 채점표(CSV)를 만듭니다.

모델 이름과 프롬프트를 가리고(블라인드) 순서를 섞어서, 평가자마다 같은 시트를 줍니다.
CSV를 구글 시트에 올려 팀원(그리고 동의한 당사자 평가자)이 채점합니다.
image 칸의 =IMAGE(...) 수식 덕분에 구글 시트에서 그림이 셀 안에 바로 보입니다.

    python src/make_human_sheet.py --raters R1 R2 R3

평가자 이름은 결과 파일(일치도 표 등)에 그대로 남으므로 실명 대신 R1, R2 … 같은 코드를 쓰세요.
누가 어떤 코드인지는 레포 밖(개인 메모)에만 적어 둡니다.

결과:
    outputs/human_sheet_<이름>.csv   평가자에게 줄 시트
    outputs/human_key.csv           blind_id ↔ desc_id 대응표 (평가자에게 보여주지 마세요)
"""
import argparse
import random

import pandas as pd

from common import CRITERIA, DATA, OUT, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=None, help="이 프롬프트로 만든 해설만 넣기 (예: generate_v2_guided)")
    ap.add_argument("--repo", default="suminalliema/docent-lab", help="이미지를 불러올 공개 GitHub 레포")
    args = ap.parse_args()

    descs = read_jsonl(OUT / "descriptions.jsonl")
    if args.prompt:
        descs = [d for d in descs if d["prompt"] == args.prompt]
    artworks = pd.read_csv(DATA / "artworks.csv").fillna("").set_index("id")
    rng = random.Random(args.seed)
    rng.shuffle(descs)

    key, rows = [], []
    for i, d in enumerate(descs, 1):
        blind_id = f"D{i:03d}"
        key.append({"blind_id": blind_id, "desc_id": d["desc_id"]})
        art = artworks.loc[d["artwork_id"]]
        url = f"https://raw.githubusercontent.com/{args.repo}/main/data/{art['image_path']}"
        row = {"blind_id": blind_id, "artwork_id": d["artwork_id"], "title": art["title"],
               "image": f'=IMAGE("{url}")',  # 구글 시트에 올리면 그림이 셀 안에 보여요
               "image_url": url, "description": d["description"],
               "hallucination_count": "", "hallucination_spans": ""}
        row.update({c: "" for c in CRITERIA})
        row["comment"] = ""
        rows.append(row)

    OUT.mkdir(exist_ok=True)
    pd.DataFrame(key).to_csv(OUT / "human_key.csv", index=False, encoding="utf-8-sig")
    for r in args.raters:
        pd.DataFrame(rows).to_csv(OUT / f"human_sheet_{r}.csv", index=False, encoding="utf-8-sig")
    print(f"해설 {len(rows)}개 × 평가자 {len(args.raters)}명 시트 생성 → {OUT}")


if __name__ == "__main__":
    main()
