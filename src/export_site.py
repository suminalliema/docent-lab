"""5단계: 팀이 검수·승인한 해설만 손끝 도슨트 사이트용 JSON으로 내보냅니다.

data/reviewed.csv 에서 status 가 approved 인 행만 내보냅니다.
AI 초안을 그대로 게시하지 않고, 사람이 고친 final_text 를 게시합니다 (human-in-the-loop).

    python src/export_site.py --site ../sonkkeut-docent.github.io

결과: <site>/docent/artworks.json  (사이트의 '웹 도슨트' 메뉴에서 읽을 파일)
"""
import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

from common import DATA, read_jsonl, OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(OUT / "site_preview"), help="사이트 레포 경로")
    args = ap.parse_args()

    site = Path(args.site)
    artworks = pd.read_csv(DATA / "artworks.csv").fillna("").set_index("id")
    reviewed = pd.read_csv(DATA / "reviewed.csv").fillna("")
    approved = reviewed[reviewed["status"].str.strip().str.lower() == "approved"]
    descs = {d["desc_id"]: d for d in read_jsonl(OUT / "descriptions.jsonl")}

    items = []
    for _, r in approved.iterrows():
        art = artworks.loc[r["artwork_id"]]
        src_img = DATA / art["image_path"]
        rel_img = f"docent/images/{src_img.name}"
        (site / "docent" / "images").mkdir(parents=True, exist_ok=True)
        if src_img.exists():
            shutil.copy2(src_img, site / rel_img)
        d = descs.get(r["desc_id"], {})
        items.append({
            "id": r["artwork_id"],
            "title": art["title"],
            "artist": art["artist"],
            "year": str(art["year"]),
            "image": rel_img,
            "alt": r.get("short_alt", ""),          # 이미지 자체의 짧은 대체텍스트
            "description": r["final_text"],         # 사람이 검수한 해설 본문
            "license": art["license"],
            "source": art["source_url"],
            "ai": {"model": d.get("model", ""), "prompt": d.get("prompt", ""),
                   "edited": r["final_text"].strip() != d.get("description", "").strip()},
            "reviewedBy": r["reviewer"],
            "reviewedAt": r.get("reviewed_at") or date.today().isoformat(),
        })

    out = site / "docent" / "artworks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"updated": date.today().isoformat(), "artworks": items},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"승인된 해설 {len(items)}개 → {out}")


if __name__ == "__main__":
    main()
