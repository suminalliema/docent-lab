"""(선택) 웹 채팅으로 직접 받은 해설을 실험 데이터에 합칩니다. API 없이 무료.

ChatGPT, Claude 같은 웹 채팅(무료 플랜)에 작품 이미지와 prompts/의 프롬프트를 그대로 붙여넣고,
받은 답을 data/manual_descriptions.csv 에 한 줄씩 적은 뒤 실행합니다.

    python src/import_manual.py

같은 프롬프트를 썼다면 API로 만든 해설과 똑같이 판정·사람 평가·분석에 들어갑니다.
웹 채팅은 설정(temperature 등)을 고정할 수 없으니, README에 "웹 채팅으로 수집"이라고 밝혀 두세요.
"""
import pandas as pd

from common import DATA, OUT, append_jsonl, read_jsonl


def main():
    df = pd.read_csv(DATA / "manual_descriptions.csv").fillna("")
    out_path = OUT / "descriptions.jsonl"
    done = {r["desc_id"] for r in read_jsonl(out_path)}
    n = 0
    for _, r in df.iterrows():
        if not r["description"].strip():
            continue
        desc_id = f"{r['artwork_id']}__{r['generator']}__{r['prompt']}"
        if desc_id in done:
            continue
        append_jsonl(out_path, {
            "desc_id": desc_id, "artwork_id": r["artwork_id"], "generator": r["generator"],
            "model": r["model"] + " (web)", "prompt": r["prompt"],
            "description": r["description"].strip(), "n_chars": len(r["description"].strip()),
        })
        n += 1
    print(f"웹 채팅 해설 {n}개 추가 → {out_path}")


if __name__ == "__main__":
    main()
