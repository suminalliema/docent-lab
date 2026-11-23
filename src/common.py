"""공통 유틸: 설정 읽기, 이미지 인코딩, 모델 호출(OpenAI / Anthropic / Google)."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
PROMPTS = ROOT / "prompts"

load_dotenv(ROOT / ".env")


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")


def read_image(path: str | Path) -> tuple[bytes, str]:
    p = Path(path)
    if not p.is_absolute():
        p = DATA / p
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    return p.read_bytes(), mime


# ---------------------------------------------------------------- 모델 호출

_LAST_CALL: dict[str, float] = {}


def _wait_rate_limit(provider: str) -> None:
    """무료 등급의 분당 요청 한도를 넘지 않도록 provider마다 호출 간격을 둡니다 (config.yaml의 min_interval_sec)."""
    gap = (load_config().get("min_interval_sec") or {}).get(provider, 0)
    elapsed = time.time() - _LAST_CALL.get(provider, 0)
    if gap and elapsed < gap:
        time.sleep(gap - elapsed)
    _LAST_CALL[provider] = time.time()


def call_model(provider: str, model: str, prompt: str, image: bytes, mime: str,
               temperature: float = 0.3, max_tokens: int = 1200, retries: int = 4) -> str:
    """이미지 1장 + 텍스트 프롬프트를 보내고 응답 텍스트를 돌려줍니다."""
    for attempt in range(retries):
        _wait_rate_limit(provider)
        try:
            if provider == "ollama":
                return _ollama(model, prompt, image, temperature, max_tokens)
            if provider == "openai":
                return _openai(model, prompt, image, mime, temperature, max_tokens)
            if provider == "anthropic":
                return _anthropic(model, prompt, image, mime, temperature, max_tokens)
            if provider == "google":
                return _google(model, prompt, image, mime, temperature, max_tokens)
            raise ValueError(f"알 수 없는 provider: {provider}")
        except ValueError:
            raise
        except Exception as e:  # 네트워크·rate limit 등은 잠깐 쉬고 재시도
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 10  # 429(한도 초과)면 점점 길게 쉼
            print(f"  ! {provider}/{model} 오류: {e} → {wait}초 후 재시도")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _ollama(model, prompt, image, temperature, max_tokens):
    """내 맥에서 돌리는 오픈소스 모델 (Ollama 앱이 켜져 있어야 함). 무료."""
    import ollama
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [image]}],
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    return resp["message"]["content"].strip()


def _openai(model, prompt, image, mime, temperature, max_tokens):
    from openai import OpenAI
    client = OpenAI()  # OPENAI_API_KEY
    b64 = base64.b64encode(image).decode()
    kwargs = dict(model=model, max_completion_tokens=max_tokens, temperature=temperature)
    messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }]
    try:
        resp = client.chat.completions.create(messages=messages, **kwargs)
    except Exception as e:
        # 추론(reasoning) 모델 중에는 temperature 지정을 거부하는 모델이 있어서, 그 경우 빼고 다시 보냅니다.
        if "temperature" not in str(e):
            raise
        kwargs.pop("temperature")
        resp = client.chat.completions.create(messages=messages, **kwargs)
    return (resp.choices[0].message.content or "").strip()


def _anthropic(model, prompt, image, mime, temperature, max_tokens):
    import anthropic
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY
    b64 = base64.b64encode(image).decode()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _google(model, prompt, image, mime, temperature, max_tokens):
    from google import genai
    from google.genai import types
    client = genai.Client()  # GEMINI_API_KEY
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=image, mime_type=mime), prompt],
        config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens),
    )
    return (resp.text or "").strip()


# ---------------------------------------------------------------- JSON 파싱

def parse_json(text: str) -> dict:
    """모델 응답에서 JSON 객체를 꺼냅니다. ```json 블록이 붙어 와도 처리합니다."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"JSON을 찾지 못함: {text[:200]}")
    return json.loads(m.group(0))


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


CRITERIA = ["C1_accuracy", "C2_structure", "C3_specificity", "C4_objectivity",
            "C5_spatial", "C6_listenability", "overall"]
