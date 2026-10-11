from __future__ import annotations

import json
from pathlib import Path


def load_system_prompt(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"prompt file not found: {p}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"prompt file is empty: {p}")
    return text


def clip(text: str, n: int = 240) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[:n - 1] + "\u2026"


def render_user_message(texts: list[str], target: int, candidates: list[int],
                        clip_length: int = 240) -> str:
    rows = ["TARGET UTTERANCE:", clip(texts[target], clip_length), "",
            "CANDIDATE UTTERANCES (preserve these IDs):"]
    for c in candidates:
        rows.extend([f"Candidate c_{c:04d}:", clip(texts[c], clip_length), ""])
    rows.append("Output one p_same for every candidate ID listed above.")
    return "\n".join(rows)


def graded_schema(candidates: list[int]) -> dict:
    enum = [f"c_{c:04d}" for c in candidates]
    return {"type": "json_schema", "json_schema": {
        "name": "graded", "strict": True, "schema": {
            "type": "object", "properties": {"ratings": {
                "type": "array", "items": {
                    "type": "object", "properties": {
                        "id": {"type": "string", "enum": enum},
                        "p_same": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
                    "required": ["id", "p_same"], "additionalProperties": False}}},
            "required": ["ratings"], "additionalProperties": False}}}


def build_payload(system_prompt: str, texts: list[str], target: int,
                  candidates: list[int], max_tokens: int = 900,
                  temperature: float = 0.0,
                  sampling_seed: int | None = None) -> dict:
    user = render_user_message(texts, target, candidates)
    payload = {"messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user}],
               "response_format": graded_schema(candidates),
               "temperature": float(temperature),
               "max_tokens": max_tokens}
    if sampling_seed is not None:
        payload["seed"] = int(sampling_seed)
    return payload


def parse_ratings(row: dict, candidates: list[int]) -> dict[int, float] | None:
    try:
        content = row["response"]["choices"][0]["message"]["content"]
        obj = json.loads(content)
    except Exception:
        return None
    allowed = {int(c) for c in candidates}
    out: dict[int, float] = {}
    for entry in obj.get("ratings", []):
        sid = str(entry.get("id", ""))
        if sid.startswith("c_") and sid[2:].isdigit() and int(sid[2:]) in allowed:
            out[int(sid[2:])] = float(entry.get("p_same", 0.0))
    return out
