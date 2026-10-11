from __future__ import annotations

import json
import math
import os
import random
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("MOCK_PORT", "8931"))
JITTER = float(os.environ.get("MOCK_JITTER", "0.06"))

TARGET_RX = re.compile(r"TARGET UTTERANCE:\n(.*?)\n\nCANDIDATE UTTERANCES", re.S)
CAND_RX = re.compile(r"Candidate c_(\d{4}):\n(.*?)(?=(?:\n\nCandidate |\n\nOutput ))", re.S)

_TRIGRAM_CACHE: dict[str, frozenset] = {}


def trigrams(text: str) -> frozenset:
    hit = _TRIGRAM_CACHE.get(text)
    if hit is not None:
        return hit
    norm = " ".join(text.lower().split())
    grams = frozenset(norm[i:i + 3] for i in range(max(0, len(norm) - 2)))
    _TRIGRAM_CACHE[text] = grams
    return grams


def similarity(a: str, b: str) -> float:
    ga, gb = trigrams(a), trigrams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def p_same(target: str, candidate: str) -> float:
    base = similarity(target, candidate)
    shaped = 0.02 + 0.96 * math.pow(base, 0.45)
    noisy = shaped + random.uniform(-JITTER, JITTER)
    return round(min(0.99, max(0.01, noisy)), 4)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        user = payload["messages"][-1]["content"]
        target_match = TARGET_RX.search(user)
        target = target_match.group(1) if target_match else ""
        ratings = [{"id": f"c_{int(cid):04d}", "p_same": p_same(target, text)}
                   for cid, text in CAND_RX.findall(user)]
        body = {"id": "mock", "object": "chat.completion",
                "model": payload.get("model", "mock"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": json.dumps({"ratings": ratings})}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 210,
                          "total_tokens": 1110}}
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 512
    allow_reuse_address = True


if __name__ == "__main__":
    print(f"mock judge on http://127.0.0.1:{PORT}/v1 (jitter={JITTER})", flush=True)
    Server(("127.0.0.1", PORT), Handler).serve_forever()
