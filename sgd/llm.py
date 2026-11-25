"""Minimal Ollama chat client (stdlib only). The model is Devstral served locally; nothing leaves the host.

    llm = Ollama()                       # OLLAMA_HOST / SGD_MODEL env vars override the defaults
    reply = llm.chat([{"role": "system", "content": "..."}, {"role": "user", "content": "..."}])
    reply.text, reply.prompt_tokens, reply.output_tokens, reply.seconds
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass

DEFAULT_MODEL = "devstral-small-2"
NUM_CTX = 16384


@dataclass
class Reply:
    text: str
    prompt_tokens: int
    output_tokens: int
    seconds: float


class Ollama:
    def __init__(self, model: str | None = None, host: str | None = None, num_ctx: int = NUM_CTX,
                 temperature: float = 0.0, seed: int = 0, max_output: int = 2048, timeout: float = 1800):
        self.model = model or os.environ.get("SGD_MODEL", DEFAULT_MODEL)
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.options = {"num_ctx": num_ctx, "temperature": temperature, "seed": seed, "num_predict": max_output}
        self.timeout = timeout

    def chat(self, messages: list[dict], json_mode: bool = True) -> Reply:
        body = {"model": self.model, "messages": messages, "stream": False, "options": self.options}
        if json_mode:
            body["format"] = "json"
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = json.loads(r.read())
        return Reply(d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0),
                     time.time() - t0)
