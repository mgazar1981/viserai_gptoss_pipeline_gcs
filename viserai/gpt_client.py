from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import os, json, random, requests

SYSTEM_PROMPT = """You are Viserai GPT-OSS, a market analysis model.
Return STRICT JSON only with keys:
- reasoning: string
- analysis: string
- y1: list[float] length H
- y2: list[float] length H
- sources: list[object] (optional)
No extra text outside JSON.
"""

@dataclass
class GPTResp:
    reasoning: str
    analysis: str
    y1: List[float]
    y2: List[float]
    sources: List[Dict[str, Any]]

class GPTClient:
    def generate_pass(self, prompt: str, H: int) -> GPTResp:
        raise NotImplementedError

class DummyClient(GPTClient):
    def generate_pass(self, prompt: str, H: int) -> GPTResp:
        def rw():
            out=[]
            x=100.0
            for _ in range(H):
                x += random.uniform(-1,1)
                out.append(float(x))
            return out
        return GPTResp(
            reasoning="(dummy) reasoning trace",
            analysis="(dummy) analysis placeholder",
            y1=rw(),
            y2=rw(),
            sources=[],
        )

class OpenAICompatClient(GPTClient):
    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: int = 180):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def generate_pass(self, prompt: str, H: int) -> GPTResp:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "temperature": 0.4,
            "messages": [
                {"role":"system","content": SYSTEM_PROMPT + f"\nH={H}"},
                {"role":"user","content": prompt},
            ],
            "response_format": {"type":"json_object"},
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        obj = json.loads(content)

        def flist(v):
            if isinstance(v, list) and len(v)==H:
                return [float(x) for x in v]
            return [0.0]*H

        return GPTResp(
            reasoning=str(obj.get("reasoning","")),
            analysis=str(obj.get("analysis","")),
            y1=flist(obj.get("y1")),
            y2=flist(obj.get("y2")),
            sources=obj.get("sources", []) or [],
        )

def get_gpt_client() -> GPTClient:
    mode = os.environ.get("VISERAI_GPT_MODE", "openai_compat").lower()
    if mode == "dummy":
        return DummyClient()
    base_url = os.environ.get("VISERAI_OPENAI_BASE_URL", "http://localhost:8000")
    api_key = os.environ.get("VISERAI_OPENAI_API_KEY", "")
    model = os.environ.get("VISERAI_OPENAI_MODEL", "gpt-oss-120b")
    timeout_s = int(os.environ.get("VISERAI_OPENAI_TIMEOUT_S", "180"))
    return OpenAICompatClient(base_url, api_key, model, timeout_s=timeout_s)
