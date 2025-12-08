import json
from typing import Any, Iterator
import httpx


class OLlamaClient:
    def __init__(self, base_url: str, *, client: httpx.Client | None = None):
        self.url = base_url
        if client is None:
            client = httpx.Client(base_url=base_url)
        self.client = client

    def get_models(self) -> dict[str, dict[str, Any]]:
        response = self.client.get("/api/tags")
        return response.json()

    def answer(self, model: str, prompt: str) -> str:
        response = self.client.post(
            "/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
        )
        return response.json()

    def stream_answer(self, model: str, prompt: str) -> Iterator[str]:
        data = {
            "model": model,
            "prompt": prompt,
            "stream": True,
        }

        with self.client.stream("POST", "/api/generate", json=data) as response:
            response.raise_for_status()
            for chunk in response.iter_lines():
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                yield data["response"]
        return ""
