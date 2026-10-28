import json
from types import SimpleNamespace

import pytest

from hopgraph.llm import LLM, CostMeter


class FakeMessages:
    """Replays queued JSON outputs; records every request."""

    def __init__(self, outputs: list):
        self.outputs = list(outputs)
        self.requests: list[dict] = []

    def create(self, **params):
        self.requests.append(params)
        out = self.outputs.pop(0)
        text = out if isinstance(out, str) else json.dumps(out)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=1000, output_tokens=200, cache_creation_input_tokens=0, cache_read_input_tokens=0
            ),
        )


class FakeClient:
    def __init__(self, outputs: list):
        self.messages = FakeMessages(outputs)


@pytest.fixture
def fake_llm():
    def make(outputs: list, max_usd: float | None = None) -> LLM:
        return LLM(FakeClient(outputs), CostMeter(max_usd=max_usd))

    return make
