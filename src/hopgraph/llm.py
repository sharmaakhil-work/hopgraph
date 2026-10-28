import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from hopgraph.config import (
    BATCH_DISCOUNT,
    CACHE_READ_MULT,
    CACHE_WRITE_MULT,
    NO_EFFORT_MODELS,
    PRICES,
)

T = TypeVar("T", bound=BaseModel)

# JSON-schema keywords structured outputs doesn't accept; pydantic still enforces them after parsing.
_UNSUPPORTED_KEYS = {
    "default",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "maxItems",
    "pattern",
    "format",
    "title",
}


class BudgetExceeded(RuntimeError):
    pass


class StructuredOutputError(RuntimeError):
    def __init__(self, message: str, errors: list[str]):
        super().__init__(message)
        self.errors = errors


def strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema -> structured-outputs schema (closed objects, every property required)."""

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out = {k: walk(v) for k, v in node.items() if k not in _UNSUPPORTED_KEYS}
            if "properties" in node:  # property names are data, not keywords
                out["properties"] = {k: walk(v) for k, v in node["properties"].items()}
            if out.get("type") == "object" or "properties" in out:
                out["additionalProperties"] = False
                out["required"] = list(out.get("properties", {}).keys())
            if "minItems" in out and out["minItems"] > 1:
                out.pop("minItems")
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(model.model_json_schema())


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    batch: bool = False

    @property
    def cost_usd(self) -> float:
        inp, out = PRICES.get(self.model, (0.0, 0.0))
        cost = (
            self.input_tokens * inp
            + self.cache_write_tokens * inp * CACHE_WRITE_MULT
            + self.cache_read_tokens * inp * CACHE_READ_MULT
            + self.output_tokens * out
        ) / 1_000_000
        return cost * (BATCH_DISCOUNT if self.batch else 1.0)

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            model=self.model,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            batch=self.batch,
        )

    @classmethod
    def from_response(cls, model: str, usage: Any, batch: bool = False) -> "Usage":
        return cls(
            model=model,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            batch=batch,
        )


@dataclass
class CostMeter:
    """Running spend with a hard cap. Every call's usage is appended to a JSONL ledger."""

    max_usd: float | None = None
    ledger_path: Path | None = None
    label: str = "default"
    spent_usd: float = 0.0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check(self) -> None:
        if self.max_usd is not None and self.spent_usd >= self.max_usd:
            raise BudgetExceeded(f"spend ${self.spent_usd:.2f} reached cap ${self.max_usd:.2f}")

    def record(self, usage: Usage, purpose: str) -> None:
        with self._lock:
            self.spent_usd += usage.cost_usd
            self.calls += 1
            if self.ledger_path:
                self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
                with self.ledger_path.open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "ts": time.time(),
                                "label": self.label,
                                "purpose": purpose,
                                "model": usage.model,
                                "input": usage.input_tokens,
                                "output": usage.output_tokens,
                                "cache_write": usage.cache_write_tokens,
                                "cache_read": usage.cache_read_tokens,
                                "batch": usage.batch,
                                "usd": round(usage.cost_usd, 6),
                            }
                        )
                        + "\n"
                    )


def _text_of(response: Any) -> str:
    return next((b.text for b in response.content if b.type == "text"), "")


class LLM:
    def __init__(self, client: Any = None, meter: CostMeter | None = None):
        self.client = client or anthropic.Anthropic()
        self.meter = meter or CostMeter()

    def request_params(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        schema: dict,
        max_tokens: int,
        effort: str | None = None,
        cache_system: bool = True,
    ) -> dict:
        output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
        if effort and model not in NO_EFFORT_MODELS:
            output_config["effort"] = effort
        system_block: dict = {"type": "text", "text": system}
        if cache_system:
            system_block["cache_control"] = {"type": "ephemeral"}
        return {
            "model": model,
            "max_tokens": max_tokens,
            "system": [system_block],
            "messages": messages,
            "output_config": output_config,
        }

    def structured(
        self,
        *,
        model: str,
        system: str,
        user: str,
        output: type[T],
        purpose: str,
        max_tokens: int = 4096,
        effort: str | None = None,
        validator: Callable[[T], list[str]] | None = None,
        max_retries: int = 2,
        cache_system: bool = True,
    ) -> tuple[T, Usage]:
        """One structured call, validated by pydantic plus an optional semantic validator.

        On failure the errors are sent back to the model and it gets `max_retries` more attempts.
        """
        schema = strict_schema(output)
        messages: list[dict] = [{"role": "user", "content": user}]
        total = Usage(model=model)
        errors: list[str] = []
        for _attempt in range(max_retries + 1):
            self.meter.check()
            params = self.request_params(
                model=model,
                system=system,
                messages=messages,
                schema=schema,
                max_tokens=max_tokens,
                effort=effort,
                cache_system=cache_system,
            )
            response = self.client.messages.create(**params)
            usage = Usage.from_response(model, response.usage)
            self.meter.record(usage, purpose)
            total = total + usage

            if response.stop_reason == "refusal":
                raise StructuredOutputError("model refused", ["refusal"])
            text = _text_of(response)
            errors = self._validate(text, output, validator)
            if not errors:
                return output.model_validate_json(text), total
            messages = messages + [
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": "Your output failed validation:\n- "
                    + "\n- ".join(errors)
                    + "\nReturn the corrected full JSON.",
                },
            ]
        raise StructuredOutputError(f"{purpose}: invalid after {max_retries + 1} attempts", errors)

    @staticmethod
    def _validate(text: str, output: type[T], validator: Callable[[T], list[str]] | None) -> list[str]:
        try:
            parsed = output.model_validate_json(text)
        except ValidationError as e:
            return [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()][:20]
        return validator(parsed) if validator else []

    def count_tokens(self, *, model: str, system: str, user: str) -> int:
        return self.client.messages.count_tokens(
            model=model, system=system, messages=[{"role": "user", "content": user}]
        ).input_tokens
