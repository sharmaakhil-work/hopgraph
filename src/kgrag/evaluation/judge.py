from enum import StrEnum

from pydantic import BaseModel, Field

from kgrag.ingest.resolve import normalize
from kgrag.llm import LLM


class Verdict(StrEnum):
    CORRECT = "correct"
    PARTIAL = "partial"
    WRONG = "wrong"


class Judgement(BaseModel):
    verdict: Verdict
    mentioned_entities: list[str] = Field(
        description="Every company/person/product named as part of the answer"
    )
    reason: str


JUDGE_SYSTEM = """You grade answers to questions about SEC filings against a gold answer.
- correct: states the gold answer's key facts; extra correct detail is fine.
- partial: some of the key facts, or the right entities with a wrong detail.
- wrong: incorrect, missing the key facts, or a refusal.
List every entity the candidate presents as part of its answer in mentioned_entities (not entities it only \
mentions as context)."""

SCORES = {Verdict.CORRECT: 1.0, Verdict.PARTIAL: 0.5, Verdict.WRONG: 0.0}


def entity_f1(gold: list[list[str]], predicted: list[str]) -> float:
    """gold: one alias list per required entity. A predicted name matches a gold entity if it matches any alias."""
    if not gold:
        return 0.0
    pred = {normalize(p) for p in predicted if p.strip()}
    gold_norm = [{normalize(a) for a in aliases} for aliases in gold]
    hit_gold = sum(1 for g in gold_norm if g & pred)
    hit_pred = sum(1 for p in pred if any(p in g for g in gold_norm))
    recall = hit_gold / len(gold_norm)
    precision = hit_pred / len(pred) if pred else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score(llm: LLM, model: str, item: dict, answer: str, answerable: bool) -> dict:
    if item["category"] == "out_of_scope":
        return {
            "score": 0.0 if answerable else 1.0,
            "verdict": "refused" if not answerable else "answered",
            "f1": None,
            "reason": "out-of-scope questions must be refused",
        }
    if not answerable:
        return {"score": 0.0, "verdict": "refused", "f1": None, "reason": "refused an in-scope question"}
    j, _ = llm.structured(
        model=model,
        system=JUDGE_SYSTEM,
        output=Judgement,
        purpose="judge",
        max_tokens=800,
        cache_system=True,
        user=f"Question: {item['question']}\nGold answer: {item['gold_answer']}\nCandidate answer: {answer}",
    )
    if item.get("gold_entities"):
        f1 = entity_f1(item["gold_entities"], j.mentioned_entities)
        s = 1.0 if f1 >= 0.8 else 0.5 if f1 >= 0.4 else 0.0
        return {"score": s, "verdict": j.verdict.value, "f1": round(f1, 3), "reason": j.reason}
    return {"score": SCORES[j.verdict], "verdict": j.verdict.value, "f1": None, "reason": j.reason}
