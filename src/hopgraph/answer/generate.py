from dataclasses import dataclass

from hopgraph.answer.citations import AnswerDraft, citation_errors
from hopgraph.answer.context import Context
from hopgraph.llm import LLM, StructuredOutputError, Usage

ANSWER_SYSTEM = """You answer questions about SEC filings using only the provided context.

The context has two kinds of evidence, each labeled with chunk ids in square brackets:
- GRAPH FACTS: relationships extracted from filings, each citing the chunk it came from.
- RETRIEVED PASSAGES: filing text.

Rules:
1. Every claim must cite at least one chunk id that appears in the context, copied exactly (e.g. c_1a2b3c4d5e6f).
2. Make each claim a single statement supported by its citations. For multi-hop answers, state each hop as its \
own claim with its own citation, then a final claim that combines them citing all hop chunks.
3. If the context does not answer the question, set answerable=false, return no claims, and explain what's \
missing in refusal_reason. Don't answer from general knowledge.
4. If the context only partially answers, answer the part it supports and say what isn't covered.
5. Aggregate lines and connection chains are summaries of the graph facts; cite the underlying facts."""


@dataclass
class GeneratedAnswer:
    draft: AnswerDraft
    usage: Usage | None
    citations_valid: bool
    regenerations: int
    failure: str | None = None

    @property
    def text(self) -> str:
        if not self.draft.answerable:
            return self.draft.refusal_reason or "The retrieved context does not answer this question."
        return " ".join(f"{c.text} {''.join(f'[{x}]' for x in c.citations)}" for c in self.draft.claims)


def generate_answer(llm: LLM, model: str, effort: str | None, question: str, ctx: Context) -> GeneratedAnswer:
    if ctx.empty:
        return GeneratedAnswer(
            AnswerDraft(
                answerable=False,
                claims=[],
                refusal_reason="No relevant graph facts or passages were retrieved.",
            ),
            usage=None,
            citations_valid=True,
            regenerations=0,
        )
    attempts = {"n": 0}

    def validator(d: AnswerDraft) -> list[str]:
        attempts["n"] += 1
        return citation_errors(d, ctx.allowed_ids)

    user = f"<context>\n{ctx.text}\n</context>\n\nQuestion: {question}"
    try:
        draft, usage = llm.structured(
            model=model,
            system=ANSWER_SYSTEM,
            user=user,
            output=AnswerDraft,
            purpose="answer",
            max_tokens=4096,
            effort=effort,
            validator=validator,
            max_retries=2,
        )
        return GeneratedAnswer(draft, usage, citations_valid=True, regenerations=attempts["n"] - 1)
    except StructuredOutputError as e:
        # Never return an answer whose citations don't resolve to retrieved chunks.
        return GeneratedAnswer(
            AnswerDraft(
                answerable=False,
                claims=[],
                refusal_reason="Could not produce an answer with valid citations.",
            ),
            usage=None,
            citations_valid=False,
            regenerations=max(attempts["n"] - 1, 0),
            failure="; ".join(e.errors),
        )
