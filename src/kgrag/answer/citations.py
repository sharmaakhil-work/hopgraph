from pydantic import BaseModel, Field


class Claim(BaseModel):
    text: str = Field(description="One factual statement, no inline citation markers")
    citations: list[str] = Field(
        description="Chunk ids (e.g. c_1a2b3c4d5e6f) from the context that support it"
    )


class AnswerDraft(BaseModel):
    answerable: bool = Field(description="False when the context does not contain the answer")
    claims: list[Claim]
    refusal_reason: str | None = Field(default=None, description="Only when answerable is false")


def citation_errors(draft: AnswerDraft, allowed_ids: set[str]) -> list[str]:
    """Every claim needs at least one citation, and every citation must be a chunk that was actually retrieved."""
    errors: list[str] = []
    if not draft.answerable:
        if draft.claims:
            errors.append("answerable is false, so claims must be empty")
        return errors
    if not draft.claims:
        errors.append("answerable is true but there are no claims")
    for i, c in enumerate(draft.claims):
        if not c.citations:
            errors.append(f"claims[{i}] has no citations; cite chunk ids from the context or drop the claim")
        bad = [x for x in c.citations if x not in allowed_ids]
        if bad:
            errors.append(
                f"claims[{i}] cites ids not present in the context: {bad}. "
                "Only cite ids shown in square brackets in the context."
            )
    return errors
