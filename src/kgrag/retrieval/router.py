from enum import StrEnum

from pydantic import BaseModel, Field

from kgrag.llm import LLM, Usage


class Route(StrEnum):
    GRAPH = "graph"
    VECTOR = "vector"
    HYBRID = "hybrid"


class RouteDecision(BaseModel):
    route: Route
    confidence: float = Field(description="0.0-1.0")
    reason: str = Field(description="One short sentence")


ROUTER_SYSTEM = """You route questions about SEC filings of semiconductor and AI-infrastructure companies to a \
retrieval backend.

- graph: questions about connections between entities: who supplies/competes with/acquired whom, board \
members and where else they serve, multi-hop chains ("suppliers of X's customers"), comparisons of \
relationships across entities, and counts/rankings over relationships ("which company names the most \
suppliers").
- vector: definitions, policies, risk descriptions, strategy, financial figures, and single facts stated in \
one passage ("What does NVIDIA say about export controls?", "What was Intel's R&D spend?").
- hybrid: needs both a relationship lookup and descriptive text ("What risks does AMD describe about its \
foundry supplier?"), or you are unsure.

Examples:
Q: Which companies supply both NVIDIA and AMD? -> graph, 0.95
Q: Who sits on Apple's board and also serves on another company's board in this corpus? -> graph, 0.9
Q: How does Micron describe the cyclicality of the memory market? -> vector, 0.9
Q: What is CUDA? -> vector, 0.85
Q: What does Broadcom say about the risks of its dependence on its largest customer? -> hybrid, 0.8
Q: Which company has the most subsidiaries listed? -> graph, 0.85
Q: What is the weather in Austin? -> vector, 0.5

Return the route, your confidence, and a one-sentence reason."""


def route(llm: LLM, model: str, question: str, threshold: float) -> tuple[RouteDecision, Route, Usage]:
    """Returns (model decision, effective route, usage). Low-confidence decisions fall back to hybrid."""
    decision, usage = llm.structured(
        model=model,
        system=ROUTER_SYSTEM,
        user=question,
        output=RouteDecision,
        purpose="route",
        max_tokens=256,
        cache_system=False,
    )
    effective = decision.route if decision.confidence >= threshold else Route.HYBRID
    return decision, effective, usage
