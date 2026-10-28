from dataclasses import dataclass, field

from neo4j import Driver

from hopgraph.llm import LLM, StructuredOutputError, Usage
from hopgraph.ontology import RelType
from hopgraph.retrieval.cypher_templates import (
    TEMPLATES,
    GraphPlan,
    GraphPlans,
    TemplateError,
    build_query,
    template_catalog,
)
from hopgraph.retrieval.entity_link import EntityLinker, LinkedEntity
from hopgraph.retrieval.verbalize import Verbalized, verbalize

PLANNER_SYSTEM = f"""You translate a question about SEC filings into 1-3 calls to a fixed library of graph \
query templates. You never write queries; you choose template ids and fill parameters.

TEMPLATES
{template_catalog()}

RELATIONSHIP TYPES (direction as stored): {", ".join(r.value for r in RelType if r != RelType.CUSTOMER_OF)}
- SUPPLIES points from supplier to customer. "Who supplies X" -> neighbors(entity_a=X, rel_types=[SUPPLIES], \
direction=in). "X's customers" -> direction=out.
- SUBSIDIARY_OF points from subsidiary to parent. DIRECTOR_OF/EXECUTIVE_OF point from person to company.

PARAMETERS
- entity_a / entity_b: copy the entity name exactly as written in the question.
- Leave unused parameters null or empty.
- max_hops: 1-3.

EXAMPLES
Q: Which companies supply both NVIDIA and AMD?
-> shared_neighbors(entity_a="NVIDIA", entity_b="AMD", rel_types=[SUPPLIES])
Q: Which of Apple's directors also sit on other boards?
-> board_interlocks(entity_a="Apple")
Q: Who supplies the suppliers of Dell?
-> supply_chain(entity_a="Dell", direction=in, max_hops=2)
Q: Which companies compete with TSMC's customers?
-> two_hop(entity_a="TSMC", rel_1=SUPPLIES, rel_2=COMPETES_WITH)
Q: How is Microsoft connected to Micron?
-> path_between(entity_a="Microsoft", entity_b="Micron", max_hops=3)
Q: Compare the suppliers named by Dell and HPE.
-> neighbors(entity_a="Dell", rel_types=[SUPPLIES], direction=in), neighbors(entity_a="HPE", \
rel_types=[SUPPLIES], direction=in)
Q: Which company lists the most competitors?
-> top_by_degree(rel_types=[COMPETES_WITH], entity_type="Company")"""


@dataclass
class GraphResult:
    plans: list[GraphPlan] = field(default_factory=list)
    linked: list[LinkedEntity] = field(default_factory=list)
    verbalized: Verbalized = field(default_factory=Verbalized)
    errors: list[str] = field(default_factory=list)
    usage: Usage | None = None


class GraphRetriever:
    def __init__(self, driver: Driver, linker: EntityLinker, llm: LLM, model: str, max_facts: int = 40):
        self.driver = driver
        self.linker = linker
        self.llm = llm
        self.model = model
        self.max_facts = max_facts

    def plan(self, question: str) -> tuple[GraphPlans, Usage]:
        return self.llm.structured(
            model=self.model,
            system=PLANNER_SYSTEM,
            user=question,
            output=GraphPlans,
            purpose="graph_plan",
            max_tokens=1024,
            cache_system=False,
        )

    def _run(self, cypher: str, params: dict) -> list[dict]:
        def work(tx):
            return [r.data() for r in tx.run(cypher, params)]

        with self.driver.session() as s:
            return s.execute_read(work)

    def retrieve(self, question: str) -> GraphResult:
        result = GraphResult()
        try:
            plans, result.usage = self.plan(question)
        except StructuredOutputError as e:
            result.errors.append(f"planner failed: {e}")
            return result
        result.plans = plans.plans[:3]
        scanned = self.linker.scan(question)
        rows_all: list[dict] = []
        aggregate = False
        for plan in result.plans:
            a = self._link(plan.params.entity_a, scanned, exclude=None)
            b = (
                self._link(plan.params.entity_b, scanned, exclude=a.id if a else None)
                if "b" in TEMPLATES[plan.template_id].needs
                else None
            )
            result.linked.extend(x for x in (a, b) if x)
            try:
                cypher, params = build_query(plan, a.id if a else None, b.id if b else None)
                rows_all.extend(self._run(cypher, params))
                aggregate = aggregate or TEMPLATES[plan.template_id].aggregate
            except (TemplateError, ValueError) as e:
                result.errors.append(str(e))
        result.verbalized = verbalize(rows_all, aggregate=aggregate, max_facts=self.max_facts)
        return result

    def _link(
        self, mention: str | None, scanned: list[LinkedEntity], exclude: str | None
    ) -> LinkedEntity | None:
        if mention:
            hit = self.linker.link(mention)
            if hit:
                return hit
        # the planner's mention didn't resolve; fall back to names found verbatim in the question
        return next((s for s in scanned if s.id != exclude), None) if mention else None
