"""Run hopgraph and the vector-only baseline on the same questions and publish the delta by hop count.

The baseline is the same pipeline with the route forced to `vector`: same embeddings, top-k, answer model and
citation validator. Only retrieval differs.
"""

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from hopgraph.evaluation.judge import score
from hopgraph.evaluation.questions import CATEGORIES
from hopgraph.llm import LLM
from hopgraph.pipeline import Pipeline

SYSTEMS = {"hopgraph": "auto", "vector baseline": "vector"}
LABELS = {
    "single_hop": "Single hop",
    "two_hop": "2 hop",
    "three_hop": "3 hop",
    "aggregation": "Aggregation",
    "out_of_scope": "Out of scope (refusal)",
}
README_START, README_END = "<!-- BENCHMARK:START -->", "<!-- BENCHMARK:END -->"


def run(
    pipeline: Pipeline, judge_llm: LLM, judge_model: str, questions: list[dict], out_path: Path
) -> list[dict]:
    rows = []
    for i, q in enumerate(questions, 1):
        for system, mode in SYSTEMS.items():
            r = pipeline.answer(q["question"], mode=mode, system=system)
            s = score(judge_llm, judge_model, q, r.answer, r.answerable)
            rows.append(
                {
                    "id": q["id"],
                    "category": q["category"],
                    "system": system,
                    "route": r.route,
                    "answerable": r.answerable,
                    "answer": r.answer,
                    "latency_ms": r.latency_ms,
                    "cost_usd": r.cost_usd,
                    "citations_valid": r.citations_valid,
                    **s,
                }
            )
        print(
            f"  {i}/{len(questions)} {q['id']}: "
            + ", ".join(f"{x['system']}={x['score']}" for x in rows[-len(SYSTEMS) :])
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r) for r in rows))
    return rows


def _pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))] if s else 0.0


def summarize(rows: list[dict], ingest_cost: dict) -> dict:
    by: dict[tuple[str, str], list[float]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        by[(r["system"], r["category"])].append(r["score"])
    for r in rows:
        if r["system"] == next(iter(SYSTEMS)):
            counts[r["category"]] += 1
    summary = {"accuracy": {}, "n": dict(counts), "latency": {}, "cost": {}, "ingest": ingest_cost}
    for system in SYSTEMS:
        summary["accuracy"][system] = {
            c: statistics.mean(by[(system, c)]) for c in CATEGORIES if by[(system, c)]
        }
        multi = [x for c in ("two_hop", "three_hop") for x in by[(system, c)]]
        summary["accuracy"][system]["multi_hop"] = statistics.mean(multi) if multi else None
        lat = [r["latency_ms"] for r in rows if r["system"] == system]
        summary["latency"][system] = {"p50": _pct(lat, 50), "p95": _pct(lat, 95)}
        summary["cost"][system] = statistics.mean(r["cost_usd"] for r in rows if r["system"] == system)
    return summary


def markdown_table(summary: dict) -> str:
    kg, base = "hopgraph", "vector baseline"
    acc = summary["accuracy"]
    lines = [
        "| Question type | n | Vector RAG | hopgraph (graph + vector) | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in CATEGORIES:
        if c not in acc[kg]:
            continue
        b, k = acc[base][c], acc[kg][c]
        lines.append(
            f"| {LABELS[c]} | {summary['n'].get(c, 0)} | {b:.0%} | {k:.0%} | {(k - b) * 100:+.0f} pts |"
        )
    lat, cost, ing = summary["latency"], summary["cost"], summary["ingest"]
    lines += [
        f"| **p50 / p95 latency** | | {lat[base]['p50'] / 1000:.1f}s / {lat[base]['p95'] / 1000:.1f}s | "
        f"{lat[kg]['p50'] / 1000:.1f}s / {lat[kg]['p95'] / 1000:.1f}s | |",
        f"| **Cost per query** | | ${cost[base]:.4f} | ${cost[kg]:.4f} | {cost[kg] / max(cost[base], 1e-9):.1f}x |",
        f"| **One-time ingestion** | | ${ing.get('vector_usd', 0):.2f} (local embeddings) | "
        f"${ing.get('graph_usd', 0):.2f} (Claude extraction) | |",
    ]
    return "\n".join(lines)


def chart(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    cats = [c for c in CATEGORIES if c in summary["accuracy"]["hopgraph"]]
    series = [
        ("vector baseline", "Vector RAG", "#eb6834"),
        ("hopgraph", "hopgraph (graph + vector)", "#2a78d6"),
    ]
    text_primary, text_secondary, grid = "#0b0b0b", "#52514e", "#e4e3df"

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    bar_w, gap = 0.3, 0.02
    for si, (key, label, color) in enumerate(series):
        for ci, c in enumerate(cats):
            v = summary["accuracy"][key][c] * 100
            x = ci + (si - 0.5) * (bar_w + gap)
            ax.add_patch(
                FancyBboxPatch(
                    (x - bar_w / 2, 0),
                    bar_w,
                    max(v, 0.01),
                    boxstyle="round,pad=0,rounding_size=0.04",
                    mutation_aspect=100 / 3.6,
                    linewidth=0,
                    facecolor=color,
                    label=label if ci == 0 else None,
                )
            )
            # square off the baseline end; only the data end stays rounded
            ax.add_patch(Rectangle((x - bar_w / 2, 0), bar_w, v / 2, linewidth=0, facecolor=color))
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=text_primary)
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.set_ylim(0, 110)
    ax.set_xticks(
        range(len(cats)),
        [LABELS[c].replace(" (refusal)", "") for c in cats],
        fontsize=8,
        color=text_secondary,
    )
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"], fontsize=7.5, color=text_secondary)
    ax.yaxis.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(length=0)
    ax.set_title("Answer accuracy by question type", loc="left", fontsize=10, color=text_primary, pad=14)
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=8,
        ncols=2,
        bbox_to_anchor=(1, 1.13),
        labelcolor=text_secondary,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def update_readme(readme: Path, table: str, chart_rel: str, n_docs: int) -> None:
    block = (
        f"{README_START}\n{table}\n\n![Accuracy by question type]({chart_rel})\n\n"
        f"<sub>{n_docs} SEC filings. Scores: correct = 1, partial = 0.5. Out-of-scope questions score only "
        f"when the system refuses. Full per-question results in `data/reports/benchmark.jsonl`.</sub>\n{README_END}"
    )
    text = readme.read_text()
    pattern = re.compile(re.escape(README_START) + ".*?" + re.escape(README_END), re.S)
    readme.write_text(pattern.sub(lambda _: block, text) if pattern.search(text) else block + "\n\n" + text)


def ingestion_costs(ledger: Path) -> dict:
    graph = 0.0
    if ledger.exists():
        for line in ledger.open():
            row = json.loads(line)
            if row["purpose"].startswith("extract"):
                graph += row["usd"]
    return {"graph_usd": graph, "vector_usd": 0.0}
