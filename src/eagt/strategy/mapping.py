from dataclasses import dataclass

AFFECT_LABELS = ["frustration", "confusion", "boredom", "engagement"]

@dataclass
class Strategy:
    """Pedagogical strategy + stylistic guidance used by the GenAI engine."""
    name: str
    style: str
    pedagogy: str

def map_affect_to_strategy(label: str) -> Strategy:
    """Map an affect label to an actionable tutoring strategy."""
    label = (label or "").strip().lower()

    if label == "frustration":
        return Strategy(
            name="ScaffoldAndEncourage",
            style="calm, supportive, step-by-step, short sentences",
            pedagogy=(
                "reduce complexity; decompose into atomic steps; normalize struggle; "
                "provide immediate small wins; add explicit checks for understanding"
            ),
        )

    if label == "confusion":
        return Strategy(
            name="ClarifyWithAnalogy",
            style="clear, precise, analogy-driven, Socratic checks",
            pedagogy=(
                "restate the goal; surface the misconception; use a concrete analogy; "
                "show a minimal working example; prompt the learner to predict next steps"
            ),
        )

    if label == "boredom":
        return Strategy(
            name="GamifyAndNovelty",
            style="brisk, challenge-oriented, slightly playful",
            pedagogy=(
                "introduce novelty or a real-world hook; set a micro-goal with a timer; "
                "raise desirable difficulty; alternate question formats"
            ),
        )

    # default → treat as engaged
    return Strategy(
        name="EnrichAndExtend",
        style="confident, exploratory, deeper-dive",
        pedagogy=(
            "connect to applications; add extension problems; encourage self-explanation; "
            "compare multiple solution paths"
        ),
    )
