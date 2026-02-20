"""
Affect-to-Strategy Mapping with Confidence-based Fallback.


This module implements:
- Mapping from affect labels to pedagogical strategies
- Confidence-based fallback mechanism for low-confidence predictions
- Handling of ambiguous affect pairs (e.g., boredom vs frustration)

Equation (8): When conf(ŷ) < τ OR ŷ ∈ {boredom, frustration}, use neutral strategy
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Set
import torch
import torch.nn.functional as F


# Confidence threshold τ = 0.65
CONFIDENCE_THRESHOLD = 0.65

# Affect labels (4-class classification)
AFFECT_LABELS = ["frustration", "confusion", "boredom", "engagement"]

# ambiguous pairs that can be confused
AMBIGUOUS_PAIRS: Set[frozenset] = {
    frozenset({"boredom", "frustration"}),
    frozenset({"confusion", "frustration"}),
}


@dataclass
class Strategy:
    """Pedagogical strategy + stylistic guidance used by the GenAI engine."""
    name: str
    style: str
    pedagogy: str
    is_fallback: bool = False  # True if fallback was triggered


def get_neutral_strategy() -> Strategy:
    """
    Neutral/fallback strategy for low-confidence or ambiguous predictions.


    This strategy is used when:
    1. Classifier confidence < CONFIDENCE_THRESHOLD (0.65)
    2. Prediction is in an ambiguous pair (e.g., boredom vs frustration)
    3. Confidence gap between top-2 predictions is too small

    The neutral strategy provides supportive, non-presumptive guidance.
    """
    return Strategy(
        name="NeutralSupport",
        style="warm, patient, encouraging, non-presumptive",
        pedagogy=(
            "check in on learner state; ask if they need clarification or a break; "
            "offer options (retry, hint, skip); maintain positive rapport; "
            "avoid assumptions about the learner's emotional state"
        ),
        is_fallback=True,
    )


def map_affect_to_strategy(label: str) -> Strategy:
    """
    Map an affect label to an actionable tutoring strategy.

    This is the basic mapping without confidence checking.
    For production use, prefer map_affect_to_strategy_with_confidence().

    Parameters
    ----------
    label : str
        One of: "frustration", "confusion", "boredom", "engagement"

    Returns
    -------
    Strategy
        Corresponding pedagogical strategy.
    """
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


def is_ambiguous_pair(label1: str, label2: str) -> bool:
    """
    Check if two labels form an ambiguous pair.
    certain affect pairs
    (e.g., boredom vs frustration) are frequently confused.
    """
    pair = frozenset({label1.lower(), label2.lower()})
    return pair in AMBIGUOUS_PAIRS


def map_affect_to_strategy_with_confidence(
    logits: torch.Tensor,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ambiguity_margin: float = 0.15,
) -> Tuple[Strategy, dict]:
    """
    Map affect prediction to strategy with confidence-based fallback.


    This implements the confidence-based fallback mechanism:
    1. If max confidence < threshold, use neutral strategy
    2. If top-2 predictions are ambiguous pair with small margin, use neutral
    3. Otherwise, use the strategy for the top prediction

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits from classifier of shape (C,) or (B, C) where C=4.
    confidence_threshold : float
        Minimum confidence to trust prediction. Default 0.65.
    ambiguity_margin : float
        Minimum margin between top-2 predictions for non-ambiguous pairs.

    Returns
    -------
    Tuple[Strategy, dict]
        Strategy and metadata dict with:
        - "predicted_label": str
        - "confidence": float
        - "fallback_reason": Optional[str]
        - "top2_labels": List[str]
        - "all_probs": List[float]
    """
    # Handle batch dimension
    if logits.dim() == 2:
        logits = logits[0]  # Take first sample

    # Compute probabilities
    probs = F.softmax(logits, dim=-1)

    # Get top-2 predictions
    top2_probs, top2_indices = torch.topk(probs, k=2)
    top1_prob = top2_probs[0].item()
    top2_prob = top2_probs[1].item()

    top1_label = AFFECT_LABELS[top2_indices[0].item()]
    top2_label = AFFECT_LABELS[top2_indices[1].item()]

    # Build metadata
    metadata = {
        "predicted_label": top1_label,
        "confidence": top1_prob,
        "top2_labels": [top1_label, top2_label],
        "top2_probs": [top1_prob, top2_prob],
        "all_probs": probs.tolist(),
        "fallback_reason": None,
    }

    # Check fallback conditions per Equation (8)

    # Condition 1: Low confidence
    if top1_prob < confidence_threshold:
        metadata["fallback_reason"] = f"low_confidence ({top1_prob:.2f} < {confidence_threshold})"
        return get_neutral_strategy(), metadata

    # Condition 2: Ambiguous pair with small margin
    margin = top1_prob - top2_prob
    if is_ambiguous_pair(top1_label, top2_label) and margin < ambiguity_margin:
        metadata["fallback_reason"] = (
            f"ambiguous_pair ({top1_label}/{top2_label}) "
            f"with small margin ({margin:.2f} < {ambiguity_margin})"
        )
        return get_neutral_strategy(), metadata

    # No fallback needed - use primary strategy
    return map_affect_to_strategy(top1_label), metadata


def map_batch_to_strategies(
    logits: torch.Tensor,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Tuple[list, list]:
    """
    Map a batch of predictions to strategies.

    Parameters
    ----------
    logits : torch.Tensor
        Batch of logits of shape (B, C).
    confidence_threshold : float
        Confidence threshold for fallback.

    Returns
    -------
    Tuple[list, list]
        List of strategies and list of metadata dicts.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    strategies = []
    all_metadata = []

    for i in range(logits.shape[0]):
        strategy, metadata = map_affect_to_strategy_with_confidence(
            logits[i],
            confidence_threshold=confidence_threshold,
        )
        strategies.append(strategy)
        all_metadata.append(metadata)

    return strategies, all_metadata


def get_all_strategies() -> dict:
    """Return all available strategies keyed by affect label."""
    return {
        "frustration": map_affect_to_strategy("frustration"),
        "confusion": map_affect_to_strategy("confusion"),
        "boredom": map_affect_to_strategy("boredom"),
        "engagement": map_affect_to_strategy("engagement"),
        "neutral": get_neutral_strategy(),
    }


class AdaptiveStrategySelector:
    """
    Adaptive strategy selection with temporal smoothing.


    This class maintains history of predictions and applies temporal
    smoothing to avoid rapid strategy switches that could confuse learners.
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        smoothing_window: int = 3,
        switch_threshold: float = 0.7,
    ):
        """
        Parameters
        ----------
        confidence_threshold : float
            Confidence threshold for fallback.
        smoothing_window : int
            Number of recent predictions to consider.
        switch_threshold : float
            Minimum proportion of window agreeing to switch strategy.
        """
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.switch_threshold = switch_threshold

        self.history: list = []
        self.current_strategy: Optional[Strategy] = None

    def update(self, logits: torch.Tensor) -> Tuple[Strategy, dict]:
        """
        Update with new prediction and return (possibly smoothed) strategy.

        Parameters
        ----------
        logits : torch.Tensor
            Current prediction logits.

        Returns
        -------
        Tuple[Strategy, dict]
            Strategy and metadata.
        """
        strategy, metadata = map_affect_to_strategy_with_confidence(
            logits,
            confidence_threshold=self.confidence_threshold,
        )

        # Add to history
        self.history.append(metadata["predicted_label"])
        if len(self.history) > self.smoothing_window:
            self.history.pop(0)

        # Check if we should switch strategy
        if self.current_strategy is None:
            self.current_strategy = strategy
        elif len(self.history) >= self.smoothing_window:
            # Count votes for each label
            label_counts = {}
            for label in self.history:
                label_counts[label] = label_counts.get(label, 0) + 1

            # Get majority label
            majority_label = max(label_counts, key=label_counts.get)
            majority_prop = label_counts[majority_label] / len(self.history)

            # Switch if majority exceeds threshold
            if majority_prop >= self.switch_threshold:
                self.current_strategy = map_affect_to_strategy(majority_label)
                metadata["smoothed_label"] = majority_label

        metadata["applied_smoothing"] = len(self.history) >= self.smoothing_window

        return self.current_strategy, metadata

    def reset(self):
        """Reset history and current strategy."""
        self.history = []
        self.current_strategy = None
