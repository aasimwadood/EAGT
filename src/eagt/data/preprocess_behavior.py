"""
Behavioral Feature Extraction for EAGT.


This module extracts behavioral features from learner interaction logs:
- Response latency
- Hint request frequency
- Scroll velocity
- Backtrack count
- Idle duration
- Click frequency
- Keystroke latency

All features are min-max normalized to [0, 1] range.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class BehaviorFeatureRanges:
    """
    Min-max ranges for behavioral feature normalization.
    min-max normalization.
    These defaults are based on typical learning interaction patterns.
    """
    response_latency: Tuple[float, float] = (0.0, 60.0)      # 0-60 seconds
    hint_request_freq: Tuple[float, float] = (0.0, 5.0)      # 0-5 hints per window
    scroll_velocity: Tuple[float, float] = (0.0, 1000.0)     # pixels per second
    backtrack_count: Tuple[float, float] = (0.0, 10.0)       # 0-10 backtracks
    idle_duration: Tuple[float, float] = (0.0, 30.0)         # 0-30 seconds
    click_frequency: Tuple[float, float] = (0.0, 3.0)        # clicks per second
    keystroke_latency: Tuple[float, float] = (0.0, 2.0)      # 0-2 seconds between keys

    def to_dict(self) -> Dict[str, Tuple[float, float]]:
        return {
            "response_latency": self.response_latency,
            "hint_request_freq": self.hint_request_freq,
            "scroll_velocity": self.scroll_velocity,
            "backtrack_count": self.backtrack_count,
            "idle_duration": self.idle_duration,
            "click_frequency": self.click_frequency,
            "keystroke_latency": self.keystroke_latency,
        }


@dataclass
class BehaviorFeatures:
    """
    Container for extracted behavioral features.

    """
    response_latency: float = 0.0
    hint_request_freq: float = 0.0
    scroll_velocity: float = 0.0
    backtrack_count: float = 0.0
    idle_duration: float = 0.0
    click_frequency: float = 0.0
    keystroke_latency: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array in consistent order."""
        return np.array([
            self.response_latency,
            self.hint_request_freq,
            self.scroll_velocity,
            self.backtrack_count,
            self.idle_duration,
            self.click_frequency,
            self.keystroke_latency,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "response_latency",
            "hint_request_freq",
            "scroll_velocity",
            "backtrack_count",
            "idle_duration",
            "click_frequency",
            "keystroke_latency",
        ]

    @staticmethod
    def feature_dim() -> int:
        return 7


def normalize_feature(value: float, min_val: float, max_val: float) -> float:
    """Min-max normalize a single feature to [0, 1]."""
    value = np.clip(value, min_val, max_val)
    return (value - min_val) / (max_val - min_val + 1e-8)


def extract_behavioral_features(
    events: List[Dict],
    window_start: float,
    window_end: float,
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    idle_threshold: float = 3.0,
) -> np.ndarray:
    """
    Extract behavioral features from interaction log events for a time window.


    This function processes raw interaction events (clicks, keystrokes, scrolls,
    hint requests, etc.) and computes aggregate behavioral features that indicate
    learner engagement and affect states.

    Parameters
    ----------
    events : List[Dict]
        List of event dictionaries. Each event should have:
        - "timestamp": float, time in seconds
        - "type": str, one of "click", "keystroke", "scroll", "hint_request",
                  "navigation", "response_submit", "idle"
        - "value": dict, event-specific data (e.g., {"delta_y": 100} for scroll)

    window_start : float
        Start time of the window in seconds.

    window_end : float
        End time of the window in seconds.

    feature_ranges : Dict[str, Tuple[float, float]], optional
        Min-max ranges for normalization. Uses defaults if not provided.

    idle_threshold : float
        Seconds of inactivity to count as idle. Default 3.0 seconds.

    Returns
    -------
    np.ndarray
        Normalized feature vector of shape (7,) containing:
        [response_latency, hint_request_freq, scroll_velocity,
         backtrack_count, idle_duration, click_frequency, keystroke_latency]

    Examples
    --------
    >>> events = [
    ...     {"timestamp": 0.5, "type": "click", "value": {}},
    ...     {"timestamp": 1.0, "type": "keystroke", "value": {"key": "a"}},
    ...     {"timestamp": 1.5, "type": "keystroke", "value": {"key": "b"}},
    ...     {"timestamp": 2.0, "type": "hint_request", "value": {}},
    ...     {"timestamp": 8.0, "type": "response_submit", "value": {"latency": 8.0}},
    ... ]
    >>> features = extract_behavioral_features(events, 0.0, 10.0)
    >>> features.shape
    (7,)
    """
    if feature_ranges is None:
        feature_ranges = BehaviorFeatureRanges().to_dict()

    # Filter events to window
    window_events = [
        e for e in events
        if window_start <= e.get("timestamp", 0) <= window_end
    ]

    window_duration = max(0.1, window_end - window_start)

    # Initialize raw feature accumulators
    raw_features = BehaviorFeatures()

    # Accumulators for computing features
    hint_count = 0
    click_count = 0
    scroll_distances: List[float] = []
    keystroke_times: List[float] = []
    backtrack_count = 0
    last_activity_time = window_start
    idle_total = 0.0

    # Sort events by timestamp
    window_events = sorted(window_events, key=lambda e: e.get("timestamp", 0))

    for event in window_events:
        etype = event.get("type", "").lower()
        timestamp = event.get("timestamp", 0)
        value = event.get("value", {})

        # Track idle time (gap between activities)
        gap = timestamp - last_activity_time
        if gap > idle_threshold:
            idle_total += gap
        last_activity_time = timestamp

        # Process by event type
        if etype == "hint_request":
            hint_count += 1

        elif etype == "click":
            click_count += 1

        elif etype == "scroll":
            delta_y = abs(value.get("delta_y", 0))
            delta_x = abs(value.get("delta_x", 0))
            scroll_distances.append(delta_y + delta_x)

        elif etype == "keystroke":
            keystroke_times.append(timestamp)

        elif etype == "navigation":
            direction = value.get("direction", "").lower()
            if direction in ("back", "backward", "previous"):
                backtrack_count += 1

        elif etype == "response_submit":
            # Response latency from problem presentation to submission
            latency = value.get("latency", 0)
            if latency > 0:
                raw_features.response_latency = latency

        elif etype == "problem_start":
            # Mark the start of a problem for latency calculation
            pass

    # Compute aggregate features
    raw_features.hint_request_freq = hint_count
    raw_features.click_frequency = click_count / window_duration
    raw_features.scroll_velocity = np.mean(scroll_distances) if scroll_distances else 0.0
    raw_features.backtrack_count = backtrack_count
    raw_features.idle_duration = idle_total

    # Compute average keystroke latency
    if len(keystroke_times) > 1:
        diffs = np.diff(sorted(keystroke_times))
        raw_features.keystroke_latency = np.mean(diffs)

    # Normalize all features
    normalized = []
    feature_names = BehaviorFeatures.feature_names()

    for name in feature_names:
        raw_value = getattr(raw_features, name)
        vmin, vmax = feature_ranges.get(name, (0.0, 1.0))
        normalized.append(normalize_feature(raw_value, vmin, vmax))

    return np.array(normalized, dtype=np.float32)


def extract_behavioral_features_sequence(
    events: List[Dict],
    window_sec: float = 1.0,
    stride_sec: float = 0.5,
    total_duration: Optional[float] = None,
    num_windows: Optional[int] = None,
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Extract behavioral features for a sequence of time windows.
    Temporal Synchronization

    Parameters
    ----------
    events : List[Dict]
        List of interaction events.
    window_sec : float
        Duration of each window in seconds.
    stride_sec : float
        Stride between windows in seconds.
    total_duration : float, optional
        Total duration of the session. If not provided, inferred from events.
    num_windows : int, optional
        Number of windows to extract. If provided, overrides stride calculation.
    feature_ranges : Dict, optional
        Normalization ranges.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (T, D) where T is number of windows and D is feature dim.
    """
    if not events:
        # Return zeros if no events
        n_windows = num_windows or 16
        return np.zeros((n_windows, BehaviorFeatures.feature_dim()), dtype=np.float32)

    # Determine total duration
    if total_duration is None:
        timestamps = [e.get("timestamp", 0) for e in events]
        total_duration = max(timestamps) if timestamps else window_sec

    # Determine number of windows
    if num_windows is not None:
        n_windows = num_windows
        stride_sec = max(0.1, (total_duration - window_sec) / max(1, n_windows - 1))
    else:
        n_windows = max(1, int((total_duration - window_sec) / stride_sec) + 1)

    # Extract features for each window
    features = []
    for i in range(n_windows):
        start = i * stride_sec
        end = start + window_sec
        feat = extract_behavioral_features(events, start, end, feature_ranges)
        features.append(feat)

    return np.stack(features, axis=0)


def load_behavior_events(
    source: Union[str, Path, List[Dict], dict]
) -> List[Dict]:
    """
    Load behavioral events from various sources.

    Parameters
    ----------
    source : str, Path, List[Dict], or dict
        - If str/Path: path to JSON file
        - If List[Dict]: events directly
        - If dict: expected to have "events" key

    Returns
    -------
    List[Dict]
        List of event dictionaries.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get("events", [])
        return []

    elif isinstance(source, list):
        return source

    elif isinstance(source, dict):
        return source.get("events", [])

    return []


def compute_engagement_indicators(features: np.ndarray) -> Dict[str, float]:
    """
    Compute high-level engagement indicators from behavioral features.
    how behavioral traces
    indicate engagement patterns.

    Parameters
    ----------
    features : np.ndarray
        Behavioral feature array of shape (D,) or (T, D).

    Returns
    -------
    Dict with engagement indicator scores.
    """
    if features.ndim == 1:
        features = features.reshape(1, -1)

    # Average over time
    avg_features = features.mean(axis=0)

    # Feature indices
    idx = {name: i for i, name in enumerate(BehaviorFeatures.feature_names())}

    indicators = {}

    # High response latency + high hint requests = likely confusion/frustration
    indicators["struggle_score"] = (
        avg_features[idx["response_latency"]] * 0.4 +
        avg_features[idx["hint_request_freq"]] * 0.4 +
        avg_features[idx["backtrack_count"]] * 0.2
    )

    # High idle + low clicks = likely boredom/disengagement
    indicators["disengagement_score"] = (
        avg_features[idx["idle_duration"]] * 0.5 +
        (1.0 - avg_features[idx["click_frequency"]]) * 0.3 +
        (1.0 - avg_features[idx["keystroke_latency"]]) * 0.2
    )

    # Active interaction pattern = likely engagement
    indicators["engagement_score"] = (
        avg_features[idx["click_frequency"]] * 0.3 +
        (1.0 - avg_features[idx["idle_duration"]]) * 0.3 +
        avg_features[idx["scroll_velocity"]] * 0.2 +
        (1.0 - avg_features[idx["keystroke_latency"]]) * 0.2
    )

    return indicators


class BehaviorFeatureExtractor:
    """
    Class-based interface for behavioral feature extraction.

    Maintains state for feature ranges and provides convenient methods
    for batch processing.
    """

    def __init__(
        self,
        feature_ranges: Optional[BehaviorFeatureRanges] = None,
        window_sec: float = 1.0,
        stride_sec: float = 0.5,
        idle_threshold: float = 3.0,
    ):
        self.feature_ranges = feature_ranges or BehaviorFeatureRanges()
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.idle_threshold = idle_threshold

    def extract(
        self,
        events: List[Dict],
        window_start: float,
        window_end: float,
    ) -> np.ndarray:
        """Extract features for a single window."""
        return extract_behavioral_features(
            events,
            window_start,
            window_end,
            self.feature_ranges.to_dict(),
            self.idle_threshold,
        )

    def extract_sequence(
        self,
        events: List[Dict],
        total_duration: Optional[float] = None,
        num_windows: Optional[int] = None,
    ) -> np.ndarray:
        """Extract features for a sequence of windows."""
        return extract_behavioral_features_sequence(
            events,
            self.window_sec,
            self.stride_sec,
            total_duration,
            num_windows,
            self.feature_ranges.to_dict(),
        )

    @property
    def feature_dim(self) -> int:
        return BehaviorFeatures.feature_dim()

    @property
    def feature_names(self) -> List[str]:
        return BehaviorFeatures.feature_names()
