"""
experiment.py — Voice A/B testing framework
============================================
Simple framework for comparing TTS voice variants across calls.
Tracks which variant was used per call and stores results for analysis.

Usage:
    from experiment import get_active_experiment, VoiceVariant

    experiment = get_active_experiment()
    variant = experiment.pick_variant()
    # variant.speaker = "shubh", variant.pace = 1.0, etc.
"""

import json
import logging
import os
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("experiment")

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"


@dataclass
class VoiceVariant:
    """A single TTS voice configuration to test."""
    speaker: str
    pace: float = 1.0
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.speaker}-p{self.pace}"


@dataclass
class VoiceExperiment:
    """An A/B experiment comparing multiple voice variants."""
    name: str
    variants: list[VoiceVariant] = field(default_factory=list)
    active: bool = True

    def pick_variant(self) -> VoiceVariant:
        """Randomly select a variant for this call."""
        if not self.variants:
            return VoiceVariant(speaker="shubh", label="shubh-default")
        return random.choice(self.variants)


@dataclass
class ExperimentResult:
    """Result of a single call with a specific voice variant."""
    experiment_name: str
    variant_label: str
    room_name: str
    store_name: str
    timestamp: str = ""
    quality_score: float = 0.0
    topics_covered: list[str] = field(default_factory=list)
    transcript_path: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# Default: single-variant baseline (no experiment, just "shubh")
DEFAULT_EXPERIMENT = VoiceExperiment(
    name="baseline",
    variants=[VoiceVariant(speaker="shubh", label="shubh-baseline")],
)

# Active experiment — change this to run A/B tests
# Example multi-variant experiment:
# ACTIVE_EXPERIMENT = VoiceExperiment(
#     name="speaker-comparison-v1",
#     variants=[
#         VoiceVariant(speaker="shubh", label="shubh-control"),
#         VoiceVariant(speaker="ritu", label="ritu-female"),
#         VoiceVariant(speaker="kabir", label="kabir-alt-male"),
#     ],
# )
ACTIVE_EXPERIMENT = DEFAULT_EXPERIMENT


def get_active_experiment() -> VoiceExperiment:
    """Return the currently active voice experiment."""
    return ACTIVE_EXPERIMENT


def record_result(result: ExperimentResult):
    """Save experiment result to experiments/ directory."""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    results_file = EXPERIMENTS_DIR / f"{result.experiment_name}.jsonl"
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    logger.info(f"Recorded experiment result: {result.variant_label} for {result.store_name}")


def load_results(experiment_name: str) -> list[ExperimentResult]:
    """Load all results for a given experiment."""
    results_file = EXPERIMENTS_DIR / f"{experiment_name}.jsonl"
    if not results_file.exists():
        return []
    results = []
    for line in results_file.read_text().splitlines():
        if line.strip():
            data = json.loads(line)
            results.append(ExperimentResult(**data))
    return results


def summarize_experiment(experiment_name: str) -> dict:
    """Summarize results by variant for a given experiment."""
    results = load_results(experiment_name)
    if not results:
        return {"experiment": experiment_name, "variants": {}, "total_calls": 0}

    by_variant = {}
    for r in results:
        if r.variant_label not in by_variant:
            by_variant[r.variant_label] = {"scores": [], "call_count": 0}
        by_variant[r.variant_label]["scores"].append(r.quality_score)
        by_variant[r.variant_label]["call_count"] += 1

    summary = {}
    for label, data in by_variant.items():
        scores = data["scores"]
        summary[label] = {
            "call_count": data["call_count"],
            "avg_score": round(sum(scores) / len(scores), 3) if scores else 0,
            "min_score": round(min(scores), 3) if scores else 0,
            "max_score": round(max(scores), 3) if scores else 0,
        }

    return {
        "experiment": experiment_name,
        "variants": summary,
        "total_calls": len(results),
    }
