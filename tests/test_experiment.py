"""Tests for experiment.py — Voice A/B testing framework."""

import json
import pytest
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiment import (
    VoiceVariant,
    VoiceExperiment,
    ExperimentResult,
    get_active_experiment,
    record_result,
    load_results,
    summarize_experiment,
    DEFAULT_EXPERIMENT,
    EXPERIMENTS_DIR,
)


class TestVoiceVariant:
    def test_default_label(self):
        v = VoiceVariant(speaker="shubh")
        assert v.label == "shubh-p1.0"

    def test_custom_label(self):
        v = VoiceVariant(speaker="ritu", label="ritu-female")
        assert v.label == "ritu-female"

    def test_pace(self):
        v = VoiceVariant(speaker="shubh", pace=1.2)
        assert v.pace == 1.2
        assert v.label == "shubh-p1.2"


class TestVoiceExperiment:
    def test_pick_variant(self):
        exp = VoiceExperiment(
            name="test",
            variants=[VoiceVariant(speaker="shubh"), VoiceVariant(speaker="ritu")],
        )
        variant = exp.pick_variant()
        assert variant.speaker in ("shubh", "ritu")

    def test_pick_variant_empty_returns_default(self):
        exp = VoiceExperiment(name="empty", variants=[])
        variant = exp.pick_variant()
        assert variant.speaker == "shubh"

    def test_single_variant_always_picked(self):
        exp = VoiceExperiment(
            name="single",
            variants=[VoiceVariant(speaker="kabir", label="kabir-only")],
        )
        for _ in range(10):
            assert exp.pick_variant().label == "kabir-only"


class TestDefaultExperiment:
    def test_default_is_baseline(self):
        assert DEFAULT_EXPERIMENT.name == "baseline"
        assert len(DEFAULT_EXPERIMENT.variants) == 1
        assert DEFAULT_EXPERIMENT.variants[0].speaker == "shubh"

    def test_get_active_experiment(self):
        exp = get_active_experiment()
        assert exp.name == "baseline"


class TestExperimentResult:
    def test_auto_timestamp(self):
        r = ExperimentResult(
            experiment_name="test",
            variant_label="shubh-baseline",
            room_name="room-123",
            store_name="Test Store",
        )
        assert r.timestamp  # auto-filled


class TestRecordAndLoad:
    def test_record_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr("experiment.EXPERIMENTS_DIR", tmp_path)

        result = ExperimentResult(
            experiment_name="test-exp",
            variant_label="shubh-baseline",
            room_name="room-abc",
            store_name="Test Store",
            quality_score=0.85,
        )
        record_result(result)

        loaded = load_results("test-exp")
        assert len(loaded) == 1
        assert loaded[0].variant_label == "shubh-baseline"
        assert loaded[0].quality_score == 0.85

    def test_load_nonexistent_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("experiment.EXPERIMENTS_DIR", tmp_path)
        loaded = load_results("does-not-exist")
        assert loaded == []


class TestSummarize:
    def test_summarize_experiment(self, tmp_path, monkeypatch):
        monkeypatch.setattr("experiment.EXPERIMENTS_DIR", tmp_path)

        for score in [0.8, 0.9, 0.7]:
            record_result(ExperimentResult(
                experiment_name="ab-test",
                variant_label="shubh-control",
                room_name=f"room-{score}",
                store_name="Store",
                quality_score=score,
            ))
        for score in [0.6, 0.5]:
            record_result(ExperimentResult(
                experiment_name="ab-test",
                variant_label="ritu-test",
                room_name=f"room-{score}",
                store_name="Store",
                quality_score=score,
            ))

        summary = summarize_experiment("ab-test")
        assert summary["total_calls"] == 5
        assert summary["variants"]["shubh-control"]["call_count"] == 3
        assert summary["variants"]["shubh-control"]["avg_score"] == 0.8
        assert summary["variants"]["ritu-test"]["call_count"] == 2
        assert summary["variants"]["ritu-test"]["avg_score"] == 0.55
