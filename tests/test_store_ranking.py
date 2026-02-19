"""Tests for pipeline/store_discovery.py — rank_stores auto-selection."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.schemas import DiscoveredStore
from pipeline.store_discovery import rank_stores


def _store(name="Test Store", phone="+911234567890", rating=4.0,
           review_count=100, source="google_maps", relevance_score=0.5):
    return DiscoveredStore(
        name=name, phone=phone, rating=rating,
        review_count=review_count, source=source,
        relevance_score=relevance_score,
    )


class TestRankStores:
    def test_returns_top_n(self):
        stores = [_store(name=f"Store{i}") for i in range(8)]
        ranked = rank_stores(stores, top_n=4)
        assert len(ranked) == 4

    def test_phone_available_ranks_higher(self):
        with_phone = _store(name="With Phone", phone="+911234567890", rating=3.0)
        no_phone = _store(name="No Phone", phone=None, rating=3.0)
        ranked = rank_stores([no_phone, with_phone], top_n=2)
        assert ranked[0].name == "With Phone"

    def test_higher_rating_ranks_higher(self):
        low = _store(name="Low Rating", rating=2.0)
        high = _store(name="High Rating", rating=4.5)
        ranked = rank_stores([low, high], top_n=2)
        assert ranked[0].name == "High Rating"

    def test_google_maps_source_bonus(self):
        maps = _store(name="Maps", source="google_maps", rating=3.0, review_count=50)
        web = _store(name="Web", source="web_search", rating=3.0, review_count=50)
        ranked = rank_stores([web, maps], top_n=2)
        assert ranked[0].name == "Maps"

    def test_more_reviews_ranks_higher(self):
        few = _store(name="Few Reviews", review_count=5, rating=4.0)
        many = _store(name="Many Reviews", review_count=5000, rating=4.0)
        ranked = rank_stores([few, many], top_n=2)
        assert ranked[0].name == "Many Reviews"

    def test_higher_relevance_ranks_higher(self):
        low_rel = _store(name="Low Relevance", relevance_score=0.2)
        high_rel = _store(name="High Relevance", relevance_score=0.9)
        ranked = rank_stores([low_rel, high_rel], top_n=2)
        assert ranked[0].name == "High Relevance"

    def test_empty_stores_returns_empty(self):
        assert rank_stores([], top_n=4) == []

    def test_fewer_stores_than_top_n(self):
        stores = [_store(name=f"S{i}") for i in range(2)]
        ranked = rank_stores(stores, top_n=4)
        assert len(ranked) == 2
