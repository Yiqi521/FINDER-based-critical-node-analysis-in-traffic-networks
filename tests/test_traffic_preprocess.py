from __future__ import annotations

import json

import networkx as nx
import pytest

from experiments.traffic.preprocess import (
    bbox_latlon_to_osmium,
    compute_graph_stats,
    load_graph,
    preprocess_map,
)


def test_compute_graph_stats_basic() -> None:
    g = nx.path_graph(4)
    stats = compute_graph_stats(g)
    assert stats["nodes"] == 4.0
    assert stats["edges"] == 3.0
    assert stats["avg_degree"] > 0.0


def test_bbox_latlon_to_osmium_order() -> None:
    # min_lat, min_lon, max_lat, max_lon -> min_lon, min_lat, max_lon, max_lat
    assert bbox_latlon_to_osmium((35.5, 139.4, 35.9, 139.95)) == "139.4,35.5,139.95,35.9"


def test_preprocess_pbf_requires_bbox(tmp_path) -> None:
    pbf = tmp_path / "region.osm.pbf"
    pbf.write_bytes(b"")
    with pytest.raises(ValueError, match="bbox is required"):
        preprocess_map(pbf, tmp_path / "out")


def test_load_graph_pbf_raises(tmp_path) -> None:
    pbf = tmp_path / "region.osm.pbf"
    pbf.write_bytes(b"")
    with pytest.raises(ValueError, match="cannot be loaded directly"):
        load_graph(pbf)


def test_preprocess_map_from_graphml(tmp_path) -> None:
    g = nx.Graph()
    g.add_node(0, lat=35.68, lon=139.76)
    g.add_node(1, lat=35.69, lon=139.77)
    g.add_edge(0, 1, travel_time=10.0)
    in_path = tmp_path / "in.graphml"
    nx.write_graphml(g, in_path)

    result = preprocess_map(in_path, tmp_path / "out")
    assert result.graph_path.exists()
    assert result.stats_path.exists()
    payload = json.loads(result.stats_path.read_text(encoding="utf-8"))
    assert payload["nodes"] >= 1.0
