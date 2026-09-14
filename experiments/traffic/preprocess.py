from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import networkx as nx


@dataclass(slots=True)
class PreprocessResult:
    graph_path: Path
    stats_path: Path
    nodes: int
    edges: int


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _travel_time_seconds(length_m: float, speed_kmh: float) -> float:
    if speed_kmh <= 0.0:
        speed_kmh = 30.0
    mps = speed_kmh * (1000.0 / 3600.0)
    return length_m / mps if mps > 0.0 else length_m / 8.33


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def _in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float] | None) -> bool:
    if bbox is None:
        return True
    min_lat, min_lon, max_lat, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def _parse_bbox(bbox: str | None) -> tuple[float, float, float, float] | None:
    if not bbox:
        return None
    vals = [float(v.strip()) for v in bbox.split(",")]
    if len(vals) != 4:
        raise ValueError("bbox must be min_lat,min_lon,max_lat,max_lon")
    return vals[0], vals[1], vals[2], vals[3]


def bbox_latlon_to_osmium(bbox: tuple[float, float, float, float]) -> str:
    """Convert min_lat,min_lon,max_lat,max_lon to osmium extract -b (min_lon,min_lat,max_lon,max_lat)."""
    min_lat, min_lon, max_lat, max_lon = bbox
    return f"{min_lon},{min_lat},{max_lon},{max_lat}"


def find_osmium_binary() -> str | None:
    env = os.environ.get("OSMIUM", "").strip()
    if env and Path(env).is_file():
        return env
    for name in ("osmium", "osmium.exe"):
        path = shutil.which(name)
        if path:
            return path
    return None


def extract_pbf_to_osm(
    pbf_path: str | Path,
    bbox: tuple[float, float, float, float],
    out_osm: str | Path,
    *,
    osmium_bin: str | None = None,
) -> None:
    """Run osmium extract -b on a PBF to produce a small OSM XML file.

    Requires `osmium` from osmium-tool on PATH, or set OSMIUM to the executable path.
    """
    binary = osmium_bin or find_osmium_binary()
    if not binary:
        raise FileNotFoundError(
            "osmium-tool is required to clip .osm.pbf files. Install osmium-tool and ensure "
            "`osmium` is on PATH, or set the OSMIUM environment variable to the executable. "
            "See https://osmcode.org/osmium-tool/"
        )
    out = Path(out_osm)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    bbox_arg = bbox_latlon_to_osmium(bbox)
    cmd = [binary, "extract", "-b", bbox_arg, "-o", str(out), str(pbf_path)]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(f"failed to run osmium extract: {exc}") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"osmium extract failed (exit {proc.returncode}): {err}")


def load_graph(path: str | Path) -> nx.Graph:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".graphml", ".xml"} and p.name.lower().endswith(".graphml"):
        return nx.read_graphml(p)
    if suffix in {".txt", ".edgelist"}:
        return nx.read_weighted_edgelist(p, nodetype=int)
    if p.name.lower().endswith(".osm.xml") or (suffix == ".xml" and "osm" in p.name.lower()):
        return load_osm_xml_graph(p)
    if suffix == ".osm" and not p.name.lower().endswith(".osm.xml"):
        return load_osm_xml_graph(p)
    if p.name.lower().endswith(".osm.pbf"):
        raise ValueError(
            ".osm.pbf cannot be loaded directly: pass --bbox to traffic-preprocess so the "
            "pipeline can run `osmium extract` first, or convert to .osm.xml yourself."
        )
    return nx.read_graphml(p)


def load_osm_xml_graph(path: str | Path) -> nx.Graph:
    tree = ET.parse(path)
    root = tree.getroot()
    node_coords: dict[str, tuple[float, float]] = {}
    graph = nx.Graph()

    for node in root.findall("node"):
        node_id = node.attrib["id"]
        lat = _coerce_float(node.attrib.get("lat"))
        lon = _coerce_float(node.attrib.get("lon"))
        node_coords[node_id] = (lat, lon)
        graph.add_node(node_id, lat=lat, lon=lon)

    for way in root.findall("way"):
        tags = {tag.attrib.get("k"): tag.attrib.get("v") for tag in way.findall("tag")}
        if "highway" not in tags:
            continue
        way_nodes = [nd.attrib.get("ref") for nd in way.findall("nd")]
        if len(way_nodes) < 2:
            continue
        maxspeed = _coerce_float(tags.get("maxspeed"), 30.0)
        road_class = str(tags.get("highway", "unknown"))
        for a, b in zip(way_nodes[:-1], way_nodes[1:]):
            if a not in node_coords or b not in node_coords:
                continue
            lat1, lon1 = node_coords[a]
            lat2, lon2 = node_coords[b]
            length_m = _haversine_m(lat1, lon1, lat2, lon2)
            graph.add_edge(
                a,
                b,
                length=length_m,
                speed_kmh=maxspeed,
                travel_time=_travel_time_seconds(length_m, maxspeed),
                road_class=road_class,
            )
    return graph


def _relabel_to_int(g: nx.Graph) -> nx.Graph:
    if all(isinstance(n, int) for n in g.nodes):
        return g
    return nx.convert_node_labels_to_integers(g, first_label=0, ordering="default", label_attribute="orig_id")


def _filter_bbox(g: nx.Graph, bbox: tuple[float, float, float, float] | None) -> nx.Graph:
    if bbox is None:
        return g.copy()
    keep = []
    for n, data in g.nodes(data=True):
        lat = _coerce_float(data.get("lat"), float("nan"))
        lon = _coerce_float(data.get("lon"), float("nan"))
        if math.isnan(lat) or math.isnan(lon):
            continue
        if _in_bbox(lat, lon, bbox):
            keep.append(n)
    return g.subgraph(keep).copy()


def simplify_graph(
    g: nx.Graph,
    *,
    keep_lcc: bool = True,
    max_nodes: int | None = None,
    max_edges: int | None = None,
) -> nx.Graph:
    out = g.copy()
    if keep_lcc and out.number_of_nodes() > 0:
        cc = max(nx.connected_components(out), key=len)
        out = out.subgraph(cc).copy()
    if max_nodes is not None and out.number_of_nodes() > max_nodes:
        nodes = list(out.nodes)[:max_nodes]
        out = out.subgraph(nodes).copy()
    if max_edges is not None and out.number_of_edges() > max_edges:
        trimmed = nx.Graph()
        trimmed.add_nodes_from(out.nodes(data=True))
        trimmed.add_edges_from(list(out.edges(data=True))[:max_edges])
        out = trimmed
    return out


def compute_graph_stats(g: nx.Graph) -> dict[str, float]:
    nodes = g.number_of_nodes()
    edges = g.number_of_edges()
    avg_degree = float((2.0 * edges / nodes) if nodes else 0.0)
    density = float(nx.density(g)) if nodes > 1 else 0.0
    return {
        "nodes": float(nodes),
        "edges": float(edges),
        "avg_degree": avg_degree,
        "density": density,
        "connected_components": float(nx.number_connected_components(g)) if nodes else 0.0,
    }


def preprocess_map(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    bbox: str | None = None,
    keep_lcc: bool = True,
    max_nodes: int | None = None,
    max_edges: int | None = None,
) -> PreprocessResult:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(input_path)
    bbox_tuple = _parse_bbox(bbox)
    effective_path = src
    extracted_from_pbf = False

    if src.suffix.lower() == ".pbf" or src.name.lower().endswith(".osm.pbf"):
        if bbox_tuple is None:
            raise ValueError(
                "bbox is required for .osm.pbf input (min_lat,min_lon,max_lat,max_lon). "
                "Clipping with osmium avoids loading multi-gigabyte .osm.xml files."
            )
        extracted = out_dir / "extracted.osm"
        extract_pbf_to_osm(src, bbox_tuple, extracted)
        effective_path = extracted
        extracted_from_pbf = True

    g = load_graph(effective_path)
    g = _filter_bbox(g, bbox_tuple)
    g = _relabel_to_int(g)
    g = simplify_graph(g, keep_lcc=keep_lcc, max_nodes=max_nodes, max_edges=max_edges)

    graph_path = out_dir / "tokyo_subgraph.graphml"
    stats_path = out_dir / "stats.json"
    nx.write_graphml(g, graph_path)
    stats = compute_graph_stats(g)
    stats.update(
        {
            "source": str(input_path),
            "effective_osm_source": str(effective_path),
            "extracted_from_pbf": extracted_from_pbf,
            "bbox": bbox or "",
            "keep_lcc": keep_lcc,
            "max_nodes": float(max_nodes) if max_nodes is not None else 0.0,
            "max_edges": float(max_edges) if max_edges is not None else 0.0,
        }
    )
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return PreprocessResult(
        graph_path=graph_path,
        stats_path=stats_path,
        nodes=g.number_of_nodes(),
        edges=g.number_of_edges(),
    )
