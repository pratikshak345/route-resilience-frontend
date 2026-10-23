"""
graph_analysis/gatekeeper_detection.py - Gatekeeper Node Detection

Identifies **gatekeeper** nodes - intersections whose removal would
disproportionately fragment or slow the road network.  Detection is
based on betweenness centrality: nodes in the top ``GATEKEEPER_PERCENTILE``
are flagged.

Also identifies **critical edges** (top percentile by edge betweenness).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import networkx as nx

# ── project imports ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from graph_analysis.centrality import compute_all_centralities, generate_demo_graph  # noqa: E402


# ================================================================
# Detection
# ================================================================

def detect_gatekeepers(
    graph: nx.Graph,
    percentile: float | None = None,
) -> tuple[list, list[tuple]]:
    """Flag gatekeeper nodes and critical edges.

    Parameters
    ----------
    graph : nx.Graph
        Road graph **with centrality attributes already computed**
        (call :func:`compute_all_centralities` first).
    percentile : float, optional
        Top-N percentile cutoff.
        Default ``config.GATEKEEPER_PERCENTILE`` (95).

    Returns
    -------
    gatekeeper_ids : list
        Node IDs above the betweenness percentile threshold.
    critical_edges : list[tuple]
        Edge tuples ``(u, v)`` above the edge-betweenness percentile
        threshold.
    """
    if percentile is None:
        percentile = config.GATEKEEPER_PERCENTILE

    # ── Node gatekeepers ──
    bc_scores = {n: graph.nodes[n].get("betweenness", 0.0) for n in graph.nodes()}
    values = list(bc_scores.values())

    if not values or max(values) == 0:
        print("[gatekeeper] No betweenness scores found - run compute_all_centralities first.")
        return [], []

    threshold_node = float(np.percentile(values, percentile))
    gatekeeper_ids = sorted(
        [n for n, sc in bc_scores.items() if sc >= threshold_node],
        key=lambda n: bc_scores[n],
        reverse=True,
    )

    # ── Critical edges ──
    ebc_scores = {
        (u, v): d.get("edge_betweenness", 0.0)
        for u, v, d in graph.edges(data=True)
    }
    edge_values = list(ebc_scores.values())
    threshold_edge = float(np.percentile(edge_values, percentile)) if edge_values else 0.0
    critical_edges = sorted(
        [e for e, sc in ebc_scores.items() if sc >= threshold_edge],
        key=lambda e: ebc_scores[e],
        reverse=True,
    )

    print(f"[gatekeeper] Detected {len(gatekeeper_ids)} gatekeeper node(s) "
          f"(≥ {percentile}th pctl, threshold={threshold_node:.6f})")
    print(f"[gatekeeper] Detected {len(critical_edges)} critical edge(s)")

    return gatekeeper_ids, critical_edges


# ================================================================
# Heatmap data
# ================================================================

def generate_criticality_heatmap_data(
    graph: nx.Graph,
    gatekeepers: list,
) -> list[list[float]]:
    """Produce ``[lat, lon, score]`` rows for a Folium HeatMap layer.

    If no geo transform is available, ``(y, x, score)`` is returned
    instead (suitable for pixel-space overlays).

    Parameters
    ----------
    graph : nx.Graph
        Graph with ``betweenness`` and ``pos`` / ``geo_pos`` attributes.
    gatekeepers : list
        Gatekeeper node IDs (used to weight the heatmap).

    Returns
    -------
    list[list[float]]
        Each entry is ``[lat_or_y, lon_or_x, normalised_score]``.
    """
    data: list[list[float]] = []
    max_bc = max(
        (graph.nodes[n].get("betweenness", 0.0) for n in graph.nodes()),
        default=1.0,
    ) or 1.0

    for n in graph.nodes():
        score = graph.nodes[n].get("betweenness", 0.0) / max_bc
        geo = graph.nodes[n].get("geo_pos")
        if geo is not None:
            data.append([float(geo[0]), float(geo[1]), score])
        else:
            pos = graph.nodes[n].get("pos", (0, 0))
            data.append([float(pos[0]), float(pos[1]), score])

    return data


# ================================================================
# Report
# ================================================================

def save_gatekeeper_report(
    graph: nx.Graph,
    gatekeeper_ids: list,
    critical_edges: list[tuple],
    output_path: str | Path,
) -> Path:
    """Write a Markdown report listing gatekeeper nodes and critical edges.

    Parameters
    ----------
    graph : nx.Graph
        Graph with centrality attributes.
    gatekeeper_ids : list
        Gatekeeper node IDs.
    critical_edges : list[tuple]
        Critical edge tuples.
    output_path : str or Path
        Destination ``.md`` file.

    Returns
    -------
    Path
        Resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Gatekeeper Detection Report",
        "",
        f"**Total nodes analysed:** {graph.number_of_nodes()}  ",
        f"**Total edges analysed:** {graph.number_of_edges()}  ",
        f"**Percentile threshold:** {config.GATEKEEPER_PERCENTILE}  ",
        "",
        "---",
        "",
        "## Gatekeeper Nodes",
        "",
        "| Rank | Node ID | Betweenness | Degree Cent. | Closeness | Position (y, x) |",
        "|------|---------|-------------|-------------|-----------|-----------------|",
    ]

    for rank, n in enumerate(gatekeeper_ids, start=1):
        d = graph.nodes[n]
        pos = d.get("pos", ("?", "?"))
        lines.append(
            f"| {rank} | {n} | {d.get('betweenness', 0):.6f} | "
            f"{d.get('degree_cent', 0):.6f} | "
            f"{d.get('closeness', 0):.6f} | ({pos[0]}, {pos[1]}) |"
        )

    lines += [
        "",
        "---",
        "",
        "## Critical Edges",
        "",
        "| Rank | Source | Target | Edge Betweenness | Weight |",
        "|------|--------|--------|-----------------|--------|",
    ]

    for rank, (u, v) in enumerate(critical_edges, start=1):
        ed = graph.edges[u, v]
        lines.append(
            f"| {rank} | {u} | {v} | "
            f"{ed.get('edge_betweenness', 0):.6f} | "
            f"{ed.get('weight', 0):.2f} |"
        )

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[gatekeeper] Saved report -> {output_path}")
    return output_path


# ================================================================
# CLI
# ================================================================

def main() -> None:
    """Command-line interface for gatekeeper detection."""
    parser = argparse.ArgumentParser(
        description="Detect gatekeeper nodes in a road graph."
    )
    parser.add_argument(
        "--graph", type=str, default=None,
        help="Path to GraphML. If omitted, a demo graph is used.",
    )
    parser.add_argument(
        "--percentile", type=float, default=None,
        help=f"Percentile cutoff (default: {config.GATEKEEPER_PERCENTILE}).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use a synthetic demo graph.",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else config.GRAPHS_DIR / "gatekeeper_report.md"

    if args.demo or args.graph is None:
        graph = generate_demo_graph()
    else:
        graph = nx.read_graphml(args.graph)
        for n in graph.nodes():
            y = int(float(graph.nodes[n].get("pos_y", 0)))
            x = int(float(graph.nodes[n].get("pos_x", 0)))
            graph.nodes[n]["pos"] = (y, x)

    graph = compute_all_centralities(graph)
    gatekeepers, critical_edges = detect_gatekeepers(graph, percentile=args.percentile)
    save_gatekeeper_report(graph, gatekeepers, critical_edges, output_path)

    # Heatmap data preview
    heatmap = generate_criticality_heatmap_data(graph, gatekeepers)
    print(f"[gatekeeper] Heatmap data: {len(heatmap)} points generated")


if __name__ == "__main__":
    main()
