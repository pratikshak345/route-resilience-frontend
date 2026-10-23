"""
graph_healing/vector_healing.py - Dot-Product Gap Healing

Reconnects broken road segments by analysing the *direction vectors* at
dangling endpoints.  Two endpoints are healed (connected) when:

1. They are within ``HEAL_DISTANCE_THRESHOLD`` pixels of each other.
2. Their terminal direction vectors are approximately **anti-parallel**
   (dot product < ``HEAL_DOT_PRODUCT_THRESHOLD``, i.e. the roads are
   "facing" each other across the gap).
"""

import argparse
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import networkx as nx
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt  # noqa: E402

# ── project imports ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from graph_healing.mask_to_graph import find_dangling_endpoints, generate_demo_graph  # noqa: E402


# ================================================================
# Direction vector computation
# ================================================================

def compute_direction_vector(
    graph: nx.Graph,
    node: int,
    n_pixels: int = 5,
) -> np.ndarray | None:
    """Compute the outgoing direction vector at a dangling endpoint.

    Takes the last *n_pixels* of the pixel path (``pts``) on the single
    edge connected to *node*, then returns the unit vector from the
    second-to-last point to the last (terminal) point.

    Parameters
    ----------
    graph : nx.Graph
        Road graph with edge attribute ``pts`` (ndarray of (y, x)).
    node : int
        Node ID - must be degree-1.
    n_pixels : int
        Number of terminal pixels to consider for the direction.

    Returns
    -------
    np.ndarray or None
        Unit direction vector (2,), or ``None`` if the edge has fewer
        than 2 points.
    """
    edges = list(graph.edges(node, data=True))
    if not edges:
        return None

    u, v, data = edges[0]
    pts = data.get("pts")
    if isinstance(pts, str):
        import ast
        try:
            pts = ast.literal_eval(pts)
        except Exception:
            pts = None
    if pts is None or len(pts) < 2:
        # Fall back to node positions
        pos_u = np.array(graph.nodes[u]["pos"], dtype=float)
        pos_v = np.array(graph.nodes[v]["pos"], dtype=float)
        if u == node:
            vec = pos_u - pos_v
        else:
            vec = pos_v - pos_u
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None

    pts = np.array(pts, dtype=float)

    # Determine which end of the path corresponds to `node`
    node_pos = np.array(graph.nodes[node]["pos"], dtype=float)
    dist_start = np.linalg.norm(pts[0] - node_pos)
    dist_end = np.linalg.norm(pts[-1] - node_pos)

    if dist_start <= dist_end:
        # node is at the beginning of the path - reverse so node end is last
        pts = pts[::-1]

    # Take the last n_pixels points
    tail = pts[-min(n_pixels, len(pts)):]
    if len(tail) < 2:
        return None

    vec = tail[-1] - tail[-2]
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    return vec / norm


# ================================================================
# Gap healing
# ================================================================

def heal_graph(
    graph: nx.Graph,
    distance_threshold: float | None = None,
    dot_product_threshold: float | None = None,
    n_pixels: int | None = None,
) -> tuple[nx.Graph, list[dict]]:
    """Heal gaps in the road graph using dot-product anti-parallelism.

    For every pair of dangling endpoints within *distance_threshold*:

    * Compute both direction vectors.
    * If their dot product is below *dot_product_threshold*
      (i.e. the road segments point toward each other), insert a
      healing edge.

    Parameters
    ----------
    graph : nx.Graph
        Input road graph (not modified in-place - a copy is returned).
    distance_threshold : float, optional
        Max pixel distance between endpoint pair.
        Default ``config.HEAL_DISTANCE_THRESHOLD``.
    dot_product_threshold : float, optional
        Threshold for the dot product test (should be negative).
        Default ``config.HEAL_DOT_PRODUCT_THRESHOLD``.
    n_pixels : int, optional
        Pixels used for direction estimation.
        Default ``config.DIRECTION_VECTOR_PIXELS``.

    Returns
    -------
    healed_graph : nx.Graph
        Copy of *graph* with healing edges added.
    report : list[dict]
        One dict per healed gap: ``source``, ``target``, ``distance``,
        ``dot_product``.
    """
    if distance_threshold is None:
        distance_threshold = config.HEAL_DISTANCE_THRESHOLD
    if dot_product_threshold is None:
        dot_product_threshold = config.HEAL_DOT_PRODUCT_THRESHOLD
    if n_pixels is None:
        n_pixels = config.DIRECTION_VECTOR_PIXELS

    healed = graph.copy()
    endpoints = find_dangling_endpoints(healed)
    report: list[dict] = []
    candidate_pairs = 0

    for a, b in combinations(endpoints, 2):
        pos_a = np.array(healed.nodes[a]["pos"], dtype=float)
        pos_b = np.array(healed.nodes[b]["pos"], dtype=float)
        dist = float(np.linalg.norm(pos_b - pos_a))

        if dist > distance_threshold:
            continue

        candidate_pairs += 1

        dir_a = compute_direction_vector(healed, a, n_pixels)
        dir_b = compute_direction_vector(healed, b, n_pixels)

        if dir_a is None or dir_b is None:
            continue

        dot = float(np.dot(dir_a, dir_b))

        if dot < dot_product_threshold:
            # Anti-parallel - heal the gap
            pts = np.array([pos_a, pos_b])
            healed.add_edge(a, b, pts=pts, weight=dist, healed=True)
            report.append({
                "source": a,
                "target": b,
                "distance": round(dist, 2),
                "dot_product": round(dot, 4),
            })

    print(f"[vector_healing] {len(report)} gaps healed out of "
          f"{candidate_pairs} candidate pairs")
    return healed, report


# ================================================================
# Visualization
# ================================================================

def visualize_healing(
    skeleton: np.ndarray | None,
    original_graph: nx.Graph,
    healed_graph: nx.Graph,
    output_path: str | Path,
) -> Path:
    """Create a before/after figure showing healed connections.

    Parameters
    ----------
    skeleton : np.ndarray or None
        Optional skeleton image to use as background.
    original_graph : nx.Graph
        Graph *before* healing.
    healed_graph : nx.Graph
        Graph *after* healing.
    output_path : str or Path
        Destination PNG.

    Returns
    -------
    Path
        Resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, G, title in [
        (axes[0], original_graph, "Before Healing"),
        (axes[1], healed_graph, "After Healing"),
    ]:
        if skeleton is not None:
            ax.imshow(skeleton, cmap="gray", alpha=0.3)

        # Draw normal edges
        for u, v, d in G.edges(data=True):
            pts = d.get("pts")
            if isinstance(pts, str):
                import ast
                try:
                    pts = ast.literal_eval(pts)
                except Exception:
                    pts = None
            is_healed = d.get("healed", False)
            if pts is not None and hasattr(pts, "__len__") and len(pts) >= 2:
                pts_arr = np.array(pts, dtype=float)
                color = "lime" if is_healed else "deepskyblue"
                lw = 2.5 if is_healed else 0.8
                ax.plot(pts_arr[:, 1], pts_arr[:, 0], color=color, linewidth=lw)
            else:
                pu = G.nodes[u]["pos"]
                pv = G.nodes[v]["pos"]
                color = "lime" if is_healed else "deepskyblue"
                lw = 2.5 if is_healed else 0.8
                ax.plot([pu[1], pv[1]], [pu[0], pv[0]], color=color, linewidth=lw)

        # Highlight dangling endpoints
        eps = find_dangling_endpoints(G)
        for n in eps:
            y, x = G.nodes[n]["pos"]
            ax.plot(x, y, "ro", markersize=4)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_aspect("equal")
        ax.invert_yaxis()

    fig.suptitle("Road Graph Gap Healing", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[vector_healing] Saved visualization -> {output_path}")
    return output_path


# ================================================================
# CLI
# ================================================================

def main() -> None:
    """Command-line interface for vector-based gap healing."""
    parser = argparse.ArgumentParser(
        description="Heal gaps in a road graph using dot-product direction analysis."
    )
    parser.add_argument(
        "--graph", type=str, default=None,
        help="Path to input GraphML. If omitted, a demo graph is generated.",
    )
    parser.add_argument(
        "--output-graph", type=str, default=None,
        help="Output healed GraphML path.",
    )
    parser.add_argument(
        "--output-viz", type=str, default=None,
        help="Output visualization PNG path.",
    )
    parser.add_argument(
        "--distance", type=float, default=None,
        help=f"Max healing distance (default: {config.HEAL_DISTANCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--dot-threshold", type=float, default=None,
        help=f"Dot-product threshold (default: {config.HEAL_DOT_PRODUCT_THRESHOLD}).",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use a synthetic demo graph.",
    )
    args = parser.parse_args()

    # Load or generate graph
    if args.demo or args.graph is None:
        original = generate_demo_graph()
    else:
        original = nx.read_graphml(args.graph)
        # Restore pos tuples
        for n in original.nodes():
            y = int(float(original.nodes[n].get("pos_y", 0)))
            x = int(float(original.nodes[n].get("pos_x", 0)))
            original.nodes[n]["pos"] = (y, x)
        import ast
        for u, v, d in original.edges(data=True):
            if "pts" in d and isinstance(d["pts"], str):
                try:
                    d["pts"] = np.array(ast.literal_eval(d["pts"]), dtype=float)
                except Exception:
                    pass

    healed, report = heal_graph(
        original,
        distance_threshold=args.distance,
        dot_product_threshold=args.dot_threshold,
    )

    # Save healed graph
    out_graph = Path(args.output_graph) if args.output_graph else config.GRAPHS_DIR / "healed_graph.graphml"
    from graph_healing.mask_to_graph import save_graph
    save_graph(healed, out_graph)

    # Save visualization
    out_viz = Path(args.output_viz) if args.output_viz else config.GRAPHS_DIR / "healing_visualization.png"
    visualize_healing(None, original, healed, out_viz)

    # Print report
    if report:
        print("\n[vector_healing] Healing report:")
        for entry in report:
            print(f"  {entry['source']} -> {entry['target']}  "
                  f"dist={entry['distance']}  dot={entry['dot_product']}")
    else:
        print("[vector_healing] No gaps were healed.")


if __name__ == "__main__":
    main()
