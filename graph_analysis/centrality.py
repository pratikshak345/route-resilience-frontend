"""
graph_analysis/centrality.py - Network Centrality Analysis

Computes multiple centrality metrics on the road graph:
  • Betweenness centrality  (primary)
  • Degree centrality
  • Closeness centrality
  • Edge betweenness centrality

Results are stored as node / edge attributes and optionally exported
as a sorted CSV table.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

# ── project imports ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


# ================================================================
# Centrality computation
# ================================================================

def compute_all_centralities(graph: nx.Graph) -> nx.Graph:
    """Compute and attach centrality metrics to *graph*.

    The following node attributes are added:

    * ``betweenness``  - betweenness centrality (weighted)
    * ``degree_cent``  - degree centrality
    * ``closeness``    - closeness centrality (weighted)

    The following edge attribute is added:

    * ``edge_betweenness`` - edge betweenness centrality (weighted)

    Parameters
    ----------
    graph : nx.Graph
        Road network graph.  Must have ``weight`` edge attribute.

    Returns
    -------
    nx.Graph
        The same graph object with centrality attributes set.
    """
    # Work on the largest connected component to avoid infinite paths
    if not nx.is_connected(graph):
        largest_cc = max(nx.connected_components(graph), key=len)
        sub = graph.subgraph(largest_cc).copy()
        print(f"[centrality] Graph not connected - using largest component "
              f"({sub.number_of_nodes()} / {graph.number_of_nodes()} nodes)")
    else:
        sub = graph

    # 1. Betweenness centrality (PRIMARY)
    bc = nx.betweenness_centrality(sub, weight="weight")
    nx.set_node_attributes(sub, bc, "betweenness")

    # 2. Degree centrality
    dc = nx.degree_centrality(sub)
    nx.set_node_attributes(sub, dc, "degree_cent")

    # 3. Closeness centrality
    cc = nx.closeness_centrality(sub, distance="weight")
    nx.set_node_attributes(sub, cc, "closeness")

    # 4. Edge betweenness centrality
    ebc = nx.edge_betweenness_centrality(sub, weight="weight")
    nx.set_edge_attributes(sub, ebc, "edge_betweenness")

    # Propagate back to the original graph
    for n in sub.nodes():
        for attr in ("betweenness", "degree_cent", "closeness"):
            graph.nodes[n][attr] = sub.nodes[n].get(attr, 0.0)
    for u, v in sub.edges():
        graph.edges[u, v]["edge_betweenness"] = sub.edges[u, v].get("edge_betweenness", 0.0)

    # Fill zeros for disconnected nodes
    for n in graph.nodes():
        for attr in ("betweenness", "degree_cent", "closeness"):
            graph.nodes[n].setdefault(attr, 0.0)

    print(f"[centrality] Computed betweenness, degree, closeness, "
          f"and edge betweenness for {graph.number_of_nodes()} nodes")
    return graph


# ================================================================
# Table export
# ================================================================

def get_centrality_table(graph: nx.Graph) -> pd.DataFrame:
    """Build a tidy DataFrame of per-node centrality scores.

    Columns: ``node_id``, ``pos_x``, ``pos_y``, ``betweenness``,
    ``degree``, ``closeness``.  Sorted by *betweenness* descending.

    Parameters
    ----------
    graph : nx.Graph
        Graph with centrality attributes (call
        :func:`compute_all_centralities` first).

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for n in graph.nodes():
        data = graph.nodes[n]
        pos = data.get("pos", (0, 0))
        rows.append({
            "node_id": n,
            "pos_x": int(pos[1]) if isinstance(pos, (tuple, list)) else 0,
            "pos_y": int(pos[0]) if isinstance(pos, (tuple, list)) else 0,
            "betweenness": round(data.get("betweenness", 0.0), 6),
            "degree": round(data.get("degree_cent", 0.0), 6),
            "closeness": round(data.get("closeness", 0.0), 6),
        })

    df = pd.DataFrame(rows)
    df.sort_values("betweenness", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def save_centrality_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a centrality DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Centrality table (from :func:`get_centrality_table`).
    output_path : str or Path
        Destination CSV path.

    Returns
    -------
    Path
        Resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[centrality] Saved centrality table -> {output_path}")
    return output_path


# ================================================================
# Demo graph generator
# ================================================================

def generate_demo_graph(n_nodes: int = 30, rng_seed: int = 42) -> nx.Graph:
    """Create a small synthetic graph for centrality testing.

    Parameters
    ----------
    n_nodes : int
        Approximate node count.
    rng_seed : int
        Random seed.

    Returns
    -------
    nx.Graph
    """
    try:
        from graph_healing.mask_to_graph import generate_demo_graph as _gen
        return _gen(n_nodes=n_nodes, rng_seed=rng_seed)
    except ImportError:
        pass

    # Standalone fallback
    rng = np.random.default_rng(rng_seed)
    G = nx.grid_2d_graph(6, 6)
    mapping = {old: idx for idx, old in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    for n in G.nodes():
        G.nodes[n]["pos"] = (rng.integers(0, 300), rng.integers(0, 300))
    for u, v in G.edges():
        pu = np.array(G.nodes[u]["pos"], dtype=float)
        pv = np.array(G.nodes[v]["pos"], dtype=float)
        G.edges[u, v]["weight"] = float(np.linalg.norm(pv - pu))
        G.edges[u, v]["pts"] = np.array([pu, pv])
    return G


# ================================================================
# CLI
# ================================================================

def main() -> None:
    """Command-line interface for centrality analysis."""
    parser = argparse.ArgumentParser(
        description="Compute centrality metrics on a road graph."
    )
    parser.add_argument(
        "--graph", type=str, default=None,
        help="Path to a GraphML file. If omitted, a demo graph is used.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path. Default: <config.GRAPHS_DIR>/centrality.csv",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use a synthetic demo graph.",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else config.GRAPHS_DIR / "centrality.csv"

    if args.demo or args.graph is None:
        graph = generate_demo_graph()
    else:
        graph = nx.read_graphml(args.graph)
        for n in graph.nodes():
            y = int(float(graph.nodes[n].get("pos_y", 0)))
            x = int(float(graph.nodes[n].get("pos_x", 0)))
            graph.nodes[n]["pos"] = (y, x)

    graph = compute_all_centralities(graph)
    df = get_centrality_table(graph)
    save_centrality_csv(df, output_path)

    print(f"\n[centrality] Top-5 nodes by betweenness:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
