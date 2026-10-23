"""
stress_test/simulate_failure.py - Network Failure Simulation

Simulates the removal of nodes and/or edges from the road graph and
measures the impact on connectivity, path lengths, and reachability
from a reference hub (e.g. a hospital).
"""

import argparse
import copy
import sys
from pathlib import Path
from itertools import islice

import numpy as np
import networkx as nx

# ── project imports ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


# ================================================================
# Graph metric helpers
# ================================================================

def compute_graph_metrics(
    graph: nx.Graph,
    reference_hub: int | None = None,
    num_sample_paths: int | None = None,
) -> dict:
    """Compute summary resilience metrics for *graph*.

    Parameters
    ----------
    graph : nx.Graph
        Road network graph (may be disconnected).
    reference_hub : int or None
        Node ID of the reference hub (e.g. hospital). If provided,
        computes the count of nodes unreachable from this hub.
    num_sample_paths : int or None
        Number of random node pairs to sample for average shortest
        path length estimation.
        Default ``config.NUM_SAMPLE_PATHS``.

    Returns
    -------
    dict
        Keys: ``num_nodes``, ``num_edges``, ``num_components``,
        ``connectivity_ratio``, ``avg_path_length``,
        ``unreachable_from_hub``.
    """
    if num_sample_paths is None:
        num_sample_paths = config.NUM_SAMPLE_PATHS

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    components = list(nx.connected_components(graph))
    num_components = len(components)

    largest_cc_size = max(len(c) for c in components) if components else 0
    connectivity_ratio = largest_cc_size / n_nodes if n_nodes > 0 else 0.0

    # Avg shortest path length (sampled)
    avg_path_length = _sampled_avg_path_length(graph, num_sample_paths)

    # Hub reachability
    unreachable = 0
    if reference_hub is not None and reference_hub in graph:
        reachable = nx.node_connected_component(graph, reference_hub)
        unreachable = n_nodes - len(reachable)

    return {
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "num_components": num_components,
        "connectivity_ratio": round(connectivity_ratio, 4),
        "avg_path_length": round(avg_path_length, 4) if avg_path_length is not None else None,
        "unreachable_from_hub": unreachable,
    }


def _sampled_avg_path_length(
    graph: nx.Graph,
    n_samples: int,
    rng_seed: int = 42,
) -> float | None:
    """Estimate average shortest path length by sampling random node pairs.

    Parameters
    ----------
    graph : nx.Graph
        Road graph.
    n_samples : int
        Number of random pairs to test.
    rng_seed : int
        Random seed.

    Returns
    -------
    float or None
        Average shortest path length of sampled pairs, or ``None`` if
        no reachable pair was found.
    """
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        return None

    rng = np.random.default_rng(rng_seed)
    total_length = 0.0
    count = 0

    for _ in range(n_samples):
        u, v = rng.choice(nodes, size=2, replace=False)
        try:
            length = nx.shortest_path_length(graph, u, v, weight="weight")
            total_length += length
            count += 1
        except nx.NetworkXNoPath:
            continue

    return total_length / count if count > 0 else None


# ================================================================
# Failure simulation
# ================================================================

def simulate_node_failure(
    graph: nx.Graph,
    node_id: int,
    reference_hub: int | None = None,
) -> dict:
    """Simulate the failure (removal) of a single node.

    Parameters
    ----------
    graph : nx.Graph
        Original road graph.
    node_id : int
        Node to remove.
    reference_hub : int or None
        Reference hub node for reachability measurement.

    Returns
    -------
    dict
        ``before`` and ``after`` metric dicts plus ``removed_node``.
    """
    before = compute_graph_metrics(graph, reference_hub)

    damaged = copy.deepcopy(graph)
    if node_id in damaged:
        damaged.remove_node(node_id)
    else:
        print(f"[simulate_failure] Warning: node {node_id} not in graph")

    after = compute_graph_metrics(damaged, reference_hub)

    return {
        "removed_node": node_id,
        "before": before,
        "after": after,
    }


def simulate_edge_failure(
    graph: nx.Graph,
    edge: tuple,
    reference_hub: int | None = None,
) -> dict:
    """Simulate the failure (removal) of a single edge.

    Parameters
    ----------
    graph : nx.Graph
        Original road graph.
    edge : tuple
        ``(u, v)`` edge to remove.
    reference_hub : int or None
        Reference hub node.

    Returns
    -------
    dict
        ``before`` and ``after`` metric dicts plus ``removed_edge``.
    """
    before = compute_graph_metrics(graph, reference_hub)

    damaged = copy.deepcopy(graph)
    u, v = edge
    if damaged.has_edge(u, v):
        damaged.remove_edge(u, v)
    else:
        print(f"[simulate_failure] Warning: edge ({u}, {v}) not in graph")

    after = compute_graph_metrics(damaged, reference_hub)

    return {
        "removed_edge": (u, v),
        "before": before,
        "after": after,
    }


def simulate_batch_failure(
    graph: nx.Graph,
    nodes_to_remove: list | None = None,
    edges_to_remove: list[tuple] | None = None,
    reference_hub: int | None = None,
) -> dict:
    """Simulate simultaneous removal of multiple nodes and edges.

    Parameters
    ----------
    graph : nx.Graph
        Original road graph.
    nodes_to_remove : list or None
        List of node IDs to remove.
    edges_to_remove : list[tuple] or None
        List of ``(u, v)`` edges to remove.
    reference_hub : int or None
        Reference hub node.

    Returns
    -------
    dict
        ``before`` / ``after`` metrics plus removal lists.
    """
    nodes_to_remove = nodes_to_remove or []
    edges_to_remove = edges_to_remove or []

    before = compute_graph_metrics(graph, reference_hub)

    damaged = copy.deepcopy(graph)
    for n in nodes_to_remove:
        if n in damaged:
            damaged.remove_node(n)
    for u, v in edges_to_remove:
        if damaged.has_edge(u, v):
            damaged.remove_edge(u, v)

    after = compute_graph_metrics(damaged, reference_hub)

    return {
        "removed_nodes": nodes_to_remove,
        "removed_edges": edges_to_remove,
        "before": before,
        "after": after,
    }


# ================================================================
# Comparison helper
# ================================================================

def compare_scenarios(
    original_metrics: dict,
    failure_metrics: dict,
) -> dict:
    """Format a clean before/after comparison.

    Parameters
    ----------
    original_metrics : dict
        Metrics from the intact graph.
    failure_metrics : dict
        Metrics from the damaged graph.

    Returns
    -------
    dict
        Delta values for each metric.
    """
    comparison = {}
    for key in original_metrics:
        orig = original_metrics[key]
        fail = failure_metrics[key]
        if isinstance(orig, (int, float)) and isinstance(fail, (int, float)):
            comparison[key] = {
                "before": orig,
                "after": fail,
                "delta": round(fail - orig, 6),
            }
        else:
            comparison[key] = {"before": orig, "after": fail}
    return comparison


# ================================================================
# Demo graph
# ================================================================

def generate_demo_graph(n_nodes: int = 30, rng_seed: int = 42) -> nx.Graph:
    """Create a synthetic graph for failure simulation testing.

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
    """Command-line interface for failure simulation."""
    parser = argparse.ArgumentParser(
        description="Simulate node / edge failures on a road graph."
    )
    parser.add_argument(
        "--graph", type=str, default=None,
        help="Path to GraphML. If omitted, a demo graph is used.",
    )
    parser.add_argument(
        "--remove-node", type=int, default=None,
        help="Node ID to remove.",
    )
    parser.add_argument(
        "--remove-edge", type=str, default=None,
        help="Edge to remove as 'u,v' (e.g. '3,7').",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run a demo simulation removing a random high-degree node.",
    )
    args = parser.parse_args()

    # Load or generate
    if args.demo or args.graph is None:
        graph = generate_demo_graph()
    else:
        graph = nx.read_graphml(args.graph)
        for n in graph.nodes():
            y = int(float(graph.nodes[n].get("pos_y", 0)))
            x = int(float(graph.nodes[n].get("pos_x", 0)))
            graph.nodes[n]["pos"] = (y, x)

    print(f"[simulate_failure] Graph: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges\n")

    if args.remove_node is not None:
        result = simulate_node_failure(graph, args.remove_node)
    elif args.remove_edge is not None:
        u, v = [int(x.strip()) for x in args.remove_edge.split(",")]
        result = simulate_edge_failure(graph, (u, v))
    else:
        # Demo: remove highest-degree node
        target = max(graph.nodes(), key=lambda n: graph.degree(n))
        print(f"[simulate_failure] Demo: removing highest-degree node {target} "
              f"(degree {graph.degree(target)})")
        result = simulate_node_failure(graph, target)

    comparison = compare_scenarios(result["before"], result["after"])

    print("\n[simulate_failure] Impact comparison:")
    for metric, vals in comparison.items():
        if isinstance(vals, dict) and "delta" in vals:
            print(f"  {metric:25s}  before={vals['before']}  after={vals['after']}  "
                  f"delta={vals['delta']}")
        else:
            print(f"  {metric:25s}  {vals}")


if __name__ == "__main__":
    main()
