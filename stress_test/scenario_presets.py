"""
stress_test/scenario_presets.py - Disaster Scenario Presets

Defines three realistic disaster scenarios for Bengaluru:
  1. Flood at ORR–Sarjapur Junction
  2. Bridge collapse at KR Puram
  3. Landslide blocking Hebbal Flyover approach

Each scenario is a plain dict with metadata; ``run_preset_scenario``
wires it up to the failure simulator.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import networkx as nx

# ── project imports ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from stress_test.simulate_failure import (  # noqa: E402
    simulate_node_failure,
    simulate_edge_failure,
    compare_scenarios,
    generate_demo_graph,
)


# ================================================================
# Scenario definitions
# ================================================================

SCENARIO_FLOOD = {
    "name": "Flood at ORR-Sarjapur Junction",
    "description": (
        "Monsoon flooding cuts off the Outer Ring Road junction near "
        "Sarjapur, a critical IT corridor connector."
    ),
    "type": "node_removal",
    "target": config.LANDMARKS["outer_ring_road_junction"],
    "severity": "high",
    "affected_area_radius_m": 500,
}

SCENARIO_BRIDGE_COLLAPSE = {
    "name": "Bridge Collapse at KR Puram",
    "description": (
        "Structural failure of the KR Puram railway bridge cuts "
        "east-west connectivity."
    ),
    "type": "node_removal",
    "target": config.LANDMARKS["kr_puram_bridge"],
    "severity": "critical",
}

SCENARIO_LANDSLIDE = {
    "name": "Landslide blocks Hebbal Flyover approach",
    "description": (
        "Heavy rains cause landslide blocking the northern arterial "
        "approach at Hebbal."
    ),
    "type": "node_removal",
    "target": config.LANDMARKS["hebbal_flyover"],
    "severity": "high",
}


# ================================================================
# Scenario catalogue
# ================================================================

def get_all_scenarios() -> list[dict]:
    """Return a list of all predefined disaster scenarios.

    Returns
    -------
    list[dict]
        Each dict contains ``name``, ``description``, ``type``,
        ``target``, ``severity``, and optional extras.
    """
    return [SCENARIO_FLOOD, SCENARIO_BRIDGE_COLLAPSE, SCENARIO_LANDSLIDE]


def get_scenario_by_name(name: str) -> dict | None:
    """Look up a scenario by its ``name`` field (case-insensitive).

    Parameters
    ----------
    name : str
        Scenario name (or substring).

    Returns
    -------
    dict or None
        Matching scenario, or ``None`` if not found.
    """
    name_lower = name.lower()
    for sc in get_all_scenarios():
        if name_lower in sc["name"].lower():
            return sc
    return None


# ================================================================
# Nearest-node lookup
# ================================================================

def find_nearest_graph_node(
    graph: nx.Graph,
    lat: float,
    lon: float,
) -> int | None:
    """Find the graph node closest to a geographic coordinate.

    Uses Euclidean distance on the node ``geo_pos`` attribute
    (lat, lon) if available, otherwise on the raw ``pos`` (y, x).

    Parameters
    ----------
    graph : nx.Graph
        Road network graph.
    lat : float
        Latitude (or y-coordinate) of the target.
    lon : float
        Longitude (or x-coordinate) of the target.

    Returns
    -------
    int or None
        Node ID of the nearest node, or ``None`` if graph is empty.
    """
    if graph.number_of_nodes() == 0:
        return None

    target = np.array([lat, lon], dtype=float)
    best_node = None
    best_dist = np.inf

    for n in graph.nodes():
        geo = graph.nodes[n].get("geo_pos")
        if geo is not None:
            pos = np.array(geo, dtype=float)
        else:
            raw = graph.nodes[n].get("pos", (0, 0))
            pos = np.array(raw, dtype=float)

        dist = np.linalg.norm(pos - target)
        if dist < best_dist:
            best_dist = dist
            best_node = n

    return best_node


# ================================================================
# Run a preset scenario
# ================================================================

def run_preset_scenario(
    graph: nx.Graph,
    scenario: dict,
    reference_hub: int | None = None,
) -> dict:
    """Execute a disaster scenario on the graph.

    1. Finds the nearest graph node to the scenario's target coordinate.
    2. Runs the appropriate failure simulation (node or edge removal).
    3. Returns results with scenario metadata attached.

    Parameters
    ----------
    graph : nx.Graph
        Road network graph.
    scenario : dict
        Scenario dict (from :func:`get_all_scenarios`).
    reference_hub : int or None
        Reference hub node for reachability metrics.

    Returns
    -------
    dict
        ``scenario`` metadata, ``target_node``, ``before``, ``after``,
        and ``comparison``.
    """
    target_info = scenario["target"]
    target_node = find_nearest_graph_node(
        graph, target_info["lat"], target_info["lon"]
    )

    if target_node is None:
        print(f"[scenario] Could not find a graph node for scenario "
              f"'{scenario['name']}'")
        return {"scenario": scenario, "error": "no matching node"}

    print(f"[scenario] Running: {scenario['name']}")
    print(f"[scenario] Target landmark: {target_info['name']} -> node {target_node}")

    if scenario["type"] == "node_removal":
        result = simulate_node_failure(graph, target_node, reference_hub)
    elif scenario["type"] == "edge_removal":
        edges = list(graph.edges(target_node))
        if edges:
            result = simulate_edge_failure(graph, edges[0], reference_hub)
        else:
            result = {"before": {}, "after": {}, "error": "no edges on target node"}
    else:
        result = simulate_node_failure(graph, target_node, reference_hub)

    comparison = compare_scenarios(result["before"], result["after"])

    return {
        "scenario": scenario,
        "target_node": target_node,
        "before": result["before"],
        "after": result["after"],
        "comparison": comparison,
    }


# ================================================================
# CLI
# ================================================================

def main() -> None:
    """Command-line interface for running preset disaster scenarios."""
    parser = argparse.ArgumentParser(
        description="Run predefined disaster scenarios on a road graph."
    )
    parser.add_argument(
        "--graph", type=str, default=None,
        help="Path to GraphML. If omitted, a demo graph is used.",
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        help="Scenario name (substring match). If omitted, all are run.",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use a synthetic demo graph.",
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

    # Select scenarios
    if args.scenario:
        sc = get_scenario_by_name(args.scenario)
        if sc is None:
            print(f"[scenario] No scenario matching '{args.scenario}'")
            print("[scenario] Available:")
            for s in get_all_scenarios():
                print(f"  - {s['name']}")
            return
        scenarios = [sc]
    else:
        scenarios = get_all_scenarios()

    # Run each scenario
    for sc in scenarios:
        result = run_preset_scenario(graph, sc)
        print(f"\n{'='*60}")
        print(f"  Scenario: {sc['name']}")
        print(f"  Severity: {sc['severity']}")
        print(f"  Target node: {result.get('target_node', '?')}")
        print(f"{'='*60}")

        comparison = result.get("comparison", {})
        for metric, vals in comparison.items():
            if isinstance(vals, dict) and "delta" in vals:
                print(f"  {metric:25s}  before={vals['before']}  "
                      f"after={vals['after']}  delta={vals['delta']}")
            else:
                print(f"  {metric:25s}  {vals}")
        print()


if __name__ == "__main__":
    main()
