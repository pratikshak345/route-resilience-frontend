"""
graph_healing/mask_to_graph.py - Skeleton -> NetworkX Graph Conversion

Converts a 1-pixel-wide skeleton image into a NetworkX graph where:
  • Nodes carry ``pos`` = (y, x) pixel coordinates.
  • Edges carry ``pts`` (pixel path) and ``weight`` (Euclidean length).

Falls back to manual junction detection if the ``sknw`` package is not
available.
"""

import argparse
import sys
from pathlib import Path
from collections import deque

import numpy as np
import networkx as nx
from scipy.ndimage import convolve

# ── project imports ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


# ================================================================
# Skeleton -> Graph
# ================================================================

def skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
    """Convert a binary skeleton image to a NetworkX graph.

    Attempts to use ``sknw.build_sknw`` for the conversion. If ``sknw``
    is not installed, falls back to a manual junction/endpoint detection
    approach using neighbour counting and BFS path tracing.

    Parameters
    ----------
    skeleton : np.ndarray
        2-D boolean array (skeleton).

    Returns
    -------
    nx.Graph
        Graph with node attribute ``pos`` (y, x) and edge attributes
        ``pts`` (ndarray of pixel coords) and ``weight`` (Euclidean
        length of the edge path).
    """
    try:
        import sknw
        # sknw expects a uint8/uint16 skeleton
        skel_img = skeleton.astype(np.uint16)
        graph = sknw.build_sknw(skel_img)
        # Ensure consistent attribute names
        for n in graph.nodes():
            pts = graph.nodes[n].get("pts", graph.nodes[n].get("o", None))
            if pts is not None:
                if pts.ndim == 2:
                    graph.nodes[n]["pos"] = tuple(pts.mean(axis=0).astype(int))
                else:
                    graph.nodes[n]["pos"] = tuple(pts.astype(int))
            else:
                graph.nodes[n]["pos"] = (0, 0)
        for u, v, d in graph.edges(data=True):
            if "pts" in d:
                pts = d["pts"]
                d["weight"] = float(_path_length(pts))
            else:
                pu = np.array(graph.nodes[u]["pos"])
                pv = np.array(graph.nodes[v]["pos"])
                d["pts"] = np.array([pu, pv])
                d["weight"] = float(np.linalg.norm(pv - pu))
        print(f"[mask_to_graph] Built graph via sknw - "
              f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        return graph
    except ImportError:
        print("[mask_to_graph] sknw not available - using manual extraction")
        return _manual_skeleton_to_graph(skeleton)


# ----------------------------------------------------------------
# Manual fallback
# ----------------------------------------------------------------

def _neighbor_count(skeleton: np.ndarray) -> np.ndarray:
    """Count 8-connected neighbours for every skeleton pixel."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0  # don't count the pixel itself
    return convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)


def _manual_skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
    """Build a graph from a skeleton via manual junction detection.

    Algorithm
    ---------
    1. Count neighbours of each skeleton pixel with a 3×3 kernel.
    2. Endpoints  = skeleton pixels with exactly 1 neighbour.
    3. Junctions  = skeleton pixels with ≥ 3 neighbours.
    4. Trace paths between junctions / endpoints using BFS.
    """
    nbr_count = _neighbor_count(skeleton)
    skel_pixels = set(zip(*np.where(skeleton)))

    # Classify pixels
    endpoints = set()
    junctions = set()
    for (y, x) in skel_pixels:
        n = nbr_count[y, x]
        if n == 1:
            endpoints.add((y, x))
        elif n >= 3:
            junctions.add((y, x))

    key_points = endpoints | junctions
    if not key_points:
        # Degenerate: treat every skeleton pixel as a node
        key_points = skel_pixels.copy()

    # Map key-point coords -> node IDs
    kp_list = sorted(key_points)
    coord_to_id = {pt: idx for idx, pt in enumerate(kp_list)}

    G = nx.Graph()
    for idx, pt in enumerate(kp_list):
        G.add_node(idx, pos=pt)

    visited_edges: set[frozenset] = set()

    def _neighbours(y, x):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if (ny, nx_) in skel_pixels:
                    yield (ny, nx_)

    # BFS trace from each key point
    for start in kp_list:
        for nbr in _neighbours(*start):
            if nbr in key_points and nbr != start:
                # Direct adjacency
                edge_key = frozenset((start, nbr))
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    pts = np.array([start, nbr])
                    G.add_edge(
                        coord_to_id[start],
                        coord_to_id[nbr],
                        pts=pts,
                        weight=float(np.linalg.norm(np.array(nbr) - np.array(start))),
                    )
                continue

            # Trace along the edge
            path = [start, nbr]
            visited_trace = {start, nbr}
            current = nbr
            while current not in key_points or current == nbr:
                if current in key_points and current != start:
                    break
                found_next = False
                for nn in _neighbours(*current):
                    if nn not in visited_trace:
                        visited_trace.add(nn)
                        path.append(nn)
                        current = nn
                        found_next = True
                        break
                if not found_next:
                    break

            end = path[-1]
            if end in key_points and end != start:
                edge_key = frozenset((start, end))
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    pts = np.array(path)
                    G.add_edge(
                        coord_to_id[start],
                        coord_to_id[end],
                        pts=pts,
                        weight=float(_path_length(pts)),
                    )

    print(f"[mask_to_graph] Built graph (manual) - "
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _path_length(pts: np.ndarray) -> float:
    """Compute the cumulative Euclidean length of a pixel path.

    Parameters
    ----------
    pts : np.ndarray
        (N, 2) array of (y, x) coordinates.

    Returns
    -------
    float
        Total path length in pixels.
    """
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts.astype(float), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def find_dangling_endpoints(graph: nx.Graph) -> list:
    """Return node IDs of degree-1 ("dangling") nodes.

    These represent road segments that end abruptly - prime candidates
    for gap healing.

    Parameters
    ----------
    graph : nx.Graph
        Road network graph.

    Returns
    -------
    list
        Sorted list of node IDs with degree == 1.
    """
    return sorted(n for n in graph.nodes() if graph.degree(n) == 1)


def graph_to_geo(
    graph: nx.Graph,
    transform=None,
    crs: str | None = None,
) -> nx.Graph:
    """Convert pixel coordinates to geographic coordinates.

    If ``transform`` (an affine ``rasterio.transform.Affine``) and
    ``crs`` are provided, each node's ``pos`` (row, col) is mapped to
    ``(lat, lon)`` and stored in ``geo_pos``.

    Parameters
    ----------
    graph : nx.Graph
        Graph with node attribute ``pos`` = (y, x) in pixels.
    transform : rasterio.transform.Affine or None
        Affine transform from pixel space to CRS coordinates.
    crs : str or None
        Coordinate reference system string, e.g. ``"EPSG:4326"``.

    Returns
    -------
    nx.Graph
        Same graph, with ``geo_pos`` and ``crs`` attributes added to
        each node (or ``geo_pos`` set to ``pos`` if no transform).
    """
    for n in graph.nodes():
        y, x = graph.nodes[n]["pos"]
        if transform is not None:
            lon, lat = transform * (x, y)
            graph.nodes[n]["geo_pos"] = (lat, lon)
        else:
            graph.nodes[n]["geo_pos"] = (y, x)
        if crs is not None:
            graph.nodes[n]["crs"] = crs
    return graph


def save_graph(graph: nx.Graph, output_path: str | Path) -> Path:
    """Save a graph as GraphML.

    Parameters
    ----------
    graph : nx.Graph
        Graph to save.
    output_path : str or Path
        Destination file path (should end with ``.graphml``).

    Returns
    -------
    Path
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    G_export = graph.copy()
    for u, v, d in G_export.edges(data=True):
        for k in list(d.keys()):
            if isinstance(d[k], np.ndarray):
                d[k] = str(d[k].tolist())
            elif isinstance(d[k], (np.integer, np.int64, np.int32)):
                d[k] = int(d[k])
            elif isinstance(d[k], (np.floating, np.float64, np.float32)):
                d[k] = float(d[k])
    for n in G_export.nodes():
        pos = G_export.nodes[n].get("pos")
        if pos is not None:
            G_export.nodes[n]["pos_y"] = int(pos[0])
            G_export.nodes[n]["pos_x"] = int(pos[1])
        geo = G_export.nodes[n].get("geo_pos")
        if geo is not None:
            G_export.nodes[n]["geo_lat"] = float(geo[0])
            G_export.nodes[n]["geo_lon"] = float(geo[1])
        for k in list(G_export.nodes[n].keys()):
            val = G_export.nodes[n][k]
            if k in ["pos", "geo_pos", "pts", "o"]:
                G_export.nodes[n].pop(k, None)
            elif isinstance(val, np.ndarray):
                G_export.nodes[n][k] = str(val.tolist())
            elif isinstance(val, (np.integer, np.int64, np.int32)):
                G_export.nodes[n][k] = int(val)
            elif isinstance(val, (np.floating, np.float64, np.float32)):
                G_export.nodes[n][k] = float(val)

    nx.write_graphml(G_export, str(output_path))
    print(f"[mask_to_graph] Saved graph -> {output_path}")
    return output_path


# ================================================================
# Demo graph generator
# ================================================================

def generate_demo_graph(n_nodes: int = 30, rng_seed: int = 42) -> nx.Graph:
    """Create a synthetic road-like graph for testing.

    Produces a grid-like graph with random perturbations and a few
    degree-1 dangling endpoints.

    Parameters
    ----------
    n_nodes : int
        Approximate number of nodes (actual count depends on grid size).
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph
        Synthetic road graph with ``pos``, ``pts``, and ``weight``.
    """
    rng = np.random.default_rng(rng_seed)
    side = int(np.ceil(np.sqrt(n_nodes)))
    spacing = 50

    G = nx.Graph()
    nid = 0
    grid = {}
    for r in range(side):
        for c in range(side):
            y = r * spacing + int(rng.integers(-5, 6))
            x = c * spacing + int(rng.integers(-5, 6))
            G.add_node(nid, pos=(y, x))
            grid[(r, c)] = nid
            nid += 1

    for r in range(side):
        for c in range(side):
            # Right neighbour
            if c + 1 < side and rng.random() > 0.1:
                u, v = grid[(r, c)], grid[(r, c + 1)]
                pu, pv = np.array(G.nodes[u]["pos"]), np.array(G.nodes[v]["pos"])
                pts = np.array([pu, pv])
                G.add_edge(u, v, pts=pts, weight=float(np.linalg.norm(pv - pu)))
            # Down neighbour
            if r + 1 < side and rng.random() > 0.1:
                u, v = grid[(r, c)], grid[(r + 1, c)]
                pu, pv = np.array(G.nodes[u]["pos"]), np.array(G.nodes[v]["pos"])
                pts = np.array([pu, pv])
                G.add_edge(u, v, pts=pts, weight=float(np.linalg.norm(pv - pu)))

    # Add a few explicit dangling stubs
    for _ in range(3):
        base_node = rng.integers(0, nid)
        by, bx = G.nodes[base_node]["pos"]
        stub_y = by + int(rng.integers(20, 40))
        stub_x = bx + int(rng.integers(-10, 10))
        G.add_node(nid, pos=(stub_y, stub_x))
        pu = np.array([by, bx])
        pv = np.array([stub_y, stub_x])
        G.add_edge(base_node, nid, pts=np.array([pu, pv]),
                   weight=float(np.linalg.norm(pv - pu)))
        nid += 1

    print(f"[mask_to_graph] Generated demo graph - "
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ================================================================
# CLI
# ================================================================

def main() -> None:
    """Command-line interface for skeleton-to-graph conversion."""
    parser = argparse.ArgumentParser(
        description="Convert a skeleton image to a NetworkX graph, or generate a demo."
    )
    parser.add_argument(
        "--skeleton", type=str, default=None,
        help="Path to skeleton PNG. If omitted, a demo graph is generated.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output GraphML path. Default: <config.GRAPHS_DIR>/road_graph.graphml",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Generate a synthetic demo graph instead of loading a skeleton.",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else config.GRAPHS_DIR / "road_graph.graphml"

    if args.demo or args.skeleton is None:
        graph = generate_demo_graph()
    else:
        from skimage import io as skio
        skel = skio.imread(args.skeleton, as_gray=True) > 0.5
        graph = skeleton_to_graph(skel)

    endpoints = find_dangling_endpoints(graph)
    print(f"[mask_to_graph] Dangling endpoints: {len(endpoints)}")

    save_graph(graph, output_path)


if __name__ == "__main__":
    main()
