"""
Route Resilience - Urban Site Planner & Vacant Land Advisor

Identifies vacant and underdeveloped land parcels across Bengaluru,
analyzes their geospatial attributes (road connectivity, betweenness
centrality, distance to nearest gatekeeper bottleneck and hospital),
and uses a multi-criteria spatial decision engine to advise civil planners
and constructors on optimal site development with high urban resilience impact.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import networkx as nx

import config

# =====================================================================
# Real Bengaluru Vacant / Underutilized Land Parcels Catalog
# Calibrated to exact coordinates in the road network
# =====================================================================
VACANT_PARCELS: list[dict[str, Any]] = [
    {
        "id": "BLR-VAC-001",
        "name": "Bellandur-ORR Technology Buffer Zone",
        "zone": "Outer Ring Road (South-East)",
        "lat": 12.9260,
        "lon": 77.6630,
        "area_acres": 14.5,
        "zoning": "Commercial / Buffer",
        "terrain": "Gently Sloping / Near Lake Buffer",
        "description": "High-visibility unbuilt parcel located along the Outer Ring Road corridor between Bellandur and Sarjapur Junction.",
        "polygon": [
            [12.9245, 77.6615],
            [12.9275, 77.6618],
            [12.9272, 77.6648],
            [12.9242, 77.6645],
        ],
    },
    {
        "id": "BLR-VAC-002",
        "name": "Hebbal Northern Bypass Open Parcel",
        "zone": "Hebbal / Bellary Road Corridor (North)",
        "lat": 13.0360,
        "lon": 77.5920,
        "area_acres": 22.8,
        "zoning": "Mixed-Use / Logistics",
        "terrain": "Elevated Flat Terrain",
        "description": "Large strategic plot adjacent to the Kempegowda International Airport express corridor and Hebbal interchange.",
        "polygon": [
            [13.0340, 77.5900],
            [13.0380, 77.5905],
            [13.0375, 77.5942],
            [13.0338, 77.5938],
        ],
    },
    {
        "id": "BLR-VAC-003",
        "name": "Silk Board - Madiwala Peripheral Basin Plot",
        "zone": "BTM / Silk Board Junction (South)",
        "lat": 12.9180,
        "lon": 77.6210,
        "area_acres": 9.2,
        "zoning": "Civic / Infrastructure",
        "terrain": "Low-lying Natural Drainage Depression",
        "description": "Critical unbuilt parcel south of Silk Board Junction prone to monsoon waterlogging, ideal for sponge-city infrastructure.",
        "polygon": [
            [12.9165, 77.6195],
            [12.9195, 77.6198],
            [12.9190, 77.6225],
            [12.9162, 77.6222],
        ],
    },
    {
        "id": "BLR-VAC-004",
        "name": "KR Puram Eastern Railhead Infill",
        "zone": "KR Puram / Whitefield Connector (East)",
        "lat": 12.9980,
        "lon": 77.6620,
        "area_acres": 17.6,
        "zoning": "Industrial / Commercial",
        "terrain": "Flat Industrial Infill",
        "description": "Brownfield redevelopment plot between Old Madras Road and the railway corridor, proximate to heavy freight and tech commuter flows.",
        "polygon": [
            [12.9960, 77.6600],
            [13.0000, 77.6605],
            [12.9995, 77.6642],
            [12.9958, 77.6638],
        ],
    },
    {
        "id": "BLR-VAC-005",
        "name": "Yeshwanthpur Industrial-Transit Transition Zone",
        "zone": "Yeshwanthpur / Tumkur Road (North-West)",
        "lat": 13.0250,
        "lon": 77.5480,
        "area_acres": 19.4,
        "zoning": "Light Industrial / Logistics",
        "terrain": "Level Hard-Rock Base",
        "description": "High-connectivity plot near the Yeshwanthpur APMC market, metro depot, and national highway junction.",
        "polygon": [
            [13.0230, 77.5460],
            [13.0270, 77.5465],
            [13.0265, 77.5502],
            [13.0228, 77.5498],
        ],
    },
    {
        "id": "BLR-VAC-006",
        "name": "Victoria Hospital Core Regeneration Pocket",
        "zone": "Kalasipalya / Old City Core (Central)",
        "lat": 12.9630,
        "lon": 77.5730,
        "area_acres": 6.8,
        "zoning": "Civic / Institutional",
        "terrain": "Dense Urban Infill",
        "description": "High-density urban core infill site situated directly in the healthcare and central transit corridor.",
        "polygon": [
            [12.9615, 77.5715],
            [12.9645, 77.5718],
            [12.9640, 77.5745],
            [12.9612, 77.5742],
        ],
    },
]

# =====================================================================
# Main City Hubs & Strategic Arterial Corridors
# =====================================================================
MAIN_CITY_HUBS: dict[str, dict[str, Any]] = {
    "CBD_CORE": {
        "id": "CBD_CORE",
        "name": "Vidhana Soudha / MG Road",
        "corridor_name": "CBD / Vidhana Soudha Corridor",
        "lat": 12.9716,
        "lon": 77.5946,
        "color": "#a55eea",
        "weight": 4.5,
        "icon": "institution",
    },
    "MAJESTIC_HUB": {
        "id": "MAJESTIC_HUB",
        "name": "Kempegowda Transit Station",
        "corridor_name": "Majestic Transit Corridor",
        "lat": 12.9774,
        "lon": 77.5708,
        "color": "#fa8231",
        "weight": 4.5,
        "icon": "bus",
    },
    "VICTORIA_HOSPITAL": {
        "id": "VICTORIA_HOSPITAL",
        "name": "Emergency Trauma Hub",
        "corridor_name": "Victoria Hospital Emergency Corridor",
        "lat": 12.9628,
        "lon": 77.5748,
        "color": "#eb3b5a",
        "weight": 4.5,
        "icon": "plus",
    },
}


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in km."""
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def _find_nearest_graph_node(
    graph: nx.Graph,
    lat: float,
    lon: float,
) -> tuple[Any, float, float, float]:
    """Find the nearest graph node and return (node_id, node_lat, node_lon, dist_meters)."""
    min_dist_sq = float("inf")
    best_node = None
    best_lat = lat
    best_lon = lon

    for n in graph.nodes():
        nd = graph.nodes[n]
        n_lat = float(nd.get("lat") or nd.get("geo_lat") or nd.get("y", 0))
        n_lon = float(nd.get("lon") or nd.get("geo_lon") or nd.get("x", 0))
        dist_sq = (n_lat - lat) ** 2 + (n_lon - lon) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_node = n
            best_lat = n_lat
            best_lon = n_lon

    dist_m = haversine_distance_km(lat, lon, best_lat, best_lon) * 1000.0 if best_node is not None else 500.0
    return best_node, best_lat, best_lon, dist_m


def evaluate_site_potential(
    lat: float,
    lon: float,
    area_acres: float = 12.0,
    graph: nx.Graph | None = None,
    centrality: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run multi-criteria spatial analysis on a vacant land parcel.

    Calculates:
    - Distance to nearest arterial road and ingress access gate
    - Nearest gatekeeper bottleneck distance & centrality
    - Hospital emergency access distance
    - Surrounding road segments within ~1.8km radius
    - Main city connector routes (CBD, Majestic, Victoria Hospital)
    - Strategic construction recommendation & business case
    """
    # 1. Proximity to Victoria Hospital Emergency Hub
    hospital_lat = MAIN_CITY_HUBS["VICTORIA_HOSPITAL"]["lat"]
    hospital_lon = MAIN_CITY_HUBS["VICTORIA_HOSPITAL"]["lon"]
    dist_hospital_km = haversine_distance_km(lat, lon, hospital_lat, hospital_lon)

    # 2. Proximity to landmark bottlenecks
    landmark_dists = {}
    for k, v in config.LANDMARKS.items():
        if k != "hospital_reference":
            d_km = haversine_distance_km(lat, lon, v["lat"], v["lon"])
            landmark_dists[v["name"]] = round(d_km, 2)
    nearest_landmark_name = min(landmark_dists, key=landmark_dists.get) if landmark_dists else "City Center"
    nearest_landmark_dist_km = landmark_dists.get(nearest_landmark_name, 2.5)

    # 3. Graph connectivity, ingress point, and local betweenness centrality
    nearest_node_id = None
    nearest_road_dist_m = 350.0
    ingress_lat = lat
    ingress_lon = lon
    local_centrality = 0.025
    local_degree = 3

    surrounding_roads: list[list[list[float]]] = []
    main_city_routes: dict[str, dict[str, Any]] = {}

    # Gatekeeper centrality threshold (top 5%)
    gk_threshold = 0.028
    if centrality:
        cent_vals = list(centrality.values())
        if cent_vals:
            gk_threshold = float(np.percentile(cent_vals, config.GATEKEEPER_PERCENTILE))

    if graph is not None and len(graph) > 0:
        nearest_node_id, ingress_lat, ingress_lon, nearest_road_dist_m = _find_nearest_graph_node(graph, lat, lon)

        if nearest_node_id is not None:
            local_degree = graph.degree(nearest_node_id)
            if centrality:
                if nearest_node_id in centrality:
                    local_centrality = float(centrality[nearest_node_id])
                elif str(nearest_node_id) in centrality:
                    local_centrality = float(centrality[str(nearest_node_id)])
                elif isinstance(nearest_node_id, str) and nearest_node_id.isdigit() and int(nearest_node_id) in centrality:
                    local_centrality = float(centrality[int(nearest_node_id)])

        # -------------------------------------------------------------
        # Extract surrounding road network within ~1.8 km of parcel
        # -------------------------------------------------------------
        radius_km = 1.8
        deg_radius = radius_km / 111.0
        deg_radius_sq = deg_radius ** 2

        local_nodes: set[Any] = set()
        for n in graph.nodes():
            nd = graph.nodes[n]
            n_lat = float(nd.get("lat") or nd.get("geo_lat") or nd.get("y", 0))
            n_lon = float(nd.get("lon") or nd.get("geo_lon") or nd.get("x", 0))
            if (n_lat - lat) ** 2 + (n_lon - lon) ** 2 <= deg_radius_sq:
                local_nodes.add(n)

        # Extract edges within this local footprint
        sub_g = graph.subgraph(local_nodes)
        edge_count = 0
        for u, v in sub_g.edges():
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            u_lat = float(u_data.get("lat") or u_data.get("geo_lat") or u_data.get("y", 0))
            u_lon = float(u_data.get("lon") or u_data.get("geo_lon") or u_data.get("x", 0))
            v_lat = float(v_data.get("lat") or v_data.get("geo_lat") or v_data.get("y", 0))
            v_lon = float(v_data.get("lon") or v_data.get("geo_lon") or v_data.get("x", 0))
            surrounding_roads.append([[u_lat, u_lon], [v_lat, v_lon]])
            edge_count += 1
            if edge_count >= 1600:
                break

        # -------------------------------------------------------------
        # Compute main city connector routes (CBD, Majestic, Victoria)
        # -------------------------------------------------------------
        largest_cc = max(nx.connected_components(graph), key=len)

        # Ensure start node connects to largest component if needed
        start_node = nearest_node_id
        if start_node not in largest_cc and len(largest_cc) > 0:
            start_node = min(
                largest_cc,
                key=lambda n: (float(graph.nodes[n].get("lat", 0)) - lat) ** 2
                + (float(graph.nodes[n].get("lon", 0)) - lon) ** 2,
            )

        for hub_key, hub_info in MAIN_CITY_HUBS.items():
            hub_target_node, hub_n_lat, hub_n_lon, _ = _find_nearest_graph_node(
                graph, hub_info["lat"], hub_info["lon"]
            )
            if hub_target_node not in largest_cc and len(largest_cc) > 0:
                hub_target_node = min(
                    largest_cc,
                    key=lambda n: (float(graph.nodes[n].get("lat", 0)) - hub_info["lat"]) ** 2
                    + (float(graph.nodes[n].get("lon", 0)) - hub_info["lon"]) ** 2,
                )

            path_coords: list[list[float]] = []
            route_dist_km = 0.0
            gk_crossed = 0

            try:
                if start_node is not None and hub_target_node is not None and nx.has_path(graph, start_node, hub_target_node):
                    path = nx.shortest_path(graph, start_node, hub_target_node)
                    path_coords.append([lat, lon])

                    for i, node in enumerate(path):
                        nd = graph.nodes[node]
                        n_lat = float(nd.get("lat") or nd.get("geo_lat") or nd.get("y", 0))
                        n_lon = float(nd.get("lon") or nd.get("geo_lon") or nd.get("x", 0))
                        path_coords.append([n_lat, n_lon])

                        if i > 0:
                            prev_node = path[i - 1]
                            p_data = graph.nodes[prev_node]
                            p_lat = float(p_data.get("lat") or p_data.get("geo_lat") or p_data.get("y", 0))
                            p_lon = float(p_data.get("lon") or p_data.get("geo_lon") or p_data.get("x", 0))
                            route_dist_km += haversine_distance_km(p_lat, p_lon, n_lat, n_lon)

                        c_val = centrality.get(str(node), 0) if centrality else 0
                        if c_val >= gk_threshold:
                            gk_crossed += 1

                    path_coords.append([hub_info["lat"], hub_info["lon"]])
                else:
                    path_coords = [[lat, lon], [hub_info["lat"], hub_info["lon"]]]
                    route_dist_km = haversine_distance_km(lat, lon, hub_info["lat"], hub_info["lon"]) * 1.25
            except Exception:
                path_coords = [[lat, lon], [hub_info["lat"], hub_info["lon"]]]
                route_dist_km = haversine_distance_km(lat, lon, hub_info["lat"], hub_info["lon"]) * 1.25

            est_time_min = max(2, int(round((route_dist_km / 22.0) * 60)))

            main_city_routes[hub_key] = {
                "hub_id": hub_key,
                "hub_name": hub_info["name"],
                "name": hub_info["name"],
                "corridor_name": hub_info["corridor_name"],
                "color": hub_info["color"],
                "weight": hub_info["weight"],
                "destination_lat": hub_info["lat"],
                "destination_lon": hub_info["lon"],
                "distance_km": round(route_dist_km, 2),
                "direct_dist_km": round(haversine_distance_km(lat, lon, hub_info["lat"], hub_info["lon"]), 2),
                "est_time_min": est_time_min,
                "estimated_time_min": est_time_min,
                "gatekeepers_crossed": gk_crossed,
                "path_coords": path_coords,
                "coordinates": path_coords,
            }

    ingress_point = {
        "lat": ingress_lat,
        "lon": ingress_lon,
        "node_id": nearest_node_id,
        "distance_m": round(nearest_road_dist_m, 1),
    }

    recommendations: list[dict[str, Any]] = []

    # Rule A: Emergency Trauma Care & Disaster Relief Base
    if dist_hospital_km > 3.8 and (local_centrality > 0.03 or local_degree >= 3):
        score = min(98, int(80 + (dist_hospital_km * 2.5) + (local_centrality * 120)))
        recommendations.append({
            "type": "Emergency Trauma Care Center & Disaster Response Base",
            "category": "Healthcare & Disaster Resilience",
            "suitability_score": score,
            "badge_color": "#e74c3c",
            "rationale": (
                f"Located {dist_hospital_km:.1f} km from Victoria Hospital. "
                "During acute city-wide disasters or bridge collapses, peripheral zones become isolated. "
                "Constructing a secondary trauma and ambulance staging facility here directly cuts emergency golden-hour response times by up to 65%."
            ),
            "est_budget_cr": f"Rs. {max(30, int(area_acres * 4.2))} Cr",
            "roi_timeline": "3 - 5 Years (Public-Private Partnership)",
            "resilience_impact": "Critical life-saving triage hub; guarantees medical care when main arterial roads are severed.",
        })

    # Rule B: Multi-Modal Transit Terminal & Elevated Park-and-Ride
    if nearest_landmark_dist_km < 3.2 and (area_acres >= 10.0 or local_degree >= 4):
        score = min(96, int(82 + (area_acres * 0.8) + (local_degree * 2)))
        recommendations.append({
            "type": "Multi-Modal Transit Terminal & Park-and-Ride Hub",
            "category": "Transit & Mobility Infrastructure",
            "suitability_score": score,
            "badge_color": "#3498db",
            "rationale": (
                f"Situated {nearest_landmark_dist_km:.1f} km from bottleneck '{nearest_landmark_name}'. "
                "Constructing an integrated bus rapid transit (BRT), electric vehicle charging depot, "
                "and multi-level parking facility diverts heavy private commuter vehicles before they congest the junction."
            ),
            "est_budget_cr": f"Rs. {max(45, int(area_acres * 5.5))} Cr",
            "roi_timeline": "4 - 6 Years",
            "resilience_impact": "Absorbs peak traffic surges and provides staging grounds for mass evacuation buses during floods.",
        })

    # Rule C: Smart Stormwater Retention Basin & Urban Eco-Park (Sponge City)
    if "Silk Board" in nearest_landmark_name or "ORR" in nearest_landmark_name or lat < 12.935:
        score = 92
        recommendations.append({
            "type": "Sponge-City Stormwater Retention Basin & Eco-Park",
            "category": "Climate Adaptation & Flood Mitigation",
            "suitability_score": score,
            "badge_color": "#1abc9c",
            "rationale": (
                "Urban hydrology in this sector suffers from severe monsoon surface runoff due to concrete density. "
                "Constructing an engineered sub-surface rainwater reservoir with an above-ground public sports and biodiversity park "
                "captures up to 45 million liters of floodwater, preventing underpass submergence."
            ),
            "est_budget_cr": f"Rs. {max(20, int(area_acres * 2.8))} Cr",
            "roi_timeline": "2 - 4 Years (Municipal Green Bonds)",
            "resilience_impact": "Directly eliminates monsoon paralysis at adjoining arterial intersections.",
        })

    # Rule D: Cold-Chain Logistics Warehouse & Automated Urban Depot
    if area_acres >= 14.0:
        score = min(94, int(78 + (area_acres * 0.9)))
        recommendations.append({
            "type": "Regional Logistics & Cold-Chain Distribution Depot",
            "category": "Commercial & Supply Chain",
            "suitability_score": score,
            "badge_color": "#f39c12",
            "rationale": (
                f"Spacious {area_acres:.1f}-acre footprint allows heavy vehicular circulation without disrupting residential arteries. "
                "Essential for decentralized supply chains ensuring food, clean water, and pharmaceutical security during urban crises."
            ),
            "est_budget_cr": f"Rs. {max(50, int(area_acres * 4.8))} Cr",
            "roi_timeline": "3 - 4 Years (High Commercial Viability)",
            "resilience_impact": "Safeguards city grocery and medical supply chains against logistics breakdown.",
        })

    # Rule E: High-Tech Innovation Campus & Solar Microgrid Substation
    score = 88
    recommendations.append({
        "type": "Commercial Innovation Campus & Solar Microgrid Substation",
        "category": "Sustainable Commercial Development",
        "suitability_score": score,
        "badge_color": "#9b59b6",
        "rationale": (
            "Robust economic utilization combining grade-A commercial office space with rooftop solar generation "
            "and battery energy storage systems (BESS). Capable of operating islanded microgrid power during regional grid failures."
        ),
        "est_budget_cr": f"Rs. {max(70, int(area_acres * 7.5))} Cr",
        "roi_timeline": "5 - 7 Years",
        "resilience_impact": "Provides auxiliary power resilience to telecom towers and hospitals during storms.",
    })

    recommendations.sort(key=lambda r: r["suitability_score"], reverse=True)

    return {
        "lat": lat,
        "lon": lon,
        "area_acres": area_acres,
        "dist_hospital_km": round(dist_hospital_km, 2),
        "nearest_landmark": nearest_landmark_name,
        "nearest_landmark_dist_km": nearest_landmark_dist_km,
        "nearest_road_dist_m": round(nearest_road_dist_m, 1),
        "local_centrality": round(local_centrality, 5),
        "local_degree": local_degree,
        "recommendations": recommendations,
        "primary_recommendation": recommendations[0] if recommendations else None,
        "ingress_point": ingress_point,
        "surrounding_roads": surrounding_roads,
        "main_city_routes": main_city_routes,
    }
