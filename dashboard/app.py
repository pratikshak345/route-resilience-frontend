"""
Route Resilience - Urban Mobility Criticality Platform
======================================================
Interactive Geospatial Engineering Dashboard for Satellite Road Extraction,
Topological Healing, Betweenness Centrality, Disaster Stress Testing,
and Urban Infrastructure Planning in Bengaluru, India.

Palette:
- Near Black Background: #050809
- Card & Surface Fill: #0C1013
- Hot Pink Accent: #FF0A85
- Chartreuse Accent: #CFFF00
- Soft Grey Text: #8E9DA8 / #5A6B75
- Subtle Borders: #162025 / #222D33
"""

from __future__ import annotations

import sys
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import networkx as nx
import streamlit as st
import folium
import branca.colormap as cm
import pandas as pd
from PIL import Image
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Resolve project root so sibling-module imports work
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from dashboard.metrics import (
    compute_iou, compute_dice, compute_connectivity_ratio,
    compute_topological_accuracy, compute_avg_path_length_error,
    format_metrics_report,
)
from graph_analysis.urban_planner import VACANT_PARCELS, MAIN_CITY_HUBS, evaluate_site_potential

# ╔══════════════════════════════════════════════════════════════╗
# ║                   CACHED DATA & GRAPH LOADERS                ║
# ╚══════════════════════════════════════════════════════════════╝

def _load_binary_mask(mask_path: Path) -> np.ndarray:
    """Load a road mask from .npy or image formats and return binary uint8 array."""
    suffix = mask_path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(str(mask_path))
    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        arr = np.array(Image.open(mask_path).convert("L"))
    else:
        raise ValueError(f"Unsupported mask format: {suffix}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


@st.cache_resource
def _load_real_graph() -> tuple[nx.Graph, dict[str, float]]:
    """Load the real healed graph and precomputed centrality.csv."""
    graph_path = config.GRAPHS_DIR / "healed_graph.graphml"
    if not graph_path.exists():
        graph_path = config.GRAPHS_DIR / "raw_graph.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing road graph at {graph_path}")

    G = nx.read_graphml(str(graph_path))
    lat_min, lat_max = config.AOI["south"], config.AOI["north"]
    lon_min, lon_max = config.AOI["west"], config.AOI["east"]

    ys = [float(G.nodes[n].get("pos_y", 0)) for n in G.nodes()]
    xs = [float(G.nodes[n].get("pos_x", 0)) for n in G.nodes()]
    max_y = max(ys) if ys and max(ys) > 0 else 1024.0
    max_x = max(xs) if xs and max(xs) > 0 else 1024.0

    for n in G.nodes():
        if "geo_lat" in G.nodes[n] and "geo_lon" in G.nodes[n]:
            lat = float(G.nodes[n]["geo_lat"])
            lon = float(G.nodes[n]["geo_lon"])
        elif "lat" in G.nodes[n] and "lon" in G.nodes[n]:
            lat = float(G.nodes[n]["lat"])
            lon = float(G.nodes[n]["lon"])
        else:
            y = float(G.nodes[n].get("pos_y", 0))
            x = float(G.nodes[n].get("pos_x", 0))
            lat = lat_max - (y / max_y) * (lat_max - lat_min)
            lon = lon_min + (x / max_x) * (lon_max - lon_min)
        G.nodes[n]["pos"] = (lon, lat)
        G.nodes[n]["lat"] = lat
        G.nodes[n]["lon"] = lon

    for u, v, d in G.edges(data=True):
        if "weight" in d:
            try:
                d["weight"] = float(d["weight"])
            except Exception:
                d["weight"] = 1.0
        else:
            d["weight"] = 1.0

    cent_file = config.GRAPHS_DIR / "centrality.csv"
    if cent_file.exists():
        df_c = pd.read_csv(str(cent_file))
        cent = dict(zip(df_c["node_id"].astype(str), df_c["betweenness"].astype(float)))
    else:
        cent = nx.betweenness_centrality(G, k=min(300, len(G)))

    return G, cent


@st.cache_data
def _load_mask_overlay(mask_name: str, color_rgba: tuple[int, int, int, int]) -> tuple[str, int, int, int]:
    """Cache conversion of binary masks to base64 data URLs for fast raster map overlay."""
    import io, base64
    from scipy.ndimage import binary_dilation

    mask_path = config.PREDICTIONS_DIR / mask_name
    mask = _load_binary_mask(mask_path)
    disp = binary_dilation(mask > 0, iterations=2)
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[disp] = list(color_rgba)

    img = Image.fromarray(overlay, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{img_b64}", int(mask.sum()), mask.shape[0], mask.shape[1]


@st.cache_data
def _load_mask_comparison_metrics() -> dict[str, Any]:
    """Cache compute_iou and compute_dice so Tab 1 loads instantaneously."""
    raw = _load_binary_mask(config.PREDICTIONS_DIR / "pred_mask.npy")
    healed = _load_binary_mask(config.PREDICTIONS_DIR / "healed_mask.npy")
    return {
        "iou": float(compute_iou(raw, healed)),
        "dice": float(compute_dice(raw, healed)),
        "raw_pixels": int(raw.sum()),
        "healed_pixels": int(healed.sum()),
    }


@st.cache_data
def _get_criticality_display_data():
    """Cache top criticality edges and gatekeepers so Heatmap opens in 5ms."""
    G, cent = _load_real_graph()
    cent_values = list(cent.values())
    cent_min = min(cent_values) if cent_values else 0
    cent_max = max(cent_values) if cent_values else 1
    threshold = float(np.percentile(cent_values, config.GATEKEEPER_PERCENTILE))
    gatekeeper_nodes = [n for n, c in cent.items() if c >= threshold]

    all_edges = list(G.edges())
    scored_edges = []
    for u, v in all_edges:
        ec = (cent.get(str(u), 0) + cent.get(str(v), 0)) / 2
        scored_edges.append((ec, u, v))
    scored_edges.sort(reverse=True)

    edge_coords = []
    for ec, u, v in scored_edges[:2600]:
        pu = G.nodes[u].get("pos", (0, 0))
        pv = G.nodes[v].get("pos", (0, 0))
        edge_coords.append(([(pu[1], pu[0]), (pv[1], pv[0])], ec, str(u), str(v)))

    top_gks = []
    for n in sorted(gatekeeper_nodes, key=lambda x: cent.get(str(x), 0), reverse=True)[:100]:
        pos = G.nodes[n].get("pos", (0, 0))
        top_gks.append((pos[1], pos[0], str(n), float(cent.get(str(n), 0)), int(G.degree(n))))

    return edge_coords, top_gks, cent_min, cent_max, len(gatekeeper_nodes)


@st.cache_data
def _get_site_evaluation_cached(parcel_id: str) -> dict[str, Any] | None:
    """Cache multi-criteria analysis and shortest-path city routes per parcel."""
    p = next((x for x in VACANT_PARCELS if x["id"] == parcel_id), None)
    if not p:
        return None
    G, cent = _load_real_graph()
    return evaluate_site_potential(
        lat=p["lat"],
        lon=p["lon"],
        area_acres=p["area_acres"],
        graph=G,
        centrality=cent,
    )


# ╔══════════════════════════════════════════════════════════════╗
# ║                   RELIABLE BASEMAP BUILDER                   ║
# ╚══════════════════════════════════════════════════════════════╝

def create_base_map(
    lat: float,
    lon: float,
    zoom_start: int = 12,
    dark_mode: bool = True,
    **kwargs: Any,
) -> folium.Map:
    """
    Creates an OpenStreetMap Folium map without API keys or watermarks.
    Applies CSS filtering to render OSM raster tiles in a high-contrast dark palette
    (#050809) matching the application theme while preserving vector overlay fidelity.
    """
    m = folium.Map(
        location=[lat, lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        **kwargs,
    )
    if dark_mode:
        m.get_root().header.add_child(folium.Element("""
        <style>
            .leaflet-tile-pane {
                filter: invert(100%) hue-rotate(180deg) brightness(80%) contrast(95%) !important;
            }
            .leaflet-container {
                background: #050809 !important;
                border-radius: 8px !important;
                outline: none !important;
            }
            .leaflet-control-zoom a {
                background-color: #0C1013 !important;
                color: #FFFFFF !important;
                border-color: #1F282D !important;
            }
            .leaflet-control-zoom a:hover {
                background-color: #162025 !important;
                color: #CFFF00 !important;
            }
            .leaflet-control-layers {
                background: #0C1013 !important;
                color: #FFFFFF !important;
                border: 1px solid #1F282D !important;
                border-radius: 6px !important;
            }
            .leaflet-popup-content-wrapper, .leaflet-popup-tip {
                background: #0C1013 !important;
                color: #FFFFFF !important;
                border: 1px solid #1F282D !important;
                box-shadow: 0 4px 16px rgba(0,0,0,0.8) !important;
            }
        </style>
        """))
    return m


# ╔══════════════════════════════════════════════════════════════╗
# ║                     PAGE CONFIGURATION                       ║
# ╚══════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="Route Resilience",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern near-black (#050809), Hot Pink (#FF0A85) & Chartreuse (#CFFF00) theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Clean browser reset */
    html, body {
        background-color: #050809 !important;
        scroll-behavior: smooth;
    }

    /* Hide default Streamlit header and footer completely */
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0% !important;
        visibility: hidden !important;
    }
    footer {
        display: none !important;
        visibility: hidden !important;
    }
    #MainMenu {
        visibility: hidden !important;
    }

    .stApp {
        background-color: #050809 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #FFFFFF !important;
    }

    /* Maximize available viewport space without extra top margin */
    .main .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
        letter-spacing: -0.03em !important;
        font-weight: 700 !important;
    }

    /* Two-Column Hero Section */
    .hero-title {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.04em !important;
        color: #FFFFFF !important;
        margin: 0 0 0.35rem 0 !important;
        line-height: 1.1 !important;
    }
    .hero-title span.pink-accent {
        color: #FF0A85 !important;
    }
    .hero-headline {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.35rem !important;
        color: #CFFF00 !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        margin: 0 0 0.6rem 0 !important;
        line-height: 1.3 !important;
    }
    .hero-description {
        font-size: 0.95rem !important;
        color: #8E9DA8 !important;
        line-height: 1.6 !important;
        margin-bottom: 1.2rem !important;
    }

    /* Subtle Road-Inspired Decorative Pulse Line */
    .road-divider {
        height: 2px;
        background: linear-gradient(90deg, #FF0A85 0%, #CFFF00 55%, transparent 100%);
        margin: 0.8rem 0 1.1rem 0;
        border-radius: 2px;
        opacity: 0.9;
    }

    /* Hero Quick Highlight Stats */
    .hero-stats-row {
        display: flex;
        gap: 0.9rem;
        margin-top: 1rem;
        margin-bottom: 1.4rem;
        flex-wrap: wrap;
    }
    .hero-stat-item {
        background: #0C1013;
        border: 1px solid #162025;
        border-radius: 6px;
        padding: 0.65rem 0.95rem;
        flex: 1;
        min-width: 95px;
    }
    .hero-stat-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.30rem;
        font-weight: 700;
        color: #FFFFFF;
        display: block;
        line-height: 1.2;
    }
    .hero-stat-lbl {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #8E9DA8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Four Compact Feature Cards Grid */
    .feature-card-wrapper {
        background: #0C1013;
        border: 1px solid #162025;
        border-radius: 8px 8px 0 0;
        padding: 1.05rem 1.15rem 0.95rem 1.15rem;
        min-height: 105px;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .feature-card-wrapper:hover {
        border-color: #28353D;
        transform: translateY(-2px);
    }
    .feature-card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }
    .feature-icon-badge {
        font-size: 1.05rem;
        color: #CFFF00;
    }
    .feature-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        color: #FF0A85;
        letter-spacing: 0.5px;
    }
    .feature-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 4px;
    }
    .feature-desc-bold {
        font-size: 0.85rem;
        color: #8E9DA8;
        line-height: 1.4;
    }

    /* Button integrated into bottom of feature card */
    div.stButton > button {
        background: #10161A !important;
        color: #CFFF00 !important;
        border: 1px solid #162025 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.84rem !important;
        border-radius: 0 0 8px 8px !important;
        padding: 0.60rem 1.1rem !important;
        letter-spacing: 0.2px !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        margin-top: -1px !important;
    }
    div.stButton > button:hover {
        background: #FF0A85 !important;
        border-color: #FF0A85 !important;
        color: #050809 !important;
        transform: none !important;
    }

    /* Dedicated Site Planner Entry Strip */
    .site-planner-banner {
        background: #0C1013;
        border: 1px solid #162025;
        border-radius: 8px 8px 0 0;
        padding: 1.1rem 1.4rem;
        margin-top: 1.4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
        transition: border-color 0.2s ease;
    }
    .site-planner-banner:hover {
        border-color: #28353D;
    }
    .sp-left {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .sp-icon {
        font-size: 1.4rem;
        color: #CFFF00;
        background: #10161A;
        border: 1px solid #1F282D;
        width: 44px;
        height: 44px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .sp-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        color: #FF0A85;
        letter-spacing: 0.6px;
    }
    .sp-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.10rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .sp-desc {
        font-size: 0.83rem;
        color: #8E9DA8;
    }

    /* Map Frame */
    iframe {
        border-radius: 8px !important;
        border: 1px solid #162025 !important;
        background-color: #050809 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.7) !important;
        width: 100% !important;
    }

    /* Legend strip */
    .legend-strip {
        display: flex;
        align-items: center;
        gap: 16px;
        font-size: 0.78rem;
        color: #8E9DA8;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.5rem;
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .legend-line-pink {
        width: 14px;
        height: 3px;
        background-color: #FF0A85;
        border-radius: 2px;
        display: inline-block;
    }
    .legend-line-chartreuse {
        width: 14px;
        height: 2px;
        background-color: #CFFF00;
        border-radius: 2px;
        display: inline-block;
    }
    .legend-dot-pink {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #FF0A85;
        border: 1.5px solid #CFFF00;
        display: inline-block;
    }

    /* Compact Left Sidebar - REMOVE NESTED BUTTON LOOK AND EXTRA BUTTONS */
    section[data-testid="stSidebar"] {
        background-color: #050809 !important;
        border-right: 1px solid #141A1E !important;
        width: 255px !important;
        min-width: 255px !important;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p {
        color: #8E9DA8 !important;
    }

    .sidebar-brand {
        padding: 4px 6px 14px 6px;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #141A1E;
    }
    .brand-row {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-glyph {
        color: #FF0A85;
        font-size: 1.2rem;
    }
    .sidebar-brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }
    .sidebar-brand-sub {
        font-size: 0.74rem;
        color: #5A6B75;
        margin-top: 2px;
        margin-left: 24px;
    }
    .sidebar-nav-title {
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: #5A6B75;
        text-transform: uppercase;
        margin: 0.6rem 0 0.4rem 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Hide the extra Mission Navigation label / button completely */
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > label:first-child {
        display: none !important;
    }

    /* Clean navigation items in sidebar without nested button appearance */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 4px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: #0C1013 !important;
        border: 1px solid #162025 !important;
        border-radius: 6px !important;
        padding: 9px 12px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        color: #8E9DA8 !important;
        cursor: pointer !important;
        transition: all 0.16s ease !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        border-color: #28353D !important;
        color: #FFFFFF !important;
        background: #11161A !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
        background: #11161A !important;
        border: 1px solid #28353D !important;
        border-left: 3px solid #FF0A85 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    .sidebar-footer {
        padding: 1rem 0;
        font-size: 0.72rem;
        color: #445058;
        border-top: 1px solid #141A1E;
        margin-top: 2rem;
    }

    /* Native Metric Cards (Module Pages) */
    div[data-testid="stMetric"] {
        background: #0C1013 !important;
        border: 1px solid #162025 !important;
        border-radius: 6px !important;
        padding: 0.9rem !important;
        box-shadow: none !important;
        transition: border-color 0.2s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #28353D !important;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #8E9DA8 !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #CFFF00 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.55rem !important;
        font-weight: 700 !important;
    }

    /* Selectboxes and Form Inputs */
    div[data-baseweb="select"] > div {
        background-color: #0C1013 !important;
        border-color: #1F282D !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
    }

    /* Dataframe & Tables */
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        background-color: #0C1013 !important;
        border: 1px solid #162025 !important;
        border-radius: 6px !important;
        overflow: hidden !important;
    }

    /* Info card */
    .info-card {
        background: #0C1013;
        border-left: 3px solid #CFFF00;
        padding: 0.85rem 1.15rem;
        border-radius: 0 6px 6px 0;
        margin: 0.8rem 0 1.2rem 0;
        border-top: 1px solid #162025;
        border-right: 1px solid #162025;
        border-bottom: 1px solid #162025;
        color: #8E9DA8;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* Unified Module Page Headers */
    .page-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        font-weight: 700;
        color: #FF0A85;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .module-title {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
        color: #FFFFFF !important;
        margin: 0 0 0.35rem 0 !important;
        line-height: 1.2 !important;
    }
    .module-title span.pink-accent {
        color: #FF0A85 !important;
    }
    .module-subtitle {
        font-size: 0.92rem;
        color: #8E9DA8;
        line-height: 1.5;
        margin-bottom: 0.8rem;
    }
    .map-panel-header {
        background: #0C1013;
        border: 1px solid #162025;
        border-bottom: none;
        border-radius: 6px 6px 0 0;
        padding: 0.60rem 0.90rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.88rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .badge-pink {
        background: rgba(255, 10, 133, 0.15);
        border: 1px solid #FF0A85;
        color: #FF0A85;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        margin-right: 6px;
    }
    .badge-chartreuse {
        background: rgba(207, 255, 0, 0.15);
        border: 1px solid #CFFF00;
        color: #CFFF00;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        margin-right: 6px;
    }
    .ctrl-card {
        background: #0C1013;
        border: 1px solid #162025;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .ctrl-card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Section divider */
    .section-divider {
        height: 1px;
        background: #141A1E;
        margin: 1.8rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║                  NAVIGATION & SIDEBAR MODULE                 ║
# ╚══════════════════════════════════════════════════════════════╝

NAV_PAGES = [
    "Home",
    "Road Healing",
    "Criticality Analysis",
    "Disaster Simulator",
    "Urban Site Planner",
]

NAV_ICONS = {
    "Home": "○  Home",
    "Road Healing": "◇  Road Healing",
    "Criticality Analysis": "△  Criticality Analysis",
    "Disaster Simulator": "⚡  Disaster Simulator",
    "Urban Site Planner": "□  Urban Site Planner",
}

# Legacy tab name migration support
OLD_TO_NEW_TABS = {
    "Project Overview & Home": "Home",
    "Topological Road Healing": "Road Healing",
    "Criticality & Gatekeeper Heatmap": "Criticality Analysis",
    "Criticality": "Criticality Analysis",
    "Disaster Stress Simulator": "Disaster Simulator",
    "Urban Site Planner & City Corridors": "Urban Site Planner",
    "Site Planner": "Urban Site Planner",
}

# Default to "Home" on startup
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# Migrate any legacy state in session
if st.session_state.current_page in OLD_TO_NEW_TABS:
    st.session_state.current_page = OLD_TO_NEW_TABS[st.session_state.current_page]

if st.session_state.current_page not in NAV_PAGES:
    st.session_state.current_page = "Home"

# Function to navigate from buttons/cards and synchronize sidebar
def navigate_to(target_page: str):
    target = OLD_TO_NEW_TABS.get(target_page, target_page)
    st.session_state.current_page = target
    st.session_state.nav_radio = target

def on_sidebar_change():
    st.session_state.current_page = st.session_state.nav_radio

# Ensure nav_radio is initialized
if "nav_radio" not in st.session_state or st.session_state.nav_radio not in NAV_PAGES:
    st.session_state.nav_radio = st.session_state.current_page

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-row">
            <span class="brand-glyph">⬡</span>
            <span class="sidebar-brand-title">ROUTE RESILIENCE<span style="color:#FF0A85;">.</span></span>
        </div>
        <div class="sidebar-brand-sub">Road Network Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-nav-title'>NAVIGATION</div>", unsafe_allow_html=True)

    # Clean sidebar radio without extra button appearance
    st.radio(
        label="Navigation Menu",
        options=NAV_PAGES,
        format_func=lambda x: NAV_ICONS.get(x, x),
        key="nav_radio",
        on_change=on_sidebar_change,
        label_visibility="collapsed",
    )

    st.markdown("""
    <div class="sidebar-footer">
        <div>Bengaluru Road Network</div>
        <div style="font-size: 0.72rem; color: #445058; margin-top: 2px;">Sentinel-2 • OpenStreetMap</div>
    </div>
    """, unsafe_allow_html=True)

active_page = st.session_state.current_page


# ============================================================
# 1. HOME — HERO TWO-COLUMN LANDING PAGE & ROAD NETWORK MAP
# ============================================================
if active_page == "Home":
    # Preload authentic project graph data (NetworkX & Betweenness Centrality)
    G_home, cent_home = _load_real_graph()
    edge_coords, top_gks, cent_min, cent_max, num_gks = _get_criticality_display_data()

    # Two-Column Hero Layout: Intro on Left, Map on Right
    hero_left, hero_right = st.columns([1.05, 1.25], gap="large")

    with hero_left:
        st.markdown("""
        <h1 class="hero-title">Route Resilience<span class="pink-accent">.</span></h1>
        <div class="hero-headline">Smarter Roads. Stronger Cities.</div>
        <div class="road-divider"></div>
        <div class="hero-description">
            Map road-network weaknesses, identify critical connections, and explore urban disaster risks across Bengaluru.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="hero-stats-row">
            <div class="hero-stat-item">
                <span class="hero-stat-val">{len(G_home.nodes()):,}</span>
                <span class="hero-stat-lbl">Mapped Vertices</span>
            </div>
            <div class="hero-stat-item">
                <span class="hero-stat-val">{len(edge_coords):,}+</span>
                <span class="hero-stat-lbl">Critical Arterials</span>
            </div>
            <div class="hero-stat-item">
                <span class="hero-stat-val">{num_gks:,}</span>
                <span class="hero-stat-lbl">Gatekeeper Nodes</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.button(
            "Explore the Network →",
            key="btn_hero_explore",
            on_click=navigate_to,
            args=("Criticality Analysis",),
            use_container_width=True,
        )

    with hero_right:
        # Prominent Bengaluru Road-Network Map (OpenStreetMap with dark tactical filter, zero watermark)
        m_home = create_base_map(
            lat=config.AOI["center_lat"],
            lon=config.AOI["center_lon"],
            zoom_start=12,
            dark_mode=True,
        )

        # Upper-criticality corridors in Hot Pink (#FF0A85), connective network in Chartreuse (#CFFF00)
        num_edges = len(edge_coords)
        hot_pink_cutoff = int(num_edges * 0.35)

        for i, (locs, edge_cent, u_str, v_str) in enumerate(edge_coords):
            if i < hot_pink_cutoff:
                color = "#FF0A85"
                weight = 2.6
                opacity = 0.92
            else:
                color = "#CFFF00"
                weight = 1.6
                opacity = 0.75

            folium.PolyLine(
                locations=locs,
                color=color,
                weight=weight,
                opacity=opacity,
                tooltip=f"Road {u_str}-{v_str} | Centrality: {edge_cent:.5f}",
            ).add_to(m_home)

        # Critical intersections (gatekeeper chokepoints) in Hot Pink with Chartreuse core
        for lat_g, lon_g, n_id, c_val, deg_val in top_gks[:65]:
            folium.CircleMarker(
                location=[lat_g, lon_g],
                radius=4.5,
                color="#FF0A85",
                weight=1.5,
                fill=True,
                fill_color="#CFFF00",
                fill_opacity=1.0,
                tooltip=f"Intersection Node {n_id} (Centrality: {c_val:.4f})",
            ).add_to(m_home)

        folium.LayerControl().add_to(m_home)
        st_folium(m_home, height=440, use_container_width=True, returned_objects=[], key="landing_road_network")

        # Legend strip below map
        st.markdown("""
        <div class="legend-strip" style="justify-content: flex-end;">
            <span class="legend-item"><span class="legend-line-pink"></span> Critical Arterials</span>
            <span class="legend-item"><span class="legend-line-chartreuse"></span> Connective Corridors</span>
            <span class="legend-item"><span class="legend-dot-pink"></span> Strategic Gatekeepers</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Four Compact Capability Cards in 4 columns
    st.markdown("### Platform Modules")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        st.markdown("""
        <div class="feature-card-wrapper">
            <div class="feature-card-top">
                <span class="feature-icon-badge">◇</span>
                <span class="feature-tag">01 / HEALING</span>
            </div>
            <div class="feature-title">Road Healing</div>
            <div class="feature-desc-bold">Find and repair road gaps.</div>
        </div>
        """, unsafe_allow_html=True)
        st.button(
            "Open Road Healing →",
            key="btn_card_healing",
            on_click=navigate_to,
            args=("Road Healing",),
            use_container_width=True,
        )

    with f_col2:
        st.markdown("""
        <div class="feature-card-wrapper">
            <div class="feature-card-top">
                <span class="feature-icon-badge">△</span>
                <span class="feature-tag">02 / CRITICALITY</span>
            </div>
            <div class="feature-title">Criticality Analysis</div>
            <div class="feature-desc-bold">Identify vulnerable links.</div>
        </div>
        """, unsafe_allow_html=True)
        st.button(
            "Open Criticality →",
            key="btn_card_crit",
            on_click=navigate_to,
            args=("Criticality Analysis",),
            use_container_width=True,
        )

    with f_col3:
        st.markdown("""
        <div class="feature-card-wrapper">
            <div class="feature-card-top">
                <span class="feature-icon-badge">⚡</span>
                <span class="feature-tag">03 / SIMULATION</span>
            </div>
            <div class="feature-title">Disaster Simulator</div>
            <div class="feature-desc-bold">Test disruption impacts.</div>
        </div>
        """, unsafe_allow_html=True)
        st.button(
            "Open Simulator →",
            key="btn_card_disaster",
            on_click=navigate_to,
            args=("Disaster Simulator",),
            use_container_width=True,
        )

    with f_col4:
        st.markdown("""
        <div class="feature-card-wrapper">
            <div class="feature-card-top">
                <span class="feature-icon-badge">□</span>
                <span class="feature-tag">04 / CIVIC AI</span>
            </div>
            <div class="feature-title">Urban Site Planner</div>
            <div class="feature-desc-bold">Assess potential locations.</div>
        </div>
        """, unsafe_allow_html=True)
        st.button(
            "Open Site Planner →",
            key="btn_card_planner",
            on_click=navigate_to,
            args=("Urban Site Planner",),
            use_container_width=True,
        )


# ============================================================
# 2. ROAD HEALING — RAW VS HEALED ROAD MASKS
# ============================================================
elif active_page == "Road Healing":
    st.markdown("""
    <div class="page-tag">01 / SATELLITE ROAD EXTRACTION</div>
    <h2 class="module-title">Topological Road Healing & Occlusion Recovery<span class="pink-accent">.</span></h2>
    <div class="module-subtitle">Compare raw neural segmentation against the topologically reconstructed road network across the 1677×1612 Bengaluru AOI (10m/px).</div>
    <div class="info-card">
        <span style="color:#CFFF00; font-weight:700;">Topological Recovery:</span> Optical sensors lose road visibility under dense tree canopies and building shadows. Mathematical skeletonization and directional alignment bridge gaps into continuous topological lifelines.
    </div>
    """, unsafe_allow_html=True)

    bounds = [
        [config.AOI["south"], config.AOI["west"]],
        [config.AOI["north"], config.AOI["east"]],
    ]

    col_raw, col_healed = st.columns(2)

    # Use Hot Pink (#FF0A85) for raw occlusion mask and Chartreuse (#CFFF00) for healed continuous mask
    raw_url, raw_pixels, h_raw, w_raw = _load_mask_overlay("pred_mask.npy", (255, 10, 133, 230))
    healed_url, healed_pixels, h_heal, w_heal = _load_mask_overlay("healed_mask.npy", (207, 255, 0, 230))

    with col_raw:
        st.markdown("""
        <div class="map-panel-header">
            <span><span class="badge-pink">RAW</span> Neural Segmentation Output</span>
            <span style="color: #8E9DA8; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;">Occlusion Dropouts</span>
        </div>
        """, unsafe_allow_html=True)
        m_raw = create_base_map(
            lat=config.AOI["center_lat"],
            lon=config.AOI["center_lon"],
            zoom_start=config.MAP_ZOOM_START,
            dark_mode=True,
        )
        folium.raster_layers.ImageOverlay(
            image=raw_url,
            bounds=bounds,
            opacity=0.90,
            name="Raw Mask",
        ).add_to(m_raw)
        folium.LayerControl().add_to(m_raw)
        st_folium(m_raw, height=460, use_container_width=True, returned_objects=[], key="map_raw_real")
        st.caption(f"Raw neural detection: {raw_pixels:,} pixels with canopy dropouts.")

    with col_healed:
        st.markdown("""
        <div class="map-panel-header">
            <span><span class="badge-chartreuse">HEALED</span> Topologically Reconnected Mask</span>
            <span style="color: #8E9DA8; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;">Continuous Corridors</span>
        </div>
        """, unsafe_allow_html=True)
        m_healed = create_base_map(
            lat=config.AOI["center_lat"],
            lon=config.AOI["center_lon"],
            zoom_start=config.MAP_ZOOM_START,
            dark_mode=True,
        )
        folium.raster_layers.ImageOverlay(
            image=healed_url,
            bounds=bounds,
            opacity=0.90,
            name="Healed Mask",
        ).add_to(m_healed)
        folium.LayerControl().add_to(m_healed)
        st_folium(m_healed, height=460, use_container_width=True, returned_objects=[], key="map_healed_real")
        st.caption(f"Healed output: {healed_pixels:,} pixels with bridged topological continuity.")

    # Mask Metrics
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### Extraction & Healing Quality Indices")
    tab1_m = _load_mask_comparison_metrics()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("IoU Similarity", f"{tab1_m['iou']:.4f}", help="Intersection over Union agreement")
    m2.metric("Dice / F1 Score", f"{tab1_m['dice']:.4f}", help="Harmonic mean of precision and recall")
    m3.metric("Raw Road Pixels", f"{tab1_m['raw_pixels']:,}")
    m4.metric(
        "Healed Road Pixels",
        f"{tab1_m['healed_pixels']:,}",
        delta=f"{tab1_m['healed_pixels'] - tab1_m['raw_pixels']:,} (Noise Pruned)",
    )


# ============================================================
# 3. CRITICALITY ANALYSIS — GRAPH & BETWEENNESS CENTRALITY
# ============================================================
elif active_page == "Criticality Analysis":
    st.markdown("""
    <div class="page-tag">02 / GRAPH TOPOLOGY & SYSTEMIC RISK</div>
    <h2 class="module-title">Criticality & Gatekeeper Analysis<span class="pink-accent">.</span></h2>
    <div class="module-subtitle">Betweenness centrality modeling on Bengaluru's 45,148-node road network to identify vital lifelines and single points of failure.</div>
    <div class="info-card">
        <span style="color:#CFFF00; font-weight:700;">Betweenness Centrality:</span> Measures the fraction of shortest paths routing through each segment. <b style="color:#FF0A85;">Hot Pink lines</b> denote vital arterials; <b style="color:#CFFF00;">highlighted nodes</b> represent strategic gatekeepers where severance paralyzes connectivity.
    </div>
    """, unsafe_allow_html=True)

    G, cent = _load_real_graph()
    edge_coords, top_gks, cent_min, cent_max, num_gks = _get_criticality_display_data()

    st.markdown("""
    <div class="map-panel-header">
        <span>Arterial Criticality & Gatekeeper Heatmap</span>
        <span style="color: #8E9DA8; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;">Bengaluru AOI</span>
    </div>
    """, unsafe_allow_html=True)

    m_crit = create_base_map(
        lat=config.AOI["center_lat"],
        lon=config.AOI["center_lon"],
        zoom_start=config.MAP_ZOOM_START,
        dark_mode=True,
    )

    colormap = cm.LinearColormap(
        ["#CFFF00", "#FFA502", "#FF0A85"],
        vmin=cent_min, vmax=cent_max,
        caption="Betweenness Centrality: Connective (Chartreuse) → Arterial Lifeline (Hot Pink)",
    )
    colormap.add_to(m_crit)

    edge_group = folium.FeatureGroup(name="Arterial Network")
    for locs, edge_cent, u_str, v_str in edge_coords:
        folium.PolyLine(
            locations=locs,
            color=colormap(edge_cent),
            weight=2.8,
            opacity=0.88,
            tooltip=f"Road {u_str}-{v_str} | Centrality: {edge_cent:.5f}",
        ).add_to(edge_group)
    edge_group.add_to(m_crit)

    gk_group = folium.FeatureGroup(name="Gatekeeper Bottlenecks (Top 100)")
    for lat_g, lon_g, n_id, c_val, deg_val in top_gks:
        folium.CircleMarker(
            location=[lat_g, lon_g],
            radius=5.5,
            color="#FF0A85",
            weight=1.5,
            fill=True,
            fill_color="#CFFF00",
            fill_opacity=0.95,
            tooltip=f"Gatekeeper Node {n_id} | Centrality: {c_val:.4f}",
            popup=f"<b>Gatekeeper Node {n_id}</b><br>Centrality: {c_val:.5f}<br>Degree: {deg_val}",
        ).add_to(gk_group)
    gk_group.add_to(m_crit)

    folium.LayerControl().add_to(m_crit)
    st_folium(m_crit, height=520, use_container_width=True, returned_objects=[], key="map_crit_real")

    st.markdown("""
    <div class="legend-strip" style="justify-content: flex-end;">
        <span class="legend-item"><span class="legend-line-pink"></span> Arterial Lifelines (High Centrality)</span>
        <span class="legend-item"><span class="legend-line-chartreuse"></span> Connective Corridors</span>
        <span class="legend-item"><span class="legend-dot-pink"></span> Gatekeeper Bottlenecks</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### Road Network Topology Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Graph Nodes", f"{len(G.nodes()):,}")
    s2.metric("Total Graph Edges", f"{len(G.edges()):,}")
    s3.metric("Gatekeeper Nodes", f"{num_gks:,} (Top 5%)")
    s4.metric("Peak Centrality", f"{cent_max:.4f}")

    st.markdown("#### Top Strategic Gatekeeper Junctions (Critical Urban Chokepoints)")
    gk_table = []
    for lat_g, lon_g, n_id, c_val, deg_val in top_gks[:15]:
        gk_table.append({
            "Gatekeeper Node ID": n_id,
            "Betweenness Centrality": round(c_val, 5),
            "Connecting Arterials (Degree)": deg_val,
            "Latitude": round(lat_g, 5),
            "Longitude": round(lon_g, 5),
        })
    st.dataframe(pd.DataFrame(gk_table), use_container_width=True, hide_index=True)


# ============================================================
# 4. DISASTER SIMULATOR — DISRUPTIONS & ACCESSIBILITY
# ============================================================
elif active_page == "Disaster Simulator":
    st.markdown("""
    <div class="page-tag">03 / REAL-TIME RESILIENCE SIMULATION</div>
    <h2 class="module-title">Disaster Network Stress Test Simulator<span class="pink-accent">.</span></h2>
    <div class="module-subtitle">Simulate flood inundation, bridge collapses, and acute road severances to measure graph fragmentation and trauma hospital isolation.</div>
    <div class="info-card">
        <span style="color:#CFFF00; font-weight:700;">Stress Test Protocol:</span> Targeted arterial nodes are severed from the graph. Connected components, detour commute surges, and emergency isolation from <b>Victoria Hospital</b> are quantified dynamically.
    </div>
    """, unsafe_allow_html=True)

    G_original, cent = _load_real_graph()

    DISASTER_SCENARIOS = {
        "Monsoon Inundation - Central Silk Board Interchange": {
            "description": "Severe monsoon flash floods submerge the Silk Board underpass and flyover approaches, cutting off Hosur Road and the Electronic City Expressway.",
            "landmark": config.LANDMARKS["silk_board_junction"],
            "impact_profile": "South-East technology corridor paralyzed. Heavy commercial and commuter isolation.",
        },
        "Structural Bridge Collapse - KR Puram Railway Bridge": {
            "description": "Major structural severance on Old Madras Road at the KR Puram cable bridge isolates the Whitefield ITPL corridor from Central Bengaluru.",
            "landmark": config.LANDMARKS["kr_puram_bridge"],
            "impact_profile": "East Bengaluru freight and tech commuters severed. High detour overhead.",
        },
        "Approach Blockage / Landslide - Hebbal Flyover": {
            "description": "Heavy rainfall causes road collapse and debris inundation on Bellary Road, paralyzing the northern gateway to Kempegowda International Airport.",
            "landmark": config.LANDMARKS["hebbal_flyover"],
            "impact_profile": "Northern expressway and airport transit corridor blocked.",
        },
        "Underpass Submergence - ORR-Sarjapur Junction": {
            "description": "Flash flooding traps vehicles in the Outer Ring Road underpass, cutting off the Bellandur-Sarjapur tech belt.",
            "landmark": config.LANDMARKS["outer_ring_road_junction"],
            "impact_profile": "Bellandur tech parks and residential sectors cut off from arterial ring road.",
        },
        "Multi-Point Cascade Disaster (Silk Board + ORR + KR Puram)": {
            "description": "Severe simultaneous cloudburst flood paralyzes Silk Board, Sarjapur, and KR Puram concurrently, inducing city-wide gridlock.",
            "landmark": config.LANDMARKS["silk_board_junction"],
            "multi_landmarks": [
                config.LANDMARKS["silk_board_junction"],
                config.LANDMARKS["outer_ring_road_junction"],
                config.LANDMARKS["kr_puram_bridge"],
            ],
            "impact_profile": "Massive multi-quadrant fragmentation. Emergency medical corridors paralyzed.",
        },
    }

    col_ctrl, col_info = st.columns([1, 1.3], gap="medium")

    with col_ctrl:
        st.markdown("""
        <div class="ctrl-card">
            <div class="ctrl-card-title">1. Select Disaster Scenario Preset</div>
        """, unsafe_allow_html=True)
        scenario_name = st.selectbox("Disaster Preset", list(DISASTER_SCENARIOS.keys()), index=0)
        scenario = DISASTER_SCENARIOS[scenario_name]
        st.markdown(f"<div style='font-size:0.83rem; color:#8E9DA8; margin: 6px 0;'><b>Mechanism:</b> {scenario['description']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.83rem; color:#8E9DA8; margin-bottom: 8px;'><b>Urban Impact:</b> <span style='color:#FF0A85;'>{scenario['impact_profile']}</span></div>", unsafe_allow_html=True)

        lm = scenario["landmark"]
        st.markdown(f"<div style='font-size:0.80rem; color:#5A6B75; font-family:\"JetBrains Mono\", monospace; margin-bottom: 8px;'>Target: {lm['name']} ({lm['lat']}N, {lm['lon']}E)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        run_btn = st.button("Execute Disaster Simulation", type="primary", use_container_width=True)

    with col_info:
        st.markdown("""
        <div class="ctrl-card">
            <div class="ctrl-card-title">2. Or Click Map to Target Custom Road Junction</div>
        """, unsafe_allow_html=True)

        m_click = create_base_map(
            lat=config.AOI["center_lat"],
            lon=config.AOI["center_lon"],
            zoom_start=config.MAP_ZOOM_START,
            dark_mode=True,
        )
        all_orig_nodes = list(G_original.nodes())
        sample_step = max(1, len(all_orig_nodes) // 500)
        for n in all_orig_nodes[::sample_step]:
            pos = G_original.nodes[n].get("pos", (0, 0))
            folium.CircleMarker(
                location=[pos[1], pos[0]], radius=3.5,
                color="#FF0A85", fill=True, fill_opacity=0.6,
                tooltip=f"Junction Node {n}",
            ).add_to(m_click)

        click_result = st_folium(m_click, height=225, use_container_width=True, returned_objects=["last_clicked"], key="click_map_real")

        custom_node = None
        if click_result and click_result.get("last_clicked"):
            clicked = click_result["last_clicked"]
            click_lat, click_lon = clicked["lat"], clicked["lng"]
            min_dist = float("inf")
            for n in G_original.nodes():
                pos = G_original.nodes[n].get("pos", (0, 0))
                dist = (pos[1] - click_lat) ** 2 + (pos[0] - click_lon) ** 2
                if dist < min_dist:
                    min_dist = dist
                    custom_node = n
            if custom_node is not None:
                st.info(f"Targeted Junction Node **{custom_node}** (closest node to map click)")
        st.markdown("</div>", unsafe_allow_html=True)

    if run_btn or custom_node is not None:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("### Simulation Impact Report & Network Fracture Analysis")

        if custom_node is not None and not run_btn:
            remove_nodes = [custom_node]
            sim_name = f"Custom Severance: Node {custom_node}"
        else:
            sim_name = scenario_name
            landmarks_to_sever = scenario.get("multi_landmarks", [scenario["landmark"]])
            remove_nodes = []
            for target_lm in landmarks_to_sever:
                t_lat, t_lon = target_lm["lat"], target_lm["lon"]
                nodes_sorted = sorted(
                    G_original.nodes(),
                    key=lambda n: (float(G_original.nodes[n].get("lat", 0)) - t_lat) ** 2
                    + (float(G_original.nodes[n].get("lon", 0)) - t_lon) ** 2,
                )
                remove_nodes.extend(nodes_sorted[:4])

        remove_nodes = list(dict.fromkeys(remove_nodes))

        # Intact baselines
        cr_before = 0.9998
        cc_before = 11
        apl_before = 278.25

        G_failed = G_original.copy()
        G_failed.remove_nodes_from(remove_nodes)

        cr_after = compute_connectivity_ratio(G_failed)
        cc_after = nx.number_connected_components(G_failed)
        if len(G_failed) > 0:
            largest_cc_f = max(nx.connected_components(G_failed), key=len)
            sub_f = G_failed.subgraph(largest_cc_f)
            if len(sub_f) > 1:
                sample_sources = list(sub_f.nodes())[:15]
                t_hops = 0
                t_cnt = 0
                for s in sample_sources:
                    sp_lens = nx.single_source_shortest_path_length(sub_f, s)
                    t_hops += sum(sp_lens.values())
                    t_cnt += len(sp_lens) - 1
                apl_after = (t_hops / max(1, t_cnt)) if t_cnt > 0 else apl_before
            else:
                apl_after = 0.0
        else:
            apl_after = float("inf")

        hospital_ref = config.REFERENCE_HUB
        hospital_node = min(
            G_original.nodes(),
            key=lambda n: (float(G_original.nodes[n].get("lat", 0)) - hospital_ref["lat"]) ** 2
            + (float(G_original.nodes[n].get("lon", 0)) - hospital_ref["lon"]) ** 2,
        )

        unreachable_before = len(G_original) - len(nx.node_connected_component(G_original, hospital_node))
        if hospital_node in G_failed:
            unreachable_after = len(G_failed) - len(nx.node_connected_component(G_failed, hospital_node))
        else:
            unreachable_after = len(G_failed)

        st.markdown(f"**Disaster Event:** `{sim_name}` | **Severed Core Nodes:** `{len(remove_nodes)}`")

        # Top-level Real-World Impact KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Connectivity Retention",
            f"{cr_after*100:.1f}%",
            delta=f"{(cr_after - cr_before)*100:.2f}%",
            delta_color="inverse",
            help="Percentage of total urban network that remains reachable"
        )
        k2.metric(
            "Disconnected Islands",
            f"{cc_after} Sub-Nets",
            delta=f"+{cc_after - cc_before} Isolated",
            delta_color="inverse",
            help="Number of fragmented disconnected network components"
        )
        k3.metric(
            "Detour Commute Overhead",
            f"{apl_after:.1f} Hops",
            delta=f"+{apl_after - apl_before:.1f} Delay",
            delta_color="inverse",
            help="Average routing path length increase caused by severance"
        )
        k4.metric(
            "Cutoff from Victoria Hospital",
            f"{unreachable_after} Nodes",
            delta=f"+{unreachable_after - unreachable_before} Trapped",
            delta_color="inverse",
            help="Number of road intersections unable to reach apex emergency medical trauma care"
        )

        # Actionable Civil Defense & Emergency Response Action Plan
        st.markdown("#### Civil Defense & Emergency Response Directives")
        st.markdown(f"""
        <div class="info-card">
            <b>Emergency Directives:</b>
            <ul style="margin-top: 0.3rem; margin-bottom: 0.2rem; line-height: 1.6; padding-left: 1.2rem;">
                <li><b>Traffic Diversion:</b> Severance of {len(remove_nodes)} arterial junction(s) requires immediate diversion via peripheral ring roads before congestion cascades.</li>
                <li><b>Emergency Medical Isolation:</b> <b>{unreachable_after} road sectors</b> cannot reach Victoria Hospital. Deploy secondary mobile trauma stabilization units immediately.</li>
                <li><b>Public Transit Re-Routing:</b> BMTC buses and emergency vehicles must avoid the {lm['name']} corridor and use secondary parallel connectors.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("#### Spatial Topography Comparison: Baseline vs. Post-Severance")
        map_b, map_a = st.columns(2)

        with map_b:
            st.markdown("""
            <div class="map-panel-header">
                <span><span class="badge-chartreuse">INTACT</span> Baseline Road Network</span>
                <span style="color: #8E9DA8; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;">Connected</span>
            </div>
            """, unsafe_allow_html=True)
            m_before = create_base_map(
                lat=config.AOI["center_lat"],
                lon=config.AOI["center_lon"],
                zoom_start=config.MAP_ZOOM_START + 1,
                dark_mode=True,
            )
            edges_b = list(G_original.edges())
            step_b = max(1, len(edges_b) // 1500)
            for u, v in edges_b[::step_b]:
                pu = G_original.nodes[u].get("pos", (0, 0))
                pv = G_original.nodes[v].get("pos", (0, 0))
                folium.PolyLine(
                    [(pu[1], pu[0]), (pv[1], pv[0])],
                    color="#CFFF00", weight=2.0, opacity=0.85,
                ).add_to(m_before)
            for n in remove_nodes:
                if n in G_original:
                    pos = G_original.nodes[n].get("pos", (0, 0))
                    folium.CircleMarker(
                        [pos[1], pos[0]], radius=7,
                        color="#FF0A85", fill=True, fill_color="#FF0A85",
                        tooltip=f"Severed Chokepoint Node {n}",
                    ).add_to(m_before)
            st_folium(m_before, height=420, use_container_width=True, returned_objects=[], key="map_before_real")

        with map_a:
            st.markdown("""
            <div class="map-panel-header">
                <span><span class="badge-pink">SEVERED</span> Post-Disruption Topography</span>
                <span style="color: #8E9DA8; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;">Isolated Clusters</span>
            </div>
            """, unsafe_allow_html=True)
            m_after = create_base_map(
                lat=config.AOI["center_lat"],
                lon=config.AOI["center_lon"],
                zoom_start=config.MAP_ZOOM_START + 1,
                dark_mode=True,
            )
            edges_a = list(G_failed.edges())
            step_a = max(1, len(edges_a) // 1500)
            for u, v in edges_a[::step_a]:
                pu = G_failed.nodes[u].get("pos", (0, 0))
                pv = G_failed.nodes[v].get("pos", (0, 0))
                folium.PolyLine(
                    [(pu[1], pu[0]), (pv[1], pv[0])],
                    color="#FF0A85", weight=2.0, opacity=0.85,
                ).add_to(m_after)

            component_colors = ["#00d2d3", "#f39c12", "#CFFF00", "#9b59b6", "#3498db"]
            top_comps = sorted(nx.connected_components(G_failed), key=len, reverse=True)[:8]
            for i, comp in enumerate(top_comps):
                color = component_colors[i % len(component_colors)]
                for n in list(comp)[:60]:
                    pos = G_failed.nodes[n].get("pos", (0, 0))
                    folium.CircleMarker(
                        [pos[1], pos[0]], radius=3.5,
                        color=color, fill=True, fill_color=color,
                        tooltip=f"Node {n} - Disconnected Cluster {i + 1}",
                    ).add_to(m_after)
            st_folium(m_after, height=420, use_container_width=True, returned_objects=[], key="map_after_real")


# ============================================================
# 5. URBAN SITE PLANNER — VACANT LAND & CORRIDOR ROUTING
# ============================================================
elif active_page == "Urban Site Planner":
    st.markdown("""
    <div class="page-tag">04 / CIVIC AI & INFRASTRUCTURE ADVISOR</div>
    <h2 class="module-title">Urban Site Planner & City Corridors<span class="pink-accent">.</span></h2>
    <div class="module-subtitle">Evaluate unbuilt land parcels, analyze surrounding road connectivity, and route shortest paths to key urban centers.</div>
    <div class="info-card">
        <span style="color:#CFFF00; font-weight:700;">Infrastructure Engine:</span> Identifies vacant urban land parcels, computes highway ingress access, and evaluates real shortest-path corridors to CBD, Majestic Hub, and Victoria Hospital for municipal constructors.
    </div>
    """, unsafe_allow_html=True)

    col_sel, col_stats = st.columns([1, 1.6], gap="medium")

    with col_sel:
        st.markdown("""
        <div class="ctrl-card">
            <div class="ctrl-card-title">Select Vacant Land Parcel</div>
        """, unsafe_allow_html=True)
        parcel_names = [p["name"] for p in VACANT_PARCELS]
        selected_name = st.selectbox("Parcel Selection", parcel_names, index=0)
        selected_parcel = next(p for p in VACANT_PARCELS if p["name"] == selected_name)

        st.markdown(f"<div style='font-size:0.85rem; color:#FFFFFF; margin-top:6px;'><b>Zone:</b> {selected_parcel['zone']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.85rem; color:#8E9DA8;'><b>Footprint:</b> <code style='color:#CFFF00; background:#10161A;'>{selected_parcel['area_acres']:.1f} Acres</code> (~{selected_parcel['area_acres']*0.4047:.1f} Ha)</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.85rem; color:#8E9DA8;'><b>Zoning:</b> <code style='color:#FF0A85; background:#10161A;'>{selected_parcel['zoning']}</code></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.85rem; color:#8E9DA8;'><b>Terrain:</b> {selected_parcel['terrain']}</div>", unsafe_allow_html=True)
        st.caption(selected_parcel["description"])
        st.markdown("</div>", unsafe_allow_html=True)

    eval_result = _get_site_evaluation_cached(selected_parcel["id"])
    if eval_result is None:
        G_active, cent_active = _load_real_graph()
        eval_result = evaluate_site_potential(
            lat=selected_parcel["lat"],
            lon=selected_parcel["lon"],
            area_acres=selected_parcel["area_acres"],
            graph=G_active,
            centrality=cent_active,
        )

    with col_stats:
        st.markdown("#### Geospatial Feasibility Index")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Parcel Area", f"{selected_parcel['area_acres']} Ac", help="Total developable parcel footprint")
        k2.metric("Ingress Distance", f"{eval_result['ingress_point']['distance_m']:.0f} m", help="Distance to road network ingress gate")
        k3.metric("Hospital Hub Radius", f"{eval_result['dist_hospital_km']:.1f} km", help="Distance to Victoria Hospital Emergency Trauma Hub")
        k4.metric("Suitability Score", f"{eval_result['primary_recommendation']['suitability_score']}%", delta="High Alignment")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("#### Vacant Land Parcels & Multi-Corridor City Connectivity")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    show_surrounding = ctrl_col1.checkbox("Show Surrounding Road Network (~1.8km)", value=True)
    show_routes = ctrl_col2.checkbox("Show Routes to Main City Centers", value=True)
    show_ingress = ctrl_col3.checkbox("Show Site Ingress Point", value=True)

    st.markdown("""
    <div class="map-panel-header">
        <span>Parcel Boundary & Multi-Destination Arterial Paths</span>
        <span style="color: #8E9DA8; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;">Bengaluru AOI</span>
    </div>
    """, unsafe_allow_html=True)

    m_vacant = create_base_map(
        lat=selected_parcel["lat"],
        lon=selected_parcel["lon"],
        zoom_start=13,
        dark_mode=True,
    )

    # 1. Surrounding Road Network Markings in Chartreuse
    if show_surrounding and eval_result.get("surrounding_roads"):
        road_group = folium.FeatureGroup(name="Surrounding Road Network")
        folium.PolyLine(
            locations=eval_result["surrounding_roads"],
            color="#CFFF00",
            weight=2.2,
            opacity=0.88,
            tooltip="Local Road Network Segment",
        ).add_to(road_group)
        road_group.add_to(m_vacant)

    # 2. Main City Connector Routes (Distinct Strategic Corridors)
    if show_routes and eval_result.get("main_city_routes"):
        routes_group = folium.FeatureGroup(name="Main City Arterial Corridors")
        for hub_key, route in eval_result["main_city_routes"].items():
            route_coords = route.get("path_coords", route.get("coordinates", []))
            if route_coords:
                # Use Hot Pink (#FF0A85) for CBD and Chartreuse (#CFFF00) for other hubs
                r_color = "#FF0A85" if hub_key == "CBD_CORE" else ("#00d2d3" if hub_key == "MAJESTIC_HUB" else "#CFFF00")
                folium.PolyLine(
                    locations=route_coords,
                    color=r_color,
                    weight=4.5,
                    opacity=0.92,
                    tooltip=f"{route.get('corridor_name', 'Corridor')} ({route.get('distance_km', 0)} km, ~{route.get('est_time_min', route.get('estimated_time_min', 0))} min)",
                ).add_to(routes_group)
        routes_group.add_to(m_vacant)

    # 3. Draw All Vacant Parcels
    for p in VACANT_PARCELS:
        is_selected = (p["id"] == selected_parcel["id"])
        color = "#FF0A85" if is_selected else "#CFFF00"
        fill_color = "#FF0A85" if is_selected else "#CFFF00"
        fill_opacity = 0.55 if is_selected else 0.25
        weight = 2.5 if is_selected else 1.2

        folium.Polygon(
            locations=p["polygon"],
            color=color,
            weight=weight,
            fill=True,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            tooltip=f"{p['name']} ({p['area_acres']} Ac)",
            popup=f"<b>{p['name']}</b><br>Area: {p['area_acres']} Acres<br>Zoning: {p['zoning']}",
        ).add_to(m_vacant)

    # 4. Site Ingress Gate Marker
    if show_ingress and "ingress_point" in eval_result:
        ing = eval_result["ingress_point"]
        folium.Marker(
            location=[ing["lat"], ing["lon"]],
            icon=folium.Icon(color="red", icon="sign-in", prefix="fa"),
            tooltip=f"Site Highway Ingress Gate ({ing['distance_m']:.0f}m from parcel centroid)",
            popup=f"<b>Site Ingress Gate</b><br>Connecting Road: {eval_result['nearest_landmark']}<br>Distance: {ing['distance_m']:.0f} m",
        ).add_to(m_vacant)
        folium.PolyLine(
            locations=[[selected_parcel["lat"], selected_parcel["lon"]], [ing["lat"], ing["lon"]]],
            color="#FF0A85",
            weight=2.5,
            dash_array="6",
            tooltip="Ingress Connection Spur",
        ).add_to(m_vacant)

    # 5. City Hub Destination Markers
    for h_k, h_v in MAIN_CITY_HUBS.items():
        folium.Marker(
            location=[h_v["lat"], h_v["lon"]],
            icon=folium.Icon(color="purple" if h_k == "CBD_CORE" else ("blue" if h_k == "MAJESTIC_HUB" else "green"), icon=h_v["icon"], prefix="fa"),
            tooltip=f"{h_v['name']} ({h_v['corridor_name']})",
            popup=f"<b>{h_v['name']}</b><br>{h_v['corridor_name']}",
        ).add_to(m_vacant)

    folium.LayerControl().add_to(m_vacant)
    st_folium(m_vacant, height=520, use_container_width=True, returned_objects=[], key="map_vacant_real")

    # City Connectivity Table
    if eval_result.get("main_city_routes"):
        st.markdown("#### Main City Arterial Connectivity & Ingress Breakdown")
        route_table_data = []
        for h_k, route in eval_result["main_city_routes"].items():
            route_table_data.append({
                "Destination Hub": route.get("hub_name", route.get("name", "City Hub")),
                "Strategic Connecting Corridor": route.get("corridor_name", "Connecting Corridor"),
                "Street Distance": f"{route.get('distance_km', 0)} km",
                "Transit Time (Normal)": f"~{route.get('est_time_min', route.get('estimated_time_min', 0))} min",
                "Traversed Gatekeeper Bottlenecks": f"{route.get('gatekeepers_crossed', 0)} Chokepoint(s)",
            })
        st.dataframe(pd.DataFrame(route_table_data), use_container_width=True, hide_index=True)

    # Constructor Recommendation Card
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### Constructor Development Advisory & Feasibility Matrix")

    prim = eval_result["primary_recommendation"]
    if prim:
        st.markdown(f"""
        <div style="background: #0C1013; padding: 1.5rem 1.8rem; border-radius: 8px; border: 1px solid #162025; border-left: 5px solid #FF0A85; margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.8rem;">
                <span style="font-size: 1.25rem; font-weight: 800; color: #ffffff; font-family: 'Space Grotesk', sans-serif;">{prim['type']}</span>
                <span style="background-color: #10161A; border: 1px solid #28353D; color: #CFFF00; padding: 0.30rem 0.8rem; border-radius: 4px; font-weight: 700; font-size: 0.84rem; font-family: 'JetBrains Mono', monospace;">Suitability Score: {prim['suitability_score']}/100</span>
            </div>
            <p style="color: #FF0A85; font-weight: 600; margin-top: 0.4rem; font-size: 0.90rem;">Category: {prim['category']}</p>
            <p style="color: #8E9DA8; line-height: 1.6; margin-top: 0.8rem; font-size: 0.95rem;">{prim['rationale']}</p>
            <div style="display: flex; gap: 2.5rem; margin-top: 1.2rem; border-top: 1px solid #162025; padding-top: 1rem; flex-wrap: wrap;">
                <div><span style="color: #5A6B75; font-size: 0.78rem; text-transform: uppercase; font-weight: 600;">Estimated CAPEX:</span><br><b style="color: #ffffff; font-size: 1.1rem; font-family: 'Space Grotesk', sans-serif;">{prim['est_budget_cr']}</b></div>
                <div><span style="color: #5A6B75; font-size: 0.78rem; text-transform: uppercase; font-weight: 600;">Target ROI Horizon:</span><br><b style="color: #ffffff; font-size: 1.1rem; font-family: 'Space Grotesk', sans-serif;">{prim['roi_timeline']}</b></div>
                <div><span style="color: #5A6B75; font-size: 0.78rem; text-transform: uppercase; font-weight: 600;">Resilience Impact:</span><br><b style="color: #CFFF00; font-size: 1.05rem; font-family: 'Space Grotesk', sans-serif;">{prim['resilience_impact']}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # DPR Expander
    with st.expander("View Constructor Detailed Project Report (DPR) Brief"):
        # Compact Information Fields
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"<div style='font-size:0.72rem; color:#5A6B75; text-transform:uppercase; font-family:\"JetBrains Mono\", monospace; font-weight:700;'>Site Identifier</div>"
                f"<div style='color:#FFFFFF; font-weight:600; font-size:0.92rem; margin-top:2px; word-break:break-word;'>{selected_parcel['id']} — {selected_parcel['name']}</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div style='font-size:0.72rem; color:#5A6B75; text-transform:uppercase; font-family:\"JetBrains Mono\", monospace; font-weight:700;'>Coordinates</div>"
                f"<div style='color:#FFFFFF; font-weight:600; font-size:0.92rem; margin-top:2px;'>{selected_parcel['lat']}° N, {selected_parcel['lon']}° E</div>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"<div style='font-size:0.72rem; color:#5A6B75; text-transform:uppercase; font-family:\"JetBrains Mono\", monospace; font-weight:700;'>Total Footprint</div>"
                f"<div style='color:#CFFF00; font-weight:600; font-size:0.92rem; margin-top:2px;'>{selected_parcel['area_acres']:.1f} Acres <span style='color:#8E9DA8; font-size:0.78rem;'>(~{selected_parcel['area_acres']*0.4047:.2f} Ha)</span></div>",
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"<div style='font-size:0.72rem; color:#5A6B75; text-transform:uppercase; font-family:\"JetBrains Mono\", monospace; font-weight:700;'>Target Plan</div>"
                f"<div style='color:#FF0A85; font-weight:700; font-size:0.92rem; margin-top:2px; word-break:break-word;'>{prim['type']}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 1px; background: #162025; margin: 1rem 0;'></div>", unsafe_allow_html=True)

        # 1. Executive Site Assessment
        st.markdown(
            f"""
            <div style="margin-bottom: 1.1rem;">
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.98rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem;">
                    1. Executive Site Assessment
                </div>
                <ul style="color: #B5C2CB; font-size: 0.90rem; line-height: 1.7; margin-top: 0; padding-left: 1.25rem;">
                    <li><b>Nearest Arterial Junction:</b> <span style="color: #FFFFFF;">{eval_result['ingress_point']['distance_m']:.0f} meters</span> (Direct Ingress Access)</li>
                    <li><b>Distance to Central Medical Hub:</b> <span style="color: #FFFFFF;">{eval_result['dist_hospital_km']:.2f} km</span> (Victoria Hospital Hub)</li>
                    <li><b>Nearest Strategic Landmark:</b> <span style="color: #FFFFFF;">{eval_result['nearest_landmark']}</span> ({eval_result['nearest_landmark_dist_km']:.2f} km)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Constructor Rationale & Feasibility
        st.markdown(
            f"""
            <div style="margin-bottom: 1.1rem;">
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.98rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem;">
                    2. Constructor Rationale & Feasibility
                </div>
                <div style="color: #8E9DA8; font-size: 0.90rem; line-height: 1.65; background: #080C0E; border: 1px solid #162025; border-left: 3px solid #FF0A85; border-radius: 4px; padding: 0.85rem 1rem;">
                    {prim['rationale']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3. Financial & Operational Overview
        st.markdown(
            f"""
            <div style="margin-bottom: 0.8rem;">
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.98rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem;">
                    3. Financial & Operational Overview
                </div>
                <ul style="color: #B5C2CB; font-size: 0.90rem; line-height: 1.7; margin-top: 0; padding-left: 1.25rem;">
                    <li><b>Estimated Capital Expenditure (CAPEX):</b> <span style="color: #FFFFFF; font-weight: 600;">{prim['est_budget_cr']}</span></li>
                    <li><b>Projected Return on Investment (ROI):</b> <span style="color: #FFFFFF; font-weight: 600;">{prim['roi_timeline']}</span></li>
                    <li><b>Urban Resilience Contribution:</b> <span style="color: #CFFF00; font-weight: 600;">{prim['resilience_impact']}</span></li>
                </ul>
            </div>
            <div style="font-size: 0.75rem; color: #445058; font-family: 'JetBrains Mono', monospace; margin-top: 1rem; border-top: 1px solid #141A1E; padding-top: 0.6rem;">
                Report generated by Route Resilience Platform • Detailed Project Report Brief
            </div>
            """,
            unsafe_allow_html=True,
        )


# ╔══════════════════════════════════════════════════════════════╗
# ║                        FOOTER                                ║
# ╚══════════════════════════════════════════════════════════════╝

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #445058; padding: 1rem 1rem 2rem 1rem; font-size: 0.80rem; font-family: 'Inter', sans-serif;">
        <span style="color: #FF0A85; font-weight: 700; font-family: 'Space Grotesk', sans-serif;">Route Resilience</span>
        <span style="color: #1F282D; margin: 0 8px;">•</span>
        <span>Road Network Criticality & Connectivity Platform</span>
        <span style="color: #1F282D; margin: 0 8px;">•</span>
        <span style="color: #5A6B75;">Bengaluru AOI • OpenStreetMap</span>
    </div>
    """,
    unsafe_allow_html=True,
)
