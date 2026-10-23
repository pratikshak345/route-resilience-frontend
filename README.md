# Route Resilience: Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis

**Bharatiya Antariksh Hackathon 2026 (ISRO) — Problem Statement 4**

## Overview
This platform solves the "spectral blindness" problem in urban satellite imagery (Bengaluru). It features an integrated end-to-end pipeline:
1. **Vertical A (Computer Vision):** A context-aware Deep Learning model (U-Net/SegFormer) trained to "see through" occlusions like dense tree canopies, building shadows, and clouds to extract continuous road networks.
2. **Vertical B (Graph Theory):** Mathematical vector graph-healing logic to close topological gaps, followed by betweenness centrality analysis and simulated stress-testing of the road network to identify urban gatekeeper chokepoints.
3. **Vertical C (AI Urban Site Planner):** Multi-criteria spatial civil engineering advisory evaluating vacant land parcels for strategic infrastructure (Trauma Care Centers, Sponge City flood retention basins, Multi-modal Transit Interchanges) with highway ingress analysis and multi-corridor transit routing to Bengaluru CBD, Majestic, and Victoria Hospital.

## Architecture
```
[Satellite Imagery] -> (Data Pipeline) -> [Occlusion-Augmented Tiles]
                                                  |
                                                  v
                                      [U-Net / SegFormer Model]
                                                  |
                                                  v
                                           [Binary Mask]
                                                  |
                                                  v
                          (Skeletonize & Graph-Heal: Dot Product Logic)
                                                  |
                                                  v
                                         [Connected Graph]
                                                  |
                                                  v
                          (Betweenness Centrality & Stress Test Sim)
                                                  |
                                                  v
                               [Streamlit + Folium Interactive Platform]
```

## Data Sources & Substitutions
- **Primary Inference:** Sentinel-2 L2A (10m resolution) via Planetary Computer. (*Note: Cartosat-3/ResourceSat are restricted, so Sentinel-2 is used for demonstration.*)
- **Training Labels:** DeepGlobe Road Extraction Dataset + OSMnx (OpenStreetMap authentic Bengaluru network).

## Tech Stack
| Component | Technology |
|---|---|
| Deep Learning | PyTorch, `segmentation-models-pytorch`, Albumentations |
| Geospatial Data | `rasterio`, `geopandas`, `osmnx`, `pystac-client` |
| Graph Analysis | `networkx`, `sknw`, `scikit-image` |
| Platform & Visualization | Streamlit, Folium, `streamlit-folium` |
| Civil Engineering AI | Multi-criteria spatial decision matrix (`graph_analysis/urban_planner.py`) |

## Platform Modules

1. 🏠 **Project Overview & Home**: Mission briefing, dual-vertical architecture breakdown, key innovations, and 1-click Feature Surfer Grid.
2. 🗺️ **Raw vs. Healed Mask**: High-contrast satellite raster overlay (Coral Red occluded vs. Emerald Green healed) with real IoU & Dice coefficient scoring.
3. 🔥 **Criticality Heatmap**: 45,000+ node graph topological analysis with betweenness centrality heatmaps and strategic gatekeeper bottleneck identification (Silk Board, KR Puram, Hebbal).
4. ⚡ **Disaster Stress Simulator**: Realistic urban disaster scenarios (Silk Board waterlogging, KR Puram rail flyover collapse, Hebbal corridor disruption) with connectivity retention %, hospital cutoff metrics, detour delays, and Civil Defense Advisories.
5. 🏗️ **AI Urban Site Planner & Vacant Land Advisor**: Multi-criteria spatial intelligence evaluating vacant parcels for critical civil infrastructure with highway ingress analysis, multi-corridor transit routing to CBD/Majestic/Victoria Hospital, and automated DPR generation.

## Setup & Run Instructions

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Launch Platform Immediately
The repository includes precomputed topological graph assets and real road masks under `data/`, allowing instant out-of-the-box execution:
```bash
streamlit run dashboard/app.py
```
Open [http://127.0.0.1:8501](http://127.0.0.1:8501) in your browser.

### 3. Pipeline Execution Phases (Optional / From Scratch)

**Phase 1: Data Pipeline**
```bash
python -m data_pipeline.download_imagery --date_range "2023-01-01/2023-12-31"
python -m data_pipeline.download_labels
python -m data_pipeline.tiling --images_dir data/raw/images --masks_dir data/raw/masks
```

**Phase 2: Segmentation Training & Inference**
```bash
python -m segmentation.train --data_dir data/tiles --epochs 50 --batch_size 8 --lr 1e-4 --model_type unet
python -m segmentation.infer --model_path data/models/unet_best.pth --input_image data/raw/bengaluru.tif --output_image data/predictions/bengaluru_mask.tif
```

**Phase 3: Graph Healing & Analysis**
```bash
python -m graph_healing.skeletonize --mask_path data/predictions/bengaluru_mask.tif
python -m graph_healing.mask_to_graph --skeleton_path data/graphs/skeleton.png
python -m graph_healing.vector_healing --graph_path data/graphs/raw_graph.graphml
python -m graph_analysis.centrality --graph_path data/graphs/healed_graph.graphml
python -m graph_analysis.gatekeeper_detection --graph_path data/graphs/healed_graph.graphml
```

**Phase 4: One-Shot Real Bengaluru Road Network Regeneration**
```bash
python generate_real_dashboard_data.py
```

### Stress-Test CLI Examples
Run a synthetic node-failure demo:
```bash
python -m stress_test.simulate_failure --demo
```

Run a named disaster scenario on a demo graph:
```bash
python -m stress_test.scenario_presets --demo --scenario "KR Puram"
```

Run all preset scenarios on a saved graph:
```bash
python -m stress_test.scenario_presets --graph data/graphs/healed_graph.graphml
```

## Databricks Training
A standalone notebook script is provided at `notebooks/train_on_databricks.py` for cloud execution.

## Evaluation Metrics
| Metric | Description |
|---|---|
| IoU (Intersection over Union) | Pixel-level overlap of predicted vs actual road masks |
| Dice Coefficient / F1 Score | Harmonic mean of precision and recall for road pixels |
| Connectivity Ratio | Fraction of total graph nodes reachable within the largest component |
| Topological Accuracy | Comparison of junction precision/recall and degree distribution |
| Average Path Length Error | Mean absolute error of shortest paths between randomly sampled nodes |

## Known Limitations
- Sentinel-2's 10m resolution cannot resolve small service roads or dense residential lanes.
- Transformer models (SegFormer) require significant compute and are included as a stretch goal.
