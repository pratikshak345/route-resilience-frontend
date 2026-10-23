# Dashboard Guide

This guide explains how to run the Streamlit dashboard in demo mode, switch to real-data mode, and exercise the stress-test tools.

## Start the Dashboard

From the project root:

```bash
python -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Open the local Streamlit URL shown in the terminal.

## Demo Mode

Demo Mode is enabled by default in the dashboard sidebar. It uses synthetic masks and a synthetic Bengaluru-centered road graph, so it works before model weights, satellite imagery, or prediction outputs are available.

Use Demo Mode for:
- verifying the dashboard installation
- presenting the end-to-end workflow without large data files
- testing criticality and stress-test interactions quickly

## Real-Data Mode

To use real predictions, generate dashboard artifacts first:

```bash
python -m segmentation.prepare_dashboard_data --model_path data/models/unet_best.pth --input_image data/raw/bengaluru.tif --model_type unet
```

Then launch the dashboard and turn off Demo Mode in the sidebar.

The dashboard looks for these files in `data/predictions/`:

| Artifact | Required | Purpose |
|---|---:|---|
| `pred_mask.npy` or `pred_mask.(png/tif/tiff)` | Yes | Raw segmentation mask |
| `healed_mask.npy` or `healed_mask.(png/tif/tiff)` | Yes | Cleaned/healed road mask |
| `eval_metrics.npy` | No | Real metrics report for the Metrics tab |

The one-shot prep command writes both `.png` and `.npy` mask outputs. The `.npy` files are preferred by the dashboard because they preserve the binary mask values directly.

## Manual Real-Data Flow

If you want to run each stage manually:

```bash
python -m segmentation.infer --model_path data/models/unet_best.pth --input_image data/raw/bengaluru.tif --output_image data/predictions/pred_mask.tif --model_type unet
python -m graph_healing.skeletonize --mask data/predictions/pred_mask.tif --output data/graphs/skeleton.png --output-healed-mask data/predictions/healed_mask.png
```

After these commands complete, start the dashboard and disable Demo Mode.

## Stress Tests

Run the basic failure simulator on a demo graph:

```bash
python -m stress_test.simulate_failure --demo
```

Run one named scenario:

```bash
python -m stress_test.scenario_presets --demo --scenario "KR Puram"
```

Run all preset scenarios against a GraphML output:

```bash
python -m stress_test.scenario_presets --graph data/graphs/healed_graph.graphml
```

Available preset names include:
- `Flood at ORR-Sarjapur Junction`
- `Bridge Collapse at KR Puram`
- `Landslide blocks Hebbal Flyover approach`

## Troubleshooting

- If real mode shows missing-file warnings, confirm `pred_mask.*` and `healed_mask.*` exist in `data/predictions/`.
- If module imports fail, run commands from the project root and prefer `python -m ...` module invocations.
- If the Metrics tab has no real report, keep Demo Mode enabled or generate `data/predictions/eval_metrics.npy`.
- If GPU is unavailable, inference falls back to CPU and may take longer.