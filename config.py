"""
Route Resilience — Shared Configuration
All paths, constants, and hyperparameters in one place.
"""
import os
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TILES_DIR = DATA_DIR / "tiles"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"
GRAPHS_DIR = DATA_DIR / "graphs"
DEBUG_DIR = DATA_DIR / "debug"

# Create directories
for d in [RAW_DIR, PROCESSED_DIR, TILES_DIR, MODELS_DIR,
          PREDICTIONS_DIR, GRAPHS_DIR, DEBUG_DIR,
          TILES_DIR / "train" / "images", TILES_DIR / "train" / "masks",
          TILES_DIR / "val" / "images", TILES_DIR / "val" / "masks",
          TILES_DIR / "test" / "images", TILES_DIR / "test" / "masks"]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# AREA OF INTEREST — Bengaluru, Karnataka, India
# ============================================================
AOI = {
    "name": "Bengaluru",
    "north": 13.05,
    "south": 12.90,
    "east": 77.65,
    "west": 77.50,
    "center_lat": 12.975,
    "center_lon": 77.575,
    "crs": "EPSG:32643",        # UTM Zone 43N (India)
    "crs_wgs84": "EPSG:4326",
}

# Named locations for disaster scenarios (real Bengaluru landmarks)
LANDMARKS = {
    "outer_ring_road_junction": {"lat": 12.9352, "lon": 77.6245, "name": "ORR-Sarjapur Junction"},
    "silk_board_junction":     {"lat": 12.9177, "lon": 77.6238, "name": "Silk Board Junction"},
    "hebbal_flyover":          {"lat": 13.0358, "lon": 77.5970, "name": "Hebbal Flyover"},
    "kr_puram_bridge":         {"lat": 13.0012, "lon": 77.6769, "name": "KR Puram Railway Bridge"},
    "yeshwanthpur_junction":   {"lat": 13.0220, "lon": 77.5500, "name": "Yeshwanthpur Junction"},
    "hospital_reference":      {"lat": 12.9563, "lon": 77.5964, "name": "Victoria Hospital (Ref Hub)"},
}

# ============================================================
# DATA PIPELINE CONFIG
# ============================================================
SENTINEL2_COLLECTION = "sentinel-2-l2a"
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
MAX_CLOUD_COVER = 15  # percent

# DeepGlobe dataset
DEEPGLOBE_IMG_SIZE = 1024  # original DeepGlobe image size

# Tiling
TILE_SIZE = 512
TILE_OVERLAP = 64
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Road mask rasterization
ROAD_BUFFER_M = 5  # meters — buffer for rasterizing OSM road lines
PIXEL_SIZE_M = 10  # meters — for Sentinel-2 resolution

# OSMnx road filter — major roads only for Sentinel-2 resolution
OSM_ROAD_FILTER = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'

# ============================================================
# SEGMENTATION CONFIG
# ============================================================
# U-Net baseline
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 3
NUM_CLASSES = 1

# Training hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
ENCODER_LR = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Loss weights
DICE_WEIGHT = 0.5
BCE_WEIGHT = 0.5

# Inference
PRED_THRESHOLD = 0.5

# ============================================================
# GRAPH HEALING CONFIG
# ============================================================
# Skeletonization
MIN_OBJECT_SIZE = 50      # remove small objects below this pixel count
HOLE_AREA_THRESHOLD = 100  # fill holes smaller than this

# Gap healing
HEAL_DISTANCE_THRESHOLD = 25    # max pixel distance between endpoints to attempt healing
HEAL_DOT_PRODUCT_THRESHOLD = -0.7  # dot product threshold (close to -1 = anti-parallel)
DIRECTION_VECTOR_PIXELS = 5     # number of terminal pixels to compute direction vector

# ============================================================
# GRAPH ANALYSIS CONFIG
# ============================================================
GATEKEEPER_PERCENTILE = 95  # top N percentile for gatekeeper nodes
CRITICALITY_COLORMAP = "YlOrRd"  # matplotlib colormap for heatmap

# ============================================================
# STRESS TEST CONFIG
# ============================================================
NUM_SAMPLE_PATHS = 200  # number of random node pairs for path length calculation
REFERENCE_HUB = LANDMARKS["hospital_reference"]

# ============================================================
# DASHBOARD CONFIG
# ============================================================
STREAMLIT_PAGE_TITLE = "Route Resilience - Urban Mobility Criticality Platform"
STREAMLIT_PAGE_ICON = None
STREAMLIT_LAYOUT = "wide"
MAP_ZOOM_START = 13
ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
