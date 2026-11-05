import os
from pathlib import Path

ROOT = Path(os.environ.get("SAER_APP_ROOT", Path(__file__).resolve().parents[2])).resolve()
APP_NAME = "سعّر"
RUNTIME_CACHE_DIR = Path(os.environ.get("SAER_RUNTIME_CACHE", Path.home() / ".cache" / "saer-runtime"))
DB_DIR = RUNTIME_CACHE_DIR / "backend"
DB_PATH = DB_DIR / "saer.sqlite"
LISTINGS_CSV = ROOT / "data" / "processed" / "listings.csv"
IMAGES_CSV = ROOT / "data" / "processed" / "images.csv"
UPLOAD_DIR = RUNTIME_CACHE_DIR / "uploads"
IDENTITY_CACHE_DIR = DB_DIR / "cache" / "identity"
IMAGE_PRICE_CACHE_DIR = DB_DIR / "cache" / "image_price"
IMAGE_PRICE_RUNTIME = ROOT / "models" / "image_price_hamoud_20261028"
IMAGE_PRICE_POINTER = ROOT / "models" / "image_price_current.json"
IDENTITY_POINTER = ROOT / "models" / "identity_current.json"
TABULAR_POINTER = ROOT / "models" / "price_current.json"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
