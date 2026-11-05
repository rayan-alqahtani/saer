import os
from pathlib import Path

from backend.app.branding_compat import database_path

ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "سعّر"
DB_DIR = Path(os.environ.get("SAER_RUNTIME_CACHE", ROOT / "backend" / "instance"))
DB_PATH = database_path(DB_DIR)
UPLOAD_DIR = ROOT / "tmp" / "saer_uploads"
IDENTITY_CACHE_DIR = DB_DIR / "cache" / "identity"
IMAGE_PRICE_CACHE_DIR = DB_DIR / "cache" / "image_price"
IMAGE_PRICE_RUNTIME = ROOT / "models" / "image_price_hamoud_20261028"
IMAGE_PRICE_POINTER = ROOT / "models" / "image_price_current.json"
IDENTITY_POINTER = ROOT / "models" / "identity_current.json"
TABULAR_POINTER = ROOT / "models" / "price_current.json"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
