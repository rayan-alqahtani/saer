from __future__ import annotations

from pathlib import Path
import shutil

from PIL import Image, ImageOps
from rembg import new_session, remove


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads" / "الصور"
OUT = ROOT / "outputs" / "road-cars-cleaning" / "out"
ASSET_DIR = ROOT / "frontend" / "public" / "assets" / "road-cars"

SOURCES = [
    DOWNLOADS / "car1.jpg",
    DOWNLOADS / "car2.jpg",
    DOWNLOADS / "car 3.jpg",
    DOWNLOADS / "car4.jpg",
    DOWNLOADS / "car 5.jpg",
]


def remove_background(image: Image.Image, session) -> Image.Image:
    cutout = remove(image.convert("RGB"), session=session, post_process_mask=True)
    if not isinstance(cutout, Image.Image):
        cutout = Image.open(cutout)
    # Keep the source RGB byte-for-byte; use the model output for alpha only.
    alpha = cutout.convert("RGBA").getchannel("A")
    return Image.merge("RGBA", (*image.convert("RGB").split(), alpha))


def make_preview(images: list[Image.Image], path: Path) -> None:
    height = 360
    thumbs = [
        image.resize((max(1, round(image.width * height / image.height)), height), Image.Resampling.LANCZOS)
        for image in images
    ]
    width = sum(image.width for image in thumbs) + 24 * (len(thumbs) + 1)
    sheet = Image.new("RGBA", (width, height + 48), (15, 25, 45, 255))
    for index, image in enumerate(thumbs, start=1):
        x = 24 + sum(item.width + 24 for item in thumbs[: index - 1])
        sheet.alpha_composite(image, (x, 24))
    sheet.convert("RGB").save(path, "PNG")


def main() -> None:
    missing = [path for path in SOURCES if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source files: " + ", ".join(str(path) for path in missing))
    allowed_outputs = {f"car{i}.png" for i in range(1, 6)} | {"preview.png", "car4_original.png"}
    if OUT.exists() and any(path.name not in allowed_outputs for path in OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output files: {OUT}")
    allowed_assets = {f"car{i}.png" for i in range(1, 6)} | {"car4_original.png"}
    if ASSET_DIR.exists() and any(path.name not in allowed_assets for path in ASSET_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing public assets: {ASSET_DIR}")

    OUT.mkdir(parents=True, exist_ok=True)
    session = new_session("isnet-general-use")
    results: list[Image.Image] = []

    for index, source in enumerate(SOURCES, start=1):
        with Image.open(source) as opened:
            original = opened.convert("RGB")
        cutout = remove_background(original, session)

        # Crop to the non-transparent result; keep every source pixel's RGB values.
        bounds = cutout.getchannel("A").getbbox()
        if bounds is None:
            raise RuntimeError(f"No foreground detected in {source.name}")
        cutout = cutout.crop(bounds)
        if index == 4:
            cutout.save(OUT / "car4_original.png", "PNG")
            cutout = ImageOps.mirror(cutout)
        target = OUT / f"car{index}.png"
        cutout.save(target, "PNG")
        results.append(cutout)

        alpha = cutout.getchannel("A")
        transparent = sum(value == 0 for value in alpha.get_flattened_data()) / (cutout.width * cutout.height)
        print(f"car{index}.png | {cutout.width}x{cutout.height} | transparent: {transparent:.1%}")

    make_preview(results, OUT / "preview.png")
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for index in range(1, len(results) + 1):
        shutil.copy2(OUT / f"car{index}.png", ASSET_DIR / f"car{index}.png")
    shutil.copy2(OUT / "car4_original.png", ASSET_DIR / "car4_original.png")
    print(f"Preview: {OUT / 'preview.png'}")
    print(f"Assets: {ASSET_DIR}")


if __name__ == "__main__":
    main()
