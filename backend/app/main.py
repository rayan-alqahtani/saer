from __future__ import annotations

import io
import json
import math
import os
import base64
import hashlib
import hmac
import re
import random
import secrets
from statistics import median
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from backend.app.branding_compat import SESSION_COOKIE, session_token
from backend.app.config import APP_NAME, MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES, ROOT, UPLOAD_DIR
from backend.app.db import AuthSession, AuthUser, ImageAsset, Listing, ensure_database, get_session
from backend.app.schemas import AuthCredentials, ListingCompareRequest, TabularPriceRequest
from backend.app.services import (
    file_sha256,
    predict_identity,
    predict_image_price,
    predict_tabular,
    supported_price_pairs,
)

app = FastAPI(title=APP_NAME, version="1.0.0", description=f"واجهة {APP_NAME} المحلية")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.on_event("startup")
def startup() -> None:
    ensure_database()


def _listing_dict(row: Listing, image: ImageAsset | None = None) -> dict[str, Any]:
    return {
        "listing_id": row.uid,
        "make": row.make,
        "model": row.model_std or row.model,
        "year": row.year,
        "price_sar": row.price,
        "mileage_km": row.mileage,
        "city": row.city,
        "trim": row.trim_std or row.trim,
        "body": row.body,
        "color": row.color,
        "transmission": row.transmission,
        "fuel": row.fuel,
        "condition": row.condition,
        "listing_date": row.listing_date,
        "image_count": row.image_count,
        "image_url": f"/api/images/{image.image_uid}" if image else None,
        "historical": True,
    }


def _first_image(session: Session, uid: str) -> ImageAsset | None:
    return session.scalar(
        select(ImageAsset).where(ImageAsset.listing_uid == uid, ImageAsset.valid.is_(True)).order_by(ImageAsset.image_uid).limit(1)
    )


def _valid_image_path(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    allowed_roots = (
        (ROOT / "data" / "raw").resolve(),
        (ROOT / "frontend" / "public" / "demo" / "img").resolve(),
        (ROOT / "data" / "recovery" / "images").resolve(),
        (ROOT / "data" / "recovery" / "residual_images").resolve(),
    )
    if not any(root == path or root in path.parents for root in allowed_roots) or not path.is_file():
        raise HTTPException(404, "ملف الصورة غير متوفر")
    return path


async def _save_upload(upload: UploadFile) -> tuple[Path, str]:
    filename = (upload.filename or "").lower()
    suffix = Path(filename).suffix
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(415, "ارفع صورة بصيغة JPG أو PNG أو WebP")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="saer_", suffix=suffix, dir=UPLOAD_DIR)
    path = Path(temp_name)
    total = 0
    try:
        with os.fdopen(fd, "wb", closefd=True) as out:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "حجم الصورة يتجاوز 15 ميغابايت")
                out.write(chunk)
        if total == 0:
            raise HTTPException(400, "الملف فارغ")
        try:
            with Image.open(path) as img:
                if img.format not in {"JPEG", "PNG", "WEBP"}:
                    raise HTTPException(415, "صيغة الصورة غير مدعومة")
                width, height = img.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise HTTPException(413, "أبعاد الصورة كبيرة جدًا")
                img.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(400, "تعذر قراءة الصورة؛ اختر صورة سليمة") from exc
        return path, file_sha256(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise


@app.get("/api/health")
def health(session: Session = Depends(get_session)) -> dict[str, Any]:
    return {
        "status": "ok",
        "app": APP_NAME,
        "listings": session.scalar(select(func.count()).select_from(Listing)) or 0,
        "images": session.scalar(select(func.count()).select_from(ImageAsset).where(ImageAsset.valid.is_(True))) or 0,
    }


_SESSION_DAYS = 7


def _password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return "scrypt$%s$%s" % (
        base64.urlsafe_b64encode(salt).decode().rstrip("="),
        base64.urlsafe_b64encode(derived).decode().rstrip("="),
    )


def _password_matches(password: str, saved: str) -> bool:
    try:
        scheme, salt_text, hash_text = saved.split("$", 2)
        if scheme != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_text + "=" * (-len(salt_text) % 4))
        expected = base64.urlsafe_b64decode(hash_text + "=" * (-len(hash_text) % 4))
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _set_session_cookie(response: Response, session: Session, user: AuthUser) -> dict[str, Any]:
    token = secrets.token_urlsafe(32)
    session.add(AuthSession(
        token_hash=hashlib.sha256(token.encode()).hexdigest(),
        user_id=user.user_id,
        expires_at=time.time() + _SESSION_DAYS * 86400,
    ))
    response.set_cookie(SESSION_COOKIE, token, max_age=_SESSION_DAYS * 86400,
                        httponly=True, samesite="lax", secure=False, path="/")
    session.commit()
    return {"authenticated": True, "user": {"display_name": user.display_name, "email": user.email}}


@app.get("/api/auth/me")
def auth_me(request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    token = session_token(request.cookies)
    if not token:
        return {"authenticated": False, "user": None}
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    saved = session.get(AuthSession, token_hash)
    if not saved or saved.expires_at <= time.time():
        if saved:
            session.delete(saved)
            session.commit()
        return {"authenticated": False, "user": None}
    user = session.get(AuthUser, saved.user_id)
    if not user:
        return {"authenticated": False, "user": None}
    return {"authenticated": True, "user": {"display_name": user.display_name, "email": user.email}}


@app.post("/api/auth/register")
def auth_register(payload: AuthCredentials, response: Response, session: Session = Depends(get_session)) -> dict[str, Any]:
    email = payload.email.strip().lower()
    name = payload.display_name.strip()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        raise HTTPException(422, "أدخل بريدًا إلكترونيًا صحيحًا")
    if not name:
        raise HTTPException(422, "أدخل الاسم")
    if session.scalar(select(AuthUser).where(AuthUser.email == email)):
        raise HTTPException(409, "هذا البريد مسجل مسبقًا")
    user = AuthUser(user_id=uuid.uuid4().hex, email=email, display_name=name, password_hash=_password_hash(payload.password))
    session.add(user)
    session.flush()
    return _set_session_cookie(response, session, user)


@app.post("/api/auth/login")
def auth_login(payload: AuthCredentials, response: Response, session: Session = Depends(get_session)) -> dict[str, Any]:
    email = payload.email.strip().lower()
    user = session.scalar(select(AuthUser).where(AuthUser.email == email))
    if not user or not _password_matches(payload.password, user.password_hash):
        raise HTTPException(401, "البريد أو كلمة المرور غير صحيحة")
    return _set_session_cookie(response, session, user)


@app.post("/api/auth/logout")
def auth_logout(request: Request, response: Response, session: Session = Depends(get_session)) -> dict[str, bool]:
    token = session_token(request.cookies)
    if token:
        saved = session.get(AuthSession, hashlib.sha256(token.encode()).hexdigest())
        if saved:
            session.delete(saved)
            session.commit()
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@app.get("/api/meta")
def meta(session: Session = Depends(get_session)) -> dict[str, Any]:
    makes = session.scalars(select(Listing.make).where(Listing.make.is_not(None)).distinct().order_by(Listing.make)).all()
    years = session.scalars(select(Listing.year).where(Listing.year.is_not(None)).distinct().order_by(Listing.year.desc())).all()
    cities = session.scalars(select(Listing.city).where(Listing.city.is_not(None)).distinct().order_by(Listing.city)).all()
    trims = session.scalars(select(Listing.trim_std).where(Listing.trim_std.is_not(None), Listing.trim_std != "").distinct().order_by(Listing.trim_std)).all()
    pairs = session.execute(
        select(Listing.make, Listing.model_std)
        .where(Listing.make.is_not(None), Listing.model_std.is_not(None), Listing.model_std != "")
        .distinct()
        .order_by(Listing.make, Listing.model_std)
    ).all()
    models_by_make: dict[str, list[str]] = {}
    for make, model in pairs:
        models_by_make.setdefault(str(make), []).append(str(model))
    return {
        "app_name": APP_NAME,
        "listings_count": session.scalar(select(func.count()).select_from(Listing)) or 0,
        "images_count": session.scalar(select(func.count()).select_from(ImageAsset).where(ImageAsset.valid.is_(True))) or 0,
        "makes": makes,
        "years": years,
        "cities": cities,
        "trims": trims,
        "models_by_make": models_by_make,
        "notice": "البيانات تاريخية من مجموعة المشروع ولا تعني أن الإعلانات ما زالت متاحة.",
    }


@app.get("/api/brands")
def brands() -> dict[str, Any]:
    path = ROOT / "frontend" / "public" / "brands" / "brands.json"
    if not path.is_file():
        return {"brands": {}}
    return {"brands": json.loads(path.read_text(encoding="utf-8"))}


@app.get("/api/listings")
def listings(
    q: str | None = None,
    make: str | None = None,
    model: str | None = None,
    year: int | None = Query(default=None, ge=1900, le=2027),
    min_price: float | None = Query(default=None, ge=0),
    max_price: float | None = Query(default=None, ge=0),
    city: str | None = None,
    sort_by: str = Query(default="newest", pattern="^(newest|price_asc|price_desc|mileage_asc|year_desc|year_asc)$"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=24, ge=1, le=100),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    filters = []
    if q:
        like = f"%{q.strip()}%"
        filters.append(or_(Listing.make.ilike(like), Listing.model.ilike(like), Listing.model_std.ilike(like), Listing.city.ilike(like), Listing.trim.ilike(like)))
    if make: filters.append(Listing.make == make)
    if model: filters.append(or_(Listing.model == model, Listing.model_std == model))
    if year: filters.append(Listing.year == year)
    if min_price is not None: filters.append(Listing.price >= min_price)
    if max_price is not None: filters.append(Listing.price <= max_price)
    if city: filters.append(Listing.city == city)
    where = and_(*filters) if filters else True
    total = session.scalar(select(func.count()).select_from(Listing).where(where)) or 0
    summary_rows = session.execute(select(Listing.price, Listing.image_count).where(where)).all()
    valid_prices = [float(price) for price, _ in summary_rows if price is not None and math.isfinite(float(price)) and float(price) > 0]
    image_listings = sum(1 for _, image_count in summary_rows if image_count and image_count > 0)
    order_columns = {
        "newest": (Listing.listing_date.desc().nullslast(), Listing.uid),
        "price_asc": (Listing.price.asc().nullslast(), Listing.uid),
        "price_desc": (Listing.price.desc().nullslast(), Listing.uid),
        "mileage_asc": (Listing.mileage.asc().nullslast(), Listing.uid),
        "year_desc": (Listing.year.desc().nullslast(), Listing.uid),
        "year_asc": (Listing.year.asc().nullslast(), Listing.uid),
    }
    rows = session.scalars(select(Listing).where(where).order_by(*order_columns[sort_by]).offset(offset).limit(limit)).all()
    result = []
    for row in rows:
        result.append(_listing_dict(row, _first_image(session, row.uid)))
    return {
        "items": result,
        "total": total,
        "offset": offset,
        "limit": limit,
        "summary": {
            "priced_count": len(valid_prices),
            "median_price": median(valid_prices) if valid_prices else None,
            "average_price": sum(valid_prices) / len(valid_prices) if valid_prices else None,
            "with_images": image_listings,
            "image_coverage_pct": 100 * image_listings / total if total else 0,
        },
    }


@app.get("/api/listings/{listing_id}")
def listing_detail(listing_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    row = session.get(Listing, listing_id)
    if not row: raise HTTPException(404, "الإعلان غير موجود في قاعدة البيانات")
    item = _listing_dict(row)
    images = session.scalars(select(ImageAsset).where(ImageAsset.listing_uid == row.uid, ImageAsset.valid.is_(True)).order_by(ImageAsset.image_uid)).all()
    item["images"] = [{"image_uid": image.image_uid, "url": f"/api/images/{image.image_uid}"} for image in images]
    return item


@app.get("/api/images/{image_uid}")
def image_file(image_uid: str, session: Session = Depends(get_session)) -> FileResponse:
    image = session.get(ImageAsset, image_uid)
    if not image or not image.valid: raise HTTPException(404, "الصورة غير متوفرة")
    path = _valid_image_path(image.relative_path)
    return FileResponse(path, media_type={".webp": "image/webp", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(path.suffix.lower(), "application/octet-stream"), headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/featured")
def featured(session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    pairs = session.execute(
        select(Listing.make, Listing.model_std, func.count().label("n"))
        .where(Listing.price.is_not(None), Listing.make.is_not(None), Listing.model_std.is_not(None))
        .group_by(Listing.make, Listing.model_std)
        .order_by(func.count().desc(), Listing.make, Listing.model_std)
        .limit(30)
    ).all()
    items = []
    used_makes: set[str] = set()
    for make, model, _count in pairs:
        if make in used_makes:
            continue
        row = session.scalar(
            select(Listing)
            .where(Listing.make == make, Listing.model_std == model, Listing.price.is_not(None))
            .order_by(Listing.year.desc(), Listing.uid)
            .limit(1)
        )
        image = _first_image(session, row.uid) if row else None
        if row and image:
            items.append(_listing_dict(row, image))
            used_makes.add(make)
        if len(items) == 6:
            break
    return items


@app.get("/api/pricing-options")
def pricing_options() -> dict[str, Any]:
    try:
        pairs = supported_price_pairs()
    except Exception as exc:
        raise HTTPException(503, f"تعذر تحميل نموذج التسعير الجدولي: {exc}") from exc
    return {"pairs": pairs, "notice": "الخيارات مبنية على نطاق النموذج المدرب؛ التقدير يعتمد أيضًا على السنة والممشى والحقول المتاحة."}


@app.post("/api/recognize")
async def recognize(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    if len((await request.form()).getlist("file")) != 1:
        raise HTTPException(422, "ارفع صورة واحدة فقط للسيارة")
    path, digest = await _save_upload(file)
    try:
        try:
            return predict_identity(path, digest)
        except Exception as exc:
            raise HTTPException(503, f"تعذر تشغيل نموذج التعرف المعتمد: {str(exc)[:500]}") from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/estimate-image-price")
async def estimate_image_price(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    if len((await request.form()).getlist("file")) != 1:
        raise HTTPException(422, "ارفع صورة واحدة فقط للسيارة")
    path, digest = await _save_upload(file)
    try:
        try:
            result = predict_image_price(path, digest)
        except Exception as exc:
            raise HTTPException(503, f"تعذر تشغيل نموذج تقدير الصورة: {str(exc)[:500]}") from exc
        notice = "تقدير تجريبي من صورة واحدة؛ لا يمثل سعر بيع مؤكدًا ولا يرى السنة أو الممشى أو الحالة."
        return {**result, "notice_ar": notice, "notice": notice}
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/estimate-image-price/multiple")
async def estimate_image_price_multiple(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if len(files) != 1:
        raise HTTPException(422, "ارفع صورة واحدة فقط للسيارة")
    paths: list[Path] = []
    try:
        for upload in files:
            path, _ = await _save_upload(upload)
            paths.append(path)
        digest = hashlib.sha256("|".join(file_sha256(path) for path in paths).encode()).hexdigest()
        try:
            result = predict_image_price(paths, digest)
        except Exception as exc:
            raise HTTPException(503, f"تعذر تشغيل نموذج تقدير الصورة: {str(exc)[:500]}") from exc
        notice = f"تقدير تجريبي من {len(paths)} صورة للسيارة نفسها؛ لا يمثل سعر بيع مؤكدًا ولا يرى السنة أو الممشى أو الحالة."
        return {**result, "image_count": len(paths), "notice_ar": notice, "notice": notice}
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


@app.post("/api/estimate-tabular")
def estimate_tabular(payload: TabularPriceRequest) -> dict[str, Any]:
    try:
        result = predict_tabular(payload.model_dump())
    except Exception as exc:
        raise HTTPException(422, f"تعذر التقدير بهذه المدخلات: {str(exc)[:500]}") from exc
    return {**result, "notice": "تقدير لسعر إعلان تقريبي اعتمادًا على المواصفات المدخلة وبيانات التدريب، وليس سعر بيع مضمونًا."}


@app.get("/api/recommendations")
def recommendations(
    min_price: float = Query(default=0, ge=0),
    max_price: float = Query(default=1_000_000_000, ge=0),
    make: str | None = None,
    model: str | None = None,
    year: int | None = Query(default=None, ge=1900, le=2027),
    city: str | None = None,
    exclude: str = "",
    exclude_models: str = "",
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    if min_price > max_price: raise HTTPException(422, "الحد الأدنى يجب ألا يتجاوز الحد الأعلى")
    filters = [Listing.usable_price.is_(True), Listing.price.between(min_price, max_price)]
    if make: filters.append(Listing.make == make)
    if model: filters.append(or_(Listing.model_std == model, Listing.model == model))
    if year: filters.append(Listing.year == year)
    if city: filters.append(Listing.city == city)
    excluded_ids = {x for x in exclude.split(",") if x}
    excluded_models_set = {x.strip() for x in exclude_models.split(",") if x.strip()}
    rows = session.scalars(select(Listing).where(and_(*filters)).order_by(Listing.uid)).all()
    grouped: dict[tuple[str, str], list[Listing]] = {}
    for row in rows:
        pair = (str(row.make or ""), str(row.model_std or row.model or ""))
        if row.uid in excluded_ids or f"{pair[0]}|{pair[1]}" in excluded_models_set:
            continue
        grouped.setdefault(pair, []).append(row)
    rng = random.SystemRandom()
    candidates = [rng.choice(pair_rows) for pair_rows in grouped.values()]
    rng.shuffle(candidates)
    chosen = candidates[:5]
    return {
        "items": [_listing_dict(row, _first_image(session, row.uid)) for row in chosen],
        "count": len(chosen),
        "eligible_listing_count": sum(len(pair_rows) for pair_rows in grouped.values()),
        "eligible_model_count": len(grouped),
        "notice": "اختيرت إعلانات عشوائية من سيارات مطابقة للميزانية والفلاتر، دون تكرار الموديل ضمن الدفعة. هذه بيانات تاريخية وليست ترتيب جودة أو توفر حالي.",
    }


@app.post("/api/compare")
def compare(payload: ListingCompareRequest, session: Session = Depends(get_session)) -> dict[str, Any]:
    if len(set(payload.listing_ids)) != 2: raise HTTPException(422, "اختر إعلانين مختلفين")
    rows = [session.get(Listing, uid) for uid in payload.listing_ids]
    if any(row is None for row in rows): raise HTTPException(404, "أحد الإعلانين غير موجود")
    items = [_listing_dict(row, _first_image(session, row.uid)) for row in rows if row]
    fields = ["make", "model", "year", "price_sar", "mileage_km", "city", "trim", "body", "color", "transmission", "fuel", "condition"]
    return {"items": items, "fields": [{"key": field, "values": [item.get(field) for item in items]} for field in fields], "notice": "مقارنة بين بيانات الإعلانين المحفوظة؛ الحقول الناقصة لا تُستنتج."}


@app.get("/api/market")
def market(
    make: str | None = None,
    model: str | None = None,
    year: int | None = Query(default=None, ge=1900, le=2027),
    city: str | None = None,
    min_price: float | None = Query(default=None, ge=0),
    max_price: float | None = Query(default=None, ge=0),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    filters = [Listing.usable_price.is_(True), Listing.price.is_not(None), Listing.price > 0]
    if make: filters.append(Listing.make == make)
    if model: filters.append(or_(Listing.model_std == model, Listing.model == model))
    if year: filters.append(Listing.year == year)
    if city: filters.append(Listing.city == city)
    if min_price is not None: filters.append(Listing.price >= min_price)
    if max_price is not None: filters.append(Listing.price <= max_price)
    if min_price is not None and max_price is not None and min_price > max_price:
        raise HTTPException(422, "الحد الأدنى للسعر لا يتجاوز الحد الأعلى")
    where = and_(*filters)
    rows = session.execute(select(Listing.make, Listing.model_std, Listing.price, Listing.year, Listing.city, Listing.listing_date, Listing.mileage).where(where)).all()
    prices = [float(row.price) for row in rows if row.price is not None and math.isfinite(float(row.price))]
    if prices:
        prices_sorted = sorted(prices)
        med = median(prices_sorted)
        avg = sum(prices) / len(prices)
    else: med = avg = None
    by_make: dict[str, dict[str, Any]] = {}
    by_model: dict[tuple[str, str], dict[str, Any]] = {}
    by_vehicle_year: dict[int, list[float]] = {}
    by_listing_year: dict[int, list[float]] = {}
    by_mileage_band: dict[str, list[float]] = {}
    by_city: dict[str, dict[str, Any]] = {}
    mile_bands = [{"label": "أقل من 50 ألف كم", "min": 0, "max": 50000}, {"label": "50–100 ألف كم", "min": 50000, "max": 100000}, {"label": "100–150 ألف كم", "min": 100000, "max": 150000}, {"label": "150 ألف كم فأكثر", "min": 150000, "max": None}]
    bands = [{"label": "أقل من 40 ألف", "min": 0, "max": 40000}, {"label": "40–70 ألف", "min": 40000, "max": 70000}, {"label": "70–120 ألف", "min": 70000, "max": 120000}, {"label": "120 ألف فأكثر", "min": 120000, "max": None}]
    for band in bands:
        band["count"] = sum(1 for price in prices if price >= band["min"] and (band["max"] is None or price < band["max"]))
    for row in rows:
        if row.make:
            item = by_make.setdefault(row.make, {"make": row.make, "count": 0, "prices": []})
            item["count"] += 1
            if row.price is not None: item["prices"].append(float(row.price))
        if row.city:
            city_item = by_city.setdefault(row.city, {"city": row.city, "count": 0, "prices": []})
            city_item["count"] += 1
            if row.price is not None: city_item["prices"].append(float(row.price))
        if row.make and row.model_std:
            model_item = by_model.setdefault((row.make, row.model_std), {"make": row.make, "model": row.model_std, "count": 0, "prices": []})
            model_item["count"] += 1
            if row.price is not None: model_item["prices"].append(float(row.price))
        if row.year is not None and row.price is not None:
            by_vehicle_year.setdefault(int(row.year), []).append(float(row.price))
        if row.mileage is not None and row.price is not None:
            km = float(row.mileage)
            band = "أقل من 50 ألف كم" if km < 50000 else "50–100 ألف كم" if km < 100000 else "100–150 ألف كم" if km < 150000 else "150 ألف كم فأكثر"
            by_mileage_band.setdefault(band, []).append(float(row.price))
        listed_year = str(row.listing_date or "")[:4]
        if listed_year.isdigit() and row.price is not None:
            by_listing_year.setdefault(int(listed_year), []).append(float(row.price))
    makers = [{"make": item["make"], "count": item["count"], "median_price": median(item["prices"]) if item["prices"] else None} for item in sorted(by_make.values(), key=lambda x: (-x["count"], x["make"]))]
    cities = [{"city": item["city"], "count": item["count"], "median_price": median(item["prices"]) if item["prices"] else None} for item in sorted(by_city.values(), key=lambda x: (-x["count"], x["city"]))]
    models = [{"make": item["make"], "model": item["model"], "count": item["count"], "median_price": median(item["prices"]) if item["prices"] else None} for item in sorted(by_model.values(), key=lambda x: (-x["count"], x["make"], x["model"]))[:10]]
    year_groups = [{"year": year_value, "count": len(year_prices), "median_price": median(year_prices)} for year_value, year_prices in sorted(by_vehicle_year.items())]
    listing_year_groups = [{"year": year_value, "count": len(year_prices), "median_price": median(year_prices)} for year_value, year_prices in sorted(by_listing_year.items())]
    mileage_groups = [{"label": band["label"], "count": len(by_mileage_band.get(band["label"], [])), "median_price": median(by_mileage_band[band["label"]]) if by_mileage_band.get(band["label"]) else None} for band in mile_bands]
    mileages = [float(row.mileage) for row in rows if row.mileage is not None and math.isfinite(float(row.mileage))]
    years = sorted({int(row.year) for row in rows if row.year is not None})
    return {"count": len(prices), "count_with_listing_date": sum(item["count"] for item in listing_year_groups), "median_price": med, "average_price": avg, "min_price": min(prices) if prices else None, "max_price": max(prices) if prices else None, "median_mileage": median(mileages) if mileages else None, "price_bands": bands, "mileage_bands": mileage_groups, "by_make": makers, "by_model": models, "by_city": cities, "by_vehicle_year": year_groups, "by_listing_year": listing_year_groups, "year_range": [min(years), max(years)] if years else None, "notice": "تحليل وصفي لإعلانات المشروع وأسعار الطلب المسجلة، وليس للسوق الحي أو أسعار البيع النهائية. التوزيعات تتأثر بتغطية البيانات وتنوع السيارات."}


if (ROOT / "frontend" / "dist").is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=ROOT / "frontend" / "dist", html=True), name="frontend")

