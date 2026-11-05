from __future__ import annotations

import time
from typing import Generator

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, create_engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from backend.app.config import DB_DIR, DB_PATH


class Base(DeclarativeBase):
    pass


class Listing(Base):
    __tablename__ = "listings"

    uid: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str | None] = mapped_column(String, nullable=True)
    make: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    model: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    model_std: Mapped[str | None] = mapped_column(String, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    condition: Mapped[str | None] = mapped_column(String, nullable=True)
    price: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    mileage: Mapped[float | None] = mapped_column(Float, nullable=True)
    city: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    transmission: Mapped[str | None] = mapped_column(String, nullable=True)
    fuel: Mapped[str | None] = mapped_column(String, nullable=True)
    trim: Mapped[str | None] = mapped_column(String, nullable=True)
    trim_std: Mapped[str | None] = mapped_column(String, nullable=True)
    body: Mapped[str | None] = mapped_column(String, nullable=True)
    color: Mapped[str | None] = mapped_column(String, nullable=True)
    listing_date: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    usable_price: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    image_count: Mapped[int] = mapped_column(Integer, default=0)


class ImageAsset(Base):
    __tablename__ = "images"

    image_uid: Mapped[str] = mapped_column(String, primary_key=True)
    listing_uid: Mapped[str] = mapped_column(ForeignKey("listings.uid"), index=True)
    relative_path: Mapped[str] = mapped_column(String)
    sha256: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    image_kind: Mapped[str | None] = mapped_column(String, nullable=True)
    valid: Mapped[bool] = mapped_column(Boolean, default=False)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)


class DatasetMeta(Base):
    __tablename__ = "dataset_meta"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(String)


class AuthUser(Base):
    __tablename__ = "auth_users"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String)
    password_hash: Mapped[str] = mapped_column(String)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    token_hash: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("auth_users.user_id"), index=True)
    expires_at: Mapped[float] = mapped_column(Float, index=True)


DB_DIR.mkdir(parents=True, exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def ensure_database() -> dict[str, int | bool]:
    """Prepare local tables without importing external advertisement records."""
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        session.execute(delete(ImageAsset))
        session.execute(delete(Listing))
        session.execute(delete(DatasetMeta))
        session.commit()
    return {"rebuilt": False, "listings": 0, "images": 0}


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
