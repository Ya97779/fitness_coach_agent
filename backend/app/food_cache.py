"""Food calorie cache with explicit measurement semantics.

``FoodItem.calories`` always represents the total calories consumed in one log
entry.  Cache rows store a reusable basis instead: calories per 100g or per one
named unit.  Legacy rows are retained for audit but never used for calculations.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import json
import logging
from pathlib import Path
import re
import unicodedata
from typing import Callable, Optional, Sequence

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import database, models


logger = logging.getLogger(__name__)

PER_100G = "per_100g"
PER_UNIT = "per_unit"
LEGACY_UNKNOWN = "legacy_unknown"
DEFAULT_PORTION_UNIT = "份"
COMMON_FOOD_DATA_SOURCE = "curated_reference_v1"
COMMON_FOOD_DATA_FILE = (
    Path(__file__).resolve().parents[1] / "data" / "common_food_calories.json"
)

_WEIGHT_TO_GRAMS = {
    "g": 1.0,
    "kg": 1000.0,
}

_UNIT_ALIASES = {
    "g": "g",
    "克": "g",
    "公克": "g",
    "kg": "kg",
    "千克": "kg",
    "公斤": "kg",
    "个": "个",
    "只": "个",
    "枚": "个",
    "份": "份",
    "份量": "份",
    "碗": "碗",
    "杯": "杯",
    "片": "片",
    "块": "块",
    "根": "根",
    "袋": "袋",
    "盒": "盒",
    "瓶": "瓶",
    "毫升": "ml",
    "ml": "ml",
}


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _source_priority(source: Optional[str]) -> int:
    value = str(source or "").strip().casefold()
    if "manual" in value or "人工" in value:
        return 40
    if "reference" in value or "参考" in value:
        return 35
    if "api" in value:
        return 30
    if "llm" in value:
        return 20
    if "fallback" in value or "本地" in value:
        return 10
    return 0


@dataclass(frozen=True)
class CalorieBasis:
    normalized_name: str
    basis_type: str
    portion_qty: float
    portion_unit: str
    calories: float


@lru_cache(maxsize=1)
def load_common_food_calorie_seeds() -> Sequence[dict]:
    """Load and validate the version-controlled common-food seed dataset."""

    payload = json.loads(COMMON_FOOD_DATA_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("common food calorie seed file must be a non-empty list")

    validated = []
    seen_keys = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"common food seed row {index} must be an object")
        missing = {
            "name", "calories", "portion_qty", "portion_unit"
        } - set(item)
        if missing:
            raise ValueError(
                f"common food seed row {index} missing fields: {sorted(missing)}"
            )

        basis = build_calorie_basis(
            item["name"],
            item["calories"],
            item["portion_qty"],
            item["portion_unit"],
        )
        key = (
            basis.normalized_name,
            basis.basis_type,
            basis.portion_unit,
        )
        if key in seen_keys:
            raise ValueError(
                f"common food seed row {index} duplicates basis {key!r}"
            )
        seen_keys.add(key)
        validated.append(dict(item))

    return tuple(validated)


def normalize_food_name(name: str) -> str:
    """Return a stable cache key without changing meaningful food wording."""

    normalized = unicodedata.normalize("NFKC", str(name or ""))
    normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
    if not normalized:
        raise ValueError("food name cannot be empty")
    return normalized


def normalize_portion_unit(unit: Optional[str]) -> str:
    normalized = unicodedata.normalize("NFKC", str(unit or ""))
    normalized = re.sub(r"\s+", "", normalized).strip().casefold()
    if not normalized:
        return DEFAULT_PORTION_UNIT
    return _UNIT_ALIASES.get(normalized, normalized)


def _positive_number(value: Optional[float], default: float = 1.0) -> float:
    number = default if value is None else float(value)
    if number <= 0:
        raise ValueError("portion quantity must be greater than zero")
    return number


def build_calorie_basis(
    name: str,
    total_calories: float,
    portion_qty: Optional[float] = None,
    portion_unit: Optional[str] = None,
) -> CalorieBasis:
    """Normalize one total-calorie estimate into a reusable cache basis."""

    calories = float(total_calories)
    if calories <= 0:
        raise ValueError("calories must be greater than zero")

    quantity = _positive_number(portion_qty)
    unit = normalize_portion_unit(portion_unit)
    normalized_name = normalize_food_name(name)

    if unit in _WEIGHT_TO_GRAMS:
        grams = quantity * _WEIGHT_TO_GRAMS[unit]
        return CalorieBasis(
            normalized_name=normalized_name,
            basis_type=PER_100G,
            portion_qty=100.0,
            portion_unit="g",
            calories=calories * 100.0 / grams,
        )

    return CalorieBasis(
        normalized_name=normalized_name,
        basis_type=PER_UNIT,
        portion_qty=1.0,
        portion_unit=unit,
        calories=calories / quantity,
    )


def _requested_basis(
    portion_qty: Optional[float],
    portion_unit: Optional[str],
) -> tuple[str, str, float]:
    quantity = _positive_number(portion_qty)
    unit = normalize_portion_unit(portion_unit)
    if unit in _WEIGHT_TO_GRAMS:
        return PER_100G, "g", quantity * _WEIGHT_TO_GRAMS[unit]
    return PER_UNIT, unit, quantity


def get_cached_total_calories(
    db: Session,
    name: str,
    portion_qty: Optional[float] = None,
    portion_unit: Optional[str] = None,
) -> Optional[float]:
    """Return total calories only when the requested measurement basis matches."""

    try:
        basis_type, unit, requested_quantity = _requested_basis(
            portion_qty, portion_unit
        )
        normalized_name = normalize_food_name(name)
    except (TypeError, ValueError):
        return None

    cached = db.query(models.FoodCalorieCache).filter(
        models.FoodCalorieCache.normalized_name == normalized_name,
        models.FoodCalorieCache.basis_type == basis_type,
        models.FoodCalorieCache.portion_unit == unit,
    ).order_by(
        models.FoodCalorieCache.updated_at.desc(),
        models.FoodCalorieCache.id.desc(),
    ).first()

    if not cached:
        return None
    if basis_type == PER_100G:
        return float(cached.calories) * requested_quantity / 100.0
    return float(cached.calories) * requested_quantity


def get_cached_calorie_reference(db: Session, name: str) -> Optional[dict]:
    """Return one displayable basis, preferring per-100g reference data."""

    try:
        normalized_name = normalize_food_name(name)
    except (TypeError, ValueError):
        return None

    candidates = db.query(models.FoodCalorieCache).filter(
        models.FoodCalorieCache.normalized_name == normalized_name,
        models.FoodCalorieCache.basis_type != LEGACY_UNKNOWN,
    ).all()
    if not candidates:
        return None

    cached = max(
        candidates,
        key=lambda row: (
            row.basis_type == PER_100G,
            _source_priority(row.source),
            row.updated_at or row.created_at or datetime.min,
            row.id or 0,
        ),
    )
    return {
        "food_name": cached.name,
        "calories": float(cached.calories),
        "basis_type": cached.basis_type,
        "portion_qty": float(cached.portion_qty),
        "portion_unit": cached.portion_unit,
        "source": cached.source,
    }


def upsert_food_calorie_basis(
    db: Session,
    name: str,
    total_calories: float,
    portion_qty: Optional[float] = None,
    portion_unit: Optional[str] = None,
    source: str = "llm",
) -> models.FoodCalorieCache:
    """Insert or update one active cache basis inside the caller's transaction."""

    basis = build_calorie_basis(
        name, total_calories, portion_qty, portion_unit
    )
    cached = db.query(models.FoodCalorieCache).filter(
        models.FoodCalorieCache.normalized_name == basis.normalized_name,
        models.FoodCalorieCache.basis_type == basis.basis_type,
        models.FoodCalorieCache.portion_unit == basis.portion_unit,
    ).first()

    if cached is None:
        cached = models.FoodCalorieCache(
            name=str(name).strip(),
            normalized_name=basis.normalized_name,
            basis_type=basis.basis_type,
            portion_qty=basis.portion_qty,
            portion_unit=basis.portion_unit,
            calories=basis.calories,
            source=source or "llm",
        )
        db.add(cached)
    else:
        # Lower-confidence estimates must not overwrite an API/manual basis.
        if _source_priority(source) >= _source_priority(cached.source):
            cached.name = str(name).strip()
            cached.portion_qty = basis.portion_qty
            cached.calories = basis.calories
            cached.source = source or cached.source
            cached.updated_at = _utc_now_naive()

    db.flush()
    return cached


def seed_common_food_calorie_cache(db: Session) -> int:
    """Idempotently import the version-controlled common-food reference data."""

    seeds = load_common_food_calorie_seeds()
    for item in seeds:
        upsert_food_calorie_basis(
            db,
            item["name"],
            item["calories"],
            item["portion_qty"],
            item["portion_unit"],
            source=COMMON_FOOD_DATA_SOURCE,
        )
    return len(seeds)


def seed_common_food_calorie_cache_database(
    session_factory: Optional[Callable[[], Session]] = None,
) -> int:
    """Import common-food reference data in one independent transaction."""

    factory = session_factory or database.SessionLocal
    db = factory()
    try:
        seeded_count = seed_common_food_calorie_cache(db)
        db.commit()
        return seeded_count
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def save_food_calorie_basis(
    name: str,
    total_calories: float,
    portion_qty: Optional[float] = None,
    portion_unit: Optional[str] = None,
    source: str = "llm",
    session_factory: Optional[Callable[[], Session]] = None,
) -> bool:
    """Persist a cache basis independently from the user's food-log transaction."""

    factory = session_factory or database.SessionLocal
    db = factory()
    try:
        try:
            upsert_food_calorie_basis(
                db,
                name,
                total_calories,
                portion_qty,
                portion_unit,
                source,
            )
            db.commit()
            return True
        except IntegrityError:
            # Another worker may have inserted the same basis after our lookup.
            db.rollback()
            upsert_food_calorie_basis(
                db,
                name,
                total_calories,
                portion_qty,
                portion_unit,
                source,
            )
            db.commit()
            return True
    except Exception:
        db.rollback()
        logger.exception("food calorie cache write failed for %r", name)
        return False
    finally:
        db.close()
