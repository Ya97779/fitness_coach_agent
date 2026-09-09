"""Idempotently import common-food reference calories into the active cache."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "backend"))

from app.food_cache import seed_common_food_calorie_cache_database  # noqa: E402


def main() -> None:
    seeded_count = seed_common_food_calorie_cache_database()
    print(f"common food calorie cache seeded: {seeded_count} rows")


if __name__ == "__main__":
    main()
