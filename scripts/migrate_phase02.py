"""Apply the schema/index changes for roadmap phases 0-2.

Run once before restarting the production systemd service.  It is deliberately
separate from the web worker so four Gunicorn workers do not race to perform
schema changes.  User logs are never merged automatically.  The old food cache
is archived and replaced because its rows mix incompatible calorie bases.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys

from sqlalchemy import inspect, text


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "backend"))

from app import database, models  # noqa: E402
from app.food_cache import (  # noqa: E402
    load_common_food_calorie_seeds,
    seed_common_food_calorie_cache_database,
)


def _duplicate_keys(connection, query):
    return connection.execute(text(query)).fetchall()


def _next_backup_table_name(connection) -> str:
    """Choose a collision-free, internally generated archive table name."""

    inspector = inspect(connection)
    base = "food_calorie_cache_legacy_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    candidate = base
    suffix = 1
    while inspector.has_table(candidate):
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def _replace_legacy_food_cache(connection):
    """Archive an old-schema cache table and create the normalized v2 table."""

    inspector = inspect(connection)
    if not inspector.has_table("food_calorie_cache"):
        models.FoodCalorieCache.__table__.create(bind=connection)
        return None

    columns = {
        column["name"]
        for column in inspector.get_columns("food_calorie_cache")
    }
    required_columns = {"normalized_name", "basis_type", "updated_at"}
    if required_columns.issubset(columns):
        return None

    backup_name = _next_backup_table_name(connection)
    quote = connection.dialect.identifier_preparer.quote
    connection.execute(text(
        f"CREATE TABLE {quote(backup_name)} AS "
        f"SELECT * FROM {quote('food_calorie_cache')}"
    ))
    connection.execute(text(f"DROP TABLE {quote('food_calorie_cache')}"))
    models.FoodCalorieCache.__table__.create(bind=connection)
    return backup_name


def _ensure_food_item_estimation_columns(connection) -> None:
    """Add persistent estimation state to existing food_items tables."""

    columns = {
        column["name"]
        for column in inspect(connection).get_columns("food_items")
    }
    status_added = "calorie_status" not in columns
    if status_added:
        connection.execute(text(
            "ALTER TABLE food_items ADD COLUMN calorie_status "
            "VARCHAR(16) NOT NULL DEFAULT 'ready'"
        ))
    if "calorie_error" not in columns:
        connection.execute(text(
            "ALTER TABLE food_items ADD COLUMN calorie_error VARCHAR"
        ))
    if "calorie_status_updated_at" not in columns:
        timestamp_default = (
            "" if connection.dialect.name == "sqlite"
            else " DEFAULT CURRENT_TIMESTAMP"
        )
        connection.execute(text(
            "ALTER TABLE food_items ADD COLUMN calorie_status_updated_at "
            f"TIMESTAMP{timestamp_default}"
        ))

    if status_added:
        connection.execute(text(
            "UPDATE food_items SET calorie_status = "
            "CASE WHEN calories > 0 THEN 'ready' ELSE 'failed' END"
        ))
    connection.execute(text(
        "UPDATE food_items SET calorie_error = '历史估算未完成' "
        "WHERE calorie_status = 'failed' AND calorie_error IS NULL"
    ))
    connection.execute(text(
        "UPDATE food_items SET calorie_status_updated_at = CURRENT_TIMESTAMP "
        "WHERE calorie_status_updated_at IS NULL"
    ))

    if connection.dialect.name == "postgresql":
        connection.execute(text(
            "ALTER TABLE food_items "
            "DROP CONSTRAINT IF EXISTS ck_food_items_calorie_status"
        ))
        connection.execute(text(
            "ALTER TABLE food_items ADD CONSTRAINT "
            "ck_food_items_calorie_status CHECK "
            "(calorie_status IN ('pending', 'ready', 'failed'))"
        ))


def _upgrade_food_cache_volume_basis(connection) -> None:
    """Allow per-100ml rows and remove obsolete per-one-ml cache rows."""

    if connection.dialect.name == "postgresql":
        connection.execute(text(
            "ALTER TABLE food_calorie_cache DROP CONSTRAINT IF EXISTS "
            "ck_food_calorie_cache_basis_type"
        ))
        connection.execute(text(
            "ALTER TABLE food_calorie_cache ADD CONSTRAINT "
            "ck_food_calorie_cache_basis_type CHECK "
            "(basis_type IN "
            "('per_100g', 'per_100ml', 'per_unit', 'legacy_unknown'))"
        ))

    connection.execute(text(
        "DELETE FROM food_calorie_cache "
        "WHERE basis_type = 'per_unit' AND portion_unit IN ('ml', 'l')"
    ))


def main() -> None:
    # Validate version-controlled seed data before making schema changes.
    load_common_food_calorie_seeds()

    # Creates only missing tables, including ConversationSession/UserMemory.
    models.Base.metadata.create_all(bind=database.engine)

    with database.engine.begin() as connection:
        connection.execute(text(
            """
            CREATE TABLE IF NOT EXISTS fitcoach_schema_migrations (
                version VARCHAR(128) PRIMARY KEY,
                applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        ))

        daily_duplicates = _duplicate_keys(
            connection,
            """
            SELECT user_id, date, COUNT(*) AS duplicate_count
            FROM daily_logs
            GROUP BY user_id, date
            HAVING COUNT(*) > 1
            """,
        )
        if daily_duplicates:
            raise RuntimeError(
                "daily_logs 存在重复 (user_id, date)，请先人工合并后再迁移: "
                + repr(daily_duplicates[:20])
            )

        food_cache_backup = _replace_legacy_food_cache(connection)
        _ensure_food_item_estimation_columns(connection)
        _upgrade_food_cache_volume_basis(connection)

        connection.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_daily_log_user_date "
            "ON daily_logs (user_id, date)"
        ))
        if connection.dialect.name == "postgresql":
            connection.execute(text(
                "ALTER TABLE food_calorie_cache "
                "DROP CONSTRAINT IF EXISTS uq_food_calorie_cache_name"
            ))
        connection.execute(text(
            "DROP INDEX IF EXISTS uq_food_calorie_cache_name"
        ))
        connection.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_food_calorie_cache_basis "
            "ON food_calorie_cache "
            "(normalized_name, basis_type, portion_unit) "
            "WHERE basis_type <> 'legacy_unknown'"
        ))
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_conversation_logs_user_session_time "
            "ON conversation_logs (user_id, session_id, created_at)"
        ))
        connection.execute(text(
            "INSERT INTO fitcoach_schema_migrations(version) VALUES "
            "('phase0_2_memory_and_idempotency') "
            "ON CONFLICT (version) DO NOTHING"
        ))
        connection.execute(text(
            "INSERT INTO fitcoach_schema_migrations(version) VALUES "
            "('phase0_2_food_cache_basis_v2') "
            "ON CONFLICT (version) DO NOTHING"
        ))
        connection.execute(text(
            "INSERT INTO fitcoach_schema_migrations(version) VALUES "
            "('food_estimation_status_and_volume_v3') "
            "ON CONFLICT (version) DO NOTHING"
        ))

    seeded_count = seed_common_food_calorie_cache_database()

    if food_cache_backup:
        print(f"legacy food cache archived as {food_cache_backup}")
    print(f"common food calorie cache seeded: {seeded_count} rows")
    print("phase0_2_memory_and_idempotency migration applied")


if __name__ == "__main__":
    main()
