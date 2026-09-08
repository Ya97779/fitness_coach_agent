"""Apply the additive schema/index changes for roadmap phases 0-2.

Run once before restarting the production systemd service.  It is deliberately
separate from the web worker so four Gunicorn workers do not race to perform
schema changes.  The script never deletes or merges user data; if an existing
table contains duplicates it stops and prints the conflicting keys for manual
resolution.
"""

from pathlib import Path
import sys

from sqlalchemy import text


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "backend"))

from app import database, models  # noqa: E402


def _duplicate_keys(connection, query):
    return connection.execute(text(query)).fetchall()


def main() -> None:
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

        food_duplicates = _duplicate_keys(
            connection,
            """
            SELECT name, COUNT(*) AS duplicate_count
            FROM food_calorie_cache
            GROUP BY name
            HAVING COUNT(*) > 1
            """,
        )
        if food_duplicates:
            raise RuntimeError(
                "food_calorie_cache 存在重复 name，请先人工合并后再迁移: "
                + repr(food_duplicates[:20])
            )

        connection.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_daily_log_user_date "
            "ON daily_logs (user_id, date)"
        ))
        connection.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_food_calorie_cache_name "
            "ON food_calorie_cache (name)"
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

    print("phase0_2_memory_and_idempotency migration applied")


if __name__ == "__main__":
    main()
