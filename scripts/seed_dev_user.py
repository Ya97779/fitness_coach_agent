"""Create the isolated local user used by localhost-only development auth."""

import os
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.app import models
from backend.app.database import Base, SessionLocal, engine


def main() -> None:
    if os.getenv("FITNESS_COACH_ALLOW_DEV_SEED") != "1":
        raise RuntimeError("Refusing to seed without FITNESS_COACH_ALLOW_DEV_SEED=1")
    if not str(engine.url).startswith("sqlite"):
        raise RuntimeError("Refusing to seed: development database must be SQLite")

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = db.get(models.User, 1)
        if user is None:
            user = models.User(
                id=1,
                nickname="Local Dev",
                height=175,
                weight=70,
                age=30,
                gender="男",
                goal="维持健康",
            )
            db.add(user)
            db.commit()
            print("[dev] Created isolated local user id=1")
        else:
            print("[dev] Local user id=1 already exists")
    finally:
        db.close()


if __name__ == "__main__":
    main()
