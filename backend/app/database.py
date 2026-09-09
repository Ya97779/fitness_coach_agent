import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL 未配置，请在 .env 中设置 PostgreSQL 连接字符串")

engine_options = {
    "pool_pre_ping": True,
    "pool_recycle": 3600,
}
if DATABASE_URL.startswith("sqlite"):
    # 本地开发/测试允许后台线程访问同一个 SQLite 文件；生产 PostgreSQL
    # 仍使用下面的显式连接池参数。
    engine_options["connect_args"] = {"check_same_thread": False}
else:
    # 这些参数按“每个 Gunicorn worker”计算，避免 4 worker × 50 连接
    # 在小型生产实例上把数据库连接数直接打满。
    engine_options.update({
        "pool_size": int(os.getenv("DB_POOL_SIZE", "5")),
        "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "10")),
        "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", "30")),
    })

engine = create_engine(DATABASE_URL, **engine_options)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_or_create_daily_log(db, daily_log_model, user_id: int, log_date):
    """Get a user's day row with a safe retry for concurrent first writes.

    The unique index is installed by ``scripts/migrate_phase02.py``.  The
    fallback query also keeps older databases usable while they are being
    migrated.
    """

    log = db.query(daily_log_model).filter(
        daily_log_model.user_id == user_id,
        daily_log_model.date == log_date,
    ).first()
    if log:
        return log

    log = daily_log_model(user_id=user_id, date=log_date)
    db.add(log)
    try:
        db.flush()
        return log
    except IntegrityError:
        db.rollback()
        return db.query(daily_log_model).filter(
            daily_log_model.user_id == user_id,
            daily_log_model.date == log_date,
        ).first()
