"""Durable request deduplication across Gunicorn workers.

An in-progress or failed request is never automatically re-executed: its tools
may already have committed a side effect. Clients should use a new request ID
only after checking the corresponding business records.
"""

import hashlib
import json
from datetime import datetime

from sqlalchemy.exc import IntegrityError

from . import database, models


class RequestNotExecutable(Exception):
    """A request ID has already been used or is still executing."""

    def __init__(self, status: str):
        self.status = status
        super().__init__(request_status_message(status))


def request_status_message(status: str) -> str:
    if status == "processing":
        return "该请求正在处理中，请稍后查看结果，勿重复提交。"
    if status == "failed":
        return "该请求未能完成；请先核对饮食或运动记录，再发起新请求。"
    return "该请求编号已用于其他内容，请使用新的请求编号。"


def begin_request(user_id: int, request_id: str, session_id: str, message: str):
    """Atomically claim a request ID; return (status, completed result)."""

    if not request_id:
        return "claimed", None
    request_id = request_id[:128]
    session_id = (session_id or f"default_{user_id}")[:128]
    digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
    db = database.SessionLocal()
    try:
        # Serialize claims against a user's destructive clear-data operation.
        db.query(models.User).filter(models.User.id == user_id).with_for_update().one()
        db.add(models.RequestLedger(
            user_id=user_id,
            request_id=request_id,
            session_id=session_id,
            message_hash=digest,
            status="processing",
        ))
        try:
            db.commit()
            return "claimed", None
        except IntegrityError:
            db.rollback()
            row = db.query(models.RequestLedger).filter_by(
                user_id=user_id, request_id=request_id
            ).one()
            if row.session_id != session_id or row.message_hash != digest:
                return "conflict", None
            if row.status == "completed" and row.result_json:
                return "completed", json.loads(row.result_json)
            return row.status, None
    finally:
        db.close()


def complete_request(user_id: int, request_id: str, result) -> None:
    """Store the entire result, never a truncated session-memory excerpt."""

    if not request_id:
        return
    db = database.SessionLocal()
    try:
        row = db.query(models.RequestLedger).filter_by(
            user_id=user_id, request_id=request_id[:128], status="processing"
        ).one()
        row.result_json = json.dumps(result, ensure_ascii=False)
        row.status = "completed"
        row.updated_at = datetime.utcnow()
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def fail_request(user_id: int, request_id: str) -> None:
    """Keep the request ID reserved when an execution fails or is interrupted."""

    if not request_id:
        return
    db = database.SessionLocal()
    try:
        row = db.query(models.RequestLedger).filter_by(
            user_id=user_id, request_id=request_id[:128], status="processing"
        ).first()
        if row:
            row.status = "failed"
            row.updated_at = datetime.utcnow()
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
