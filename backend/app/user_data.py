"""Transactional deletion of a user's business records and AI memories."""

from datetime import datetime, timedelta

from . import models


class ActiveRequestError(Exception):
    pass


def clear_user_data_records(db, user_id: int) -> dict[str, int]:
    """Delete scoped data atomically; refuse while a fresh request is running."""

    try:
        # Request claims use the same row lock, closing the check/delete race.
        db.query(models.User).filter(
            models.User.id == user_id
        ).with_for_update().one()
        active_request = db.query(models.RequestLedger.id).filter(
            models.RequestLedger.user_id == user_id,
            models.RequestLedger.status == "processing",
            models.RequestLedger.created_at >= datetime.utcnow() - timedelta(minutes=10),
        ).first()
        if active_request:
            raise ActiveRequestError("仍有对话处理中，请稍后再清除数据")

        log_ids = [row.id for row in db.query(models.DailyLog.id).filter(
            models.DailyLog.user_id == user_id
        ).all()]
        counts = {"food": 0, "exercise": 0}
        if log_ids:
            counts["food"] = db.query(models.FoodItem).filter(
                models.FoodItem.log_id.in_(log_ids)
            ).delete(synchronize_session=False)
            counts["exercise"] = db.query(models.ExerciseItem).filter(
                models.ExerciseItem.log_id.in_(log_ids)
            ).delete(synchronize_session=False)
        for name, model in (
            ("daily", models.DailyLog),
            ("conversation", models.ConversationLog),
            ("session", models.ConversationSession),
            ("semantic", models.UserMemory),
            ("request", models.RequestLedger),
        ):
            counts[name] = db.query(model).filter(
                model.user_id == user_id
            ).delete(synchronize_session=False)
        db.commit()
        return counts
    except Exception:
        db.rollback()
        raise
