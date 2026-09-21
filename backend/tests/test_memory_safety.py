"""Regression tests for request deduplication and memory governance."""

import os
import sys
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import models
from app.intent_visibility import visible_assistant_text
from app.memory.memory_manager import MemoryManager
from app import request_ledger
from app.user_data import ActiveRequestError, clear_user_data_records
from scripts.migrate_phase02 import _ensure_user_memory_status


class TestMemorySafety(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        models.Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        db = self.Session()
        db.add_all([models.User(id=1), models.User(id=2)])
        db.commit()
        db.close()
        self.session_patches = [
            patch("app.request_ledger.database.SessionLocal", self.Session),
            patch("app.memory.memory_manager.database.SessionLocal", self.Session),
        ]
        for session_patch in self.session_patches:
            session_patch.start()

    def tearDown(self):
        for session_patch in self.session_patches:
            session_patch.stop()
        self.engine.dispose()

    def test_request_is_claimed_once_and_replays_full_result(self):
        self.assertEqual(
            request_ledger.begin_request(1, "r1", "s1", "记录牛奶"),
            ("claimed", None),
        )
        self.assertEqual(
            request_ledger.begin_request(1, "r1", "s1", "记录牛奶"),
            ("processing", None),
        )
        self.assertEqual(
            request_ledger.begin_request(1, "r1", "s1", "另一个问题"),
            ("conflict", None),
        )
        result = {"response": "完整回答" * 300, "agent": "nutrition"}
        request_ledger.complete_request(1, "r1", result)
        self.assertEqual(
            request_ledger.begin_request(1, "r1", "s1", "记录牛奶"),
            ("completed", result),
        )
        self.assertEqual(
            request_ledger.begin_request(2, "r1", "s1", "记录牛奶"),
            ("claimed", None),
        )

    def test_failed_request_id_is_not_reexecuted(self):
        request_ledger.begin_request(1, "failed", "s1", "记录运动")
        request_ledger.fail_request(1, "failed")
        self.assertEqual(
            request_ledger.begin_request(1, "failed", "s1", "记录运动"),
            ("failed", None),
        )

    def test_nonstream_replay_skips_graph_execution(self):
        from app.agents.graph import process_user_message

        original = {"response": "完整回复" * 200, "agent": "chat"}
        with patch("app.agents.graph._process_user_message", return_value=original) as run:
            first = process_user_message("你好", user_id=1, session_id="s1", request_id="r2")
            second = process_user_message("你好", user_id=1, session_id="s1", request_id="r2")
        self.assertEqual(first, original)
        self.assertEqual(second, original)
        run.assert_called_once()

    def test_stream_replay_returns_full_visible_content_without_rerun(self):
        from app.agents.graph import stream_user_message

        def fake_stream(**_kwargs):
            yield ("data", "正文" * 200)
            yield ("intent", {"type": "food"})

        with patch("app.agents.graph._stream_user_message_impl", side_effect=fake_stream) as run:
            first = list(stream_user_message("记录", user_id=1, session_id="s1", request_id="r3"))
            second = list(stream_user_message("记录", user_id=1, session_id="s1", request_id="r3"))
        self.assertIn(("data", "正文" * 200), first)
        self.assertIn(("data", "正文" * 200), second)
        self.assertIn(("intent", {"type": "food"}), second)
        run.assert_called_once()

    def test_empty_stream_is_failed_not_replayable(self):
        from app.agents.graph import stream_user_message

        with patch("app.agents.graph._stream_user_message_impl", return_value=iter(())):
            events = list(stream_user_message("问题", user_id=1, session_id="s1", request_id="empty"))
        self.assertTrue(any(kind == "error" for kind, _ in events))
        self.assertEqual(
            request_ledger.begin_request(1, "empty", "s1", "问题"),
            ("failed", None),
        )

    def test_only_confirmed_active_memory_enters_prompt(self):
        memory = MemoryManager(user_id=1, session_id="s1")
        self.assertTrue(memory.save_semantic_memory(
            "diet", "候选偏好", source="model", confidence=0.9, confirmed=False,
        ))
        memory.load_all_memory()
        self.assertFalse(memory.get_memory_summary()["semantic_memories"])

        self.assertTrue(memory.save_semantic_memory(
            "diet", "已确认偏好", source="user", confidence=1.0, confirmed=True,
        ))
        memory.load_all_memory()
        self.assertEqual(
            memory.get_memory_summary()["semantic_memories"][0]["value"],
            "已确认偏好",
        )

        db = self.Session()
        db.query(models.UserMemory).filter_by(user_id=1).update({"status": "candidate"})
        db.commit()
        db.close()
        memory.load_all_memory()
        self.assertFalse(memory.get_memory_summary()["semantic_memories"])

    def test_internal_intent_markers_are_hidden_from_history(self):
        self.assertEqual(
            visible_assistant_text("好的\n[INTENT:food]牛奶|breakfast|120"),
            "好的",
        )
        self.assertEqual(
            visible_assistant_text("建议。[INTENT_JSON]{\"type\":\"food\"}[/INTENT_JSON]"),
            "建议。",
        )
        memory = MemoryManager(user_id=1, session_id="s1")
        self.assertTrue(memory.save_conversation(
            "喝了牛奶", "已记录\n[INTENT:food]牛奶|breakfast|120", "chat",
        ))
        self.assertEqual(memory.load_conversation_history(session_id="s1")[-1]["content"], "已记录")

    def test_user_data_clear_removes_all_memory_layers_for_only_one_user(self):
        db = self.Session()
        db.add(models.DailyLog(user_id=1))
        db.add(models.ConversationLog(
            user_id=1, session_id="s1", agent_type="chat",
            user_message="问题", agent_response="答案",
        ))
        db.add(models.ConversationSession(user_id=1, session_id="s1"))
        db.add(models.UserMemory(
            user_id=1, memory_key="diet", memory_value="偏好",
            confidence=1.0, confirmed=True, status="active",
        ))
        db.add(models.ConversationLog(
            user_id=2, session_id="s2", agent_type="chat",
            user_message="其他用户", agent_response="保留",
        ))
        db.commit()
        db.close()
        request_ledger.begin_request(1, "done", "s1", "问题")
        request_ledger.complete_request(1, "done", {"response": "答案"})

        db = self.Session()
        clear_user_data_records(db, 1)
        self.assertEqual(db.query(models.DailyLog).filter_by(user_id=1).count(), 0)
        self.assertEqual(db.query(models.ConversationLog).filter_by(user_id=1).count(), 0)
        self.assertEqual(db.query(models.ConversationSession).filter_by(user_id=1).count(), 0)
        self.assertEqual(db.query(models.UserMemory).filter_by(user_id=1).count(), 0)
        self.assertEqual(db.query(models.RequestLedger).filter_by(user_id=1).count(), 0)
        self.assertEqual(db.query(models.ConversationLog).filter_by(user_id=2).count(), 1)
        db.close()

    def test_clear_rejects_live_request(self):
        request_ledger.begin_request(1, "live", "s1", "运动")
        db = self.Session()
        with self.assertRaises(ActiveRequestError):
            clear_user_data_records(db, 1)
        db.close()

    def test_migration_backfills_only_preexisting_confirmed_memory(self):
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as connection:
            connection.execute(text(
                "CREATE TABLE user_memories (id INTEGER PRIMARY KEY, confirmed BOOLEAN NOT NULL)"
            ))
            connection.execute(text(
                "INSERT INTO user_memories (id, confirmed) VALUES (1, 1), (2, 0)"
            ))
            _ensure_user_memory_status(connection)
            rows = connection.execute(text(
                "SELECT status FROM user_memories ORDER BY id"
            )).scalars().all()
            self.assertEqual(rows, ["active", "candidate"])
            connection.execute(text(
                "UPDATE user_memories SET status = 'candidate' WHERE id = 1"
            ))
            _ensure_user_memory_status(connection)
            self.assertEqual(connection.execute(text(
                "SELECT status FROM user_memories WHERE id = 1"
            )).scalar_one(), "candidate")
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
