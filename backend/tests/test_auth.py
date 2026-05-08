import os
import sys
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jwt


class TestCreateAccessToken(unittest.TestCase):
    """测试 JWT token 生成"""

    @patch.dict(os.environ, {"JWT_SECRET_KEY": "test-secret-key", "JWT_EXPIRE_HOURS": "24"})
    def test_create_token_returns_string(self):
        from app.auth import create_access_token
        token = create_access_token(user_id=42)
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 0)

    @patch.dict(os.environ, {"JWT_SECRET_KEY": "test-secret-key", "JWT_EXPIRE_HOURS": "24"})
    def test_token_contains_user_id(self):
        from app.auth import create_access_token
        token = create_access_token(user_id=42)
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        self.assertEqual(payload["sub"], "42")

    @patch.dict(os.environ, {"JWT_SECRET_KEY": "test-secret-key", "JWT_EXPIRE_HOURS": "24"})
    def test_token_has_expiry(self):
        from app.auth import create_access_token
        token = create_access_token(user_id=1)
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        self.assertIn("exp", payload)
        exp_time = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        self.assertGreater(exp_time, datetime.now(timezone.utc))


class TestDecodeAccessToken(unittest.TestCase):
    """测试 JWT token 解析"""

    @patch.dict(os.environ, {"JWT_SECRET_KEY": "test-secret-key", "JWT_EXPIRE_HOURS": "24"})
    def test_decode_valid_token(self):
        from app.auth import create_access_token, decode_access_token
        token = create_access_token(user_id=99)
        user_id = decode_access_token(token)
        self.assertEqual(user_id, 99)

    @patch.dict(os.environ, {"JWT_SECRET_KEY": "test-secret-key", "JWT_EXPIRE_HOURS": "24"})
    def test_decode_expired_token_raises_401(self):
        from app.auth import decode_access_token
        from fastapi import HTTPException
        expired_payload = {
            "sub": "1",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
        }
        token = jwt.encode(expired_payload, "test-secret-key", algorithm="HS256")
        with self.assertRaises(HTTPException) as ctx:
            decode_access_token(token)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("过期", ctx.exception.detail)

    @patch.dict(os.environ, {"JWT_SECRET_KEY": "test-secret-key", "JWT_EXPIRE_HOURS": "24"})
    def test_decode_invalid_token_raises_401(self):
        from app.auth import decode_access_token
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            decode_access_token("invalid.token.here")
        self.assertEqual(ctx.exception.status_code, 401)

    @patch.dict(os.environ, {"JWT_SECRET_KEY": "test-secret-key", "JWT_EXPIRE_HOURS": "24"})
    def test_decode_wrong_secret_raises_401(self):
        from app.auth import decode_access_token
        from fastapi import HTTPException
        token = jwt.encode({"sub": "1", "exp": datetime.now(timezone.utc) + timedelta(hours=1)}, "wrong-secret", algorithm="HS256")
        with self.assertRaises(HTTPException):
            decode_access_token(token)


class TestGetCurrentUser(unittest.TestCase):
    """测试 get_current_user 依赖"""

    @patch.dict(os.environ, {"JWT_SECRET_KEY": "test-secret-key", "JWT_EXPIRE_HOURS": "24"})
    @patch("app.auth.database.SessionLocal")
    def test_valid_token_returns_user(self, mock_session_local):
        from app.auth import get_current_user, create_access_token
        from fastapi.security import HTTPAuthorizationCredentials

        mock_user = MagicMock()
        mock_user.id = 42

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        mock_session_local.return_value = mock_db

        token = create_access_token(42)
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        result = get_current_user(credentials=creds, db=mock_db)
        self.assertEqual(result.id, 42)

    @patch.dict(os.environ, {"JWT_SECRET_KEY": "test-secret-key", "JWT_EXPIRE_HOURS": "24"})
    def test_invalid_token_raises_401(self):
        from app.auth import get_current_user
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad.token.here")
        mock_db = MagicMock()
        with self.assertRaises(HTTPException):
            get_current_user(credentials=creds, db=mock_db)

    @patch.dict(os.environ, {"JWT_SECRET_KEY": "test-secret-key", "JWT_EXPIRE_HOURS": "24"})
    def test_user_not_found_raises_401(self):
        from app.auth import get_current_user, create_access_token
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        token = create_access_token(999)
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        with self.assertRaises(HTTPException) as ctx:
            get_current_user(credentials=creds, db=mock_db)
        self.assertEqual(ctx.exception.status_code, 401)


class TestWxCodeToSession(unittest.TestCase):
    """测试微信 jscode2session 调用"""

    @patch.dict(os.environ, {"WECHAT_APPID": "wx_test", "WECHAT_SECRET": "secret_test"})
    @patch("app.auth.httpx.AsyncClient")
    def test_success_returns_session(self, mock_client_cls):
        from app.auth import wx_code_to_session
        import asyncio

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "openid": "openid_123",
            "unionid": "unionid_456",
            "session_key": "session_789",
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        result = asyncio.get_event_loop().run_until_complete(
            wx_code_to_session("test_code")
        )
        self.assertEqual(result["openid"], "openid_123")
        self.assertEqual(result["unionid"], "unionid_456")
        self.assertEqual(result["session_key"], "session_789")

    @patch.dict(os.environ, {"WECHAT_APPID": "wx_test", "WECHAT_SECRET": "secret_test"})
    @patch("app.auth.httpx.AsyncClient")
    def test_error_response_raises_401(self, mock_client_cls):
        from app.auth import wx_code_to_session
        from fastapi import HTTPException
        import asyncio

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "errcode": 40029,
            "errmsg": "invalid code",
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        with self.assertRaises(HTTPException) as ctx:
            asyncio.get_event_loop().run_until_complete(
                wx_code_to_session("bad_code")
            )
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("微信登录失败", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
