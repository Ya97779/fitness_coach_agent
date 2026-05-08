import os
import jwt
import httpx
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from . import database, models

security = HTTPBearer()

WECHAT_APPID = os.getenv("WECHAT_APPID", "")
WECHAT_SECRET = os.getenv("WECHAT_SECRET", "")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default-secret-change-me")
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "72"))


async def wx_code_to_session(code: str) -> dict:
    """调用微信 jscode2session 接口，换取 openid 和 session_key"""
    url = "https://api.weixin.qq.com/sns/jscode2session"
    params = {
        "appid": WECHAT_APPID,
        "secret": WECHAT_SECRET,
        "js_code": code,
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, timeout=10)
        data = resp.json()

    if "errcode" in data and data["errcode"] != 0:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"微信登录失败: {data.get('errmsg', '未知错误')}",
        )

    return {
        "openid": data["openid"],
        "unionid": data.get("unionid"),
        "session_key": data["session_key"],
    }


def create_access_token(user_id: int) -> str:
    """生成 JWT token"""
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")


def decode_access_token(token: str) -> int:
    """解析 JWT token，返回 user_id"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
        return int(payload["sub"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token 已过期"
        )
    except (jwt.InvalidTokenError, KeyError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的 Token"
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(database.get_db),
) -> models.User:
    """FastAPI 依赖：从 Authorization header 解析当前用户"""
    user_id = decode_access_token(credentials.credentials)
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在"
        )
    return user
