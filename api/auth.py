"""
api/auth.py

App-level authentication and role-based access control.

Deliberately separate from the database layer. The bot always connects to
MySQL using the single shared DB_URL_READONLY service account, regardless
of which human is logged in -- that's unchanged. This module handles WHO
is allowed to use the bot's API at all, and (via table_access.py) WHICH
tables their questions are allowed to touch.

Two roles for now: "standard" and "admin". Kept intentionally simple --
this is a framework for access control, not a finished policy. No tables
are actually restricted yet (see table_access.py) since no sensitive
categories have been identified in the target schema. The enforcement
point exists now so restricting access later is a config change, not a
redesign.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.environ["JWT_SECRET_KEY"]  # must be set in .env -- see setup notes
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8-hour session, adjust as needed

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# App-level user store. Deliberately a SEPARATE table/connection from the
# target production database -- this has nothing to do with the 200-table
# schema being queried, it's purely "who is allowed to use this tool."
# Uses the same MySQL instance for convenience, but a different schema/DB
# name, or even a separate SQLite file, would work equally well.
APP_DB_URL = os.environ.get("APP_DB_URL", os.environ.get("DB_URL"))
_app_engine = None


def get_app_engine():
    global _app_engine
    if _app_engine is None:
        _app_engine = create_engine(APP_DB_URL)
    return _app_engine


def hash_password(password: str) -> str:
    # bcrypt has a hard 72-byte input limit -- truncate defensively rather
    # than letting an unusually long password raise an error later.
    password_bytes = password.encode("utf-8")[:72]
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_bytes = plain_password.encode("utf-8")[:72]
    return bcrypt.checkpw(password_bytes, hashed_password.encode("utf-8"))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_user_by_username(username: str) -> Optional[dict]:
    engine = get_app_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT id, username, password_hash, role FROM app_users WHERE username = :username"),
            {"username": username},
        ).fetchone()
        if result is None:
            return None
        return {"id": result[0], "username": result[1], "password_hash": result[2], "role": result[3]}


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    FastAPI dependency -- add `current_user: dict = Depends(get_current_user)`
    to any endpoint that requires login. Raises 401 if token is missing,
    expired, or invalid.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user_by_username(username)
    if user is None:
        raise credentials_exception
    return user


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """
    FastAPI dependency for admin-only endpoints. Use this instead of
    get_current_user when an endpoint should reject non-admin users
    entirely (e.g. future email-sending endpoints).
    """
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action requires admin privileges.",
        )
    return current_user


# --- One-time setup: run this to create the app_users table ---
CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS app_users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'standard',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def create_users_table():
    engine = get_app_engine()
    with engine.connect() as conn:
        conn.execute(text(CREATE_USERS_TABLE_SQL))
        conn.commit()
    print("app_users table ready.")


def create_user(username: str, password: str, role: str = "standard"):
    engine = get_app_engine()
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO app_users (username, password_hash, role) VALUES (:u, :p, :r)"),
            {"u": username, "p": hash_password(password), "r": role},
        )
        conn.commit()
    print(f"User '{username}' created with role '{role}'.")


if __name__ == "__main__":
    # Run this file directly once to set up the users table and create
    # a first admin account for testing.
    create_users_table()

    print("\nCreate an initial admin user:")
    username = input("Username: ")
    password = input("Password: ")
    create_user(username, password, role="admin")