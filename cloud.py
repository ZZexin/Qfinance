"""
GitHub Gist cloud storage for Qfinance portfolio data.

Setup (one-time):
  1. Create a GitHub Personal Access Token at https://github.com/settings/tokens
     → Scope required: gist
  2. Enter the token in the app's 💼 持仓 tab → 云端连接
  3. Click "连接 / 创建 Gist" — the app auto-creates a private gist and
     saves credentials to .streamlit/secrets.toml for next time.
"""

import json
import requests
from pathlib import Path

GIST_FILENAME = "qfinance_portfolio.json"
SECRETS_PATH  = Path(__file__).parent / ".streamlit" / "secrets.toml"

EMPTY = {"holdings": [], "transactions": [], "value_history": []}


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def test_token(token: str) -> tuple[bool, str]:
    """Returns (success, github_username_or_error_message)."""
    try:
        r = requests.get("https://api.github.com/user", headers=_headers(token), timeout=8)
        if r.status_code == 200:
            return True, r.json().get("login", "")
        return False, f"HTTP {r.status_code} — 请确认 Token 有 gist 权限"
    except requests.ConnectionError:
        return False, "网络连接失败，请检查网络"
    except Exception as e:
        return False, str(e)


def create_gist(token: str) -> str:
    """Create a new private gist and return its ID."""
    r = requests.post(
        "https://api.github.com/gists",
        headers=_headers(token),
        json={
            "description": "Qfinance Portfolio Data — do not delete",
            "public": False,
            "files": {GIST_FILENAME: {"content": json.dumps(EMPTY, indent=2)}},
        },
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["id"]


def load(token: str, gist_id: str) -> dict:
    """Load portfolio data from GitHub Gist."""
    r = requests.get(
        f"https://api.github.com/gists/{gist_id}",
        headers=_headers(token),
        timeout=10,
    )
    r.raise_for_status()
    raw = r.json()["files"].get(GIST_FILENAME, {}).get("content", "{}")
    data = json.loads(raw)
    for key in EMPTY:          # ensure all keys exist
        data.setdefault(key, [])
    return data


def save(token: str, gist_id: str, data: dict):
    """Write portfolio data back to GitHub Gist."""
    r = requests.patch(
        f"https://api.github.com/gists/{gist_id}",
        headers=_headers(token),
        json={"files": {GIST_FILENAME: {
            "content": json.dumps(data, ensure_ascii=False, indent=2)
        }}},
        timeout=10,
    )
    r.raise_for_status()


def save_credentials(token: str, gist_id: str):
    """Persist token + gist_id to .streamlit/secrets.toml (never committed to git)."""
    SECRETS_PATH.parent.mkdir(exist_ok=True)
    SECRETS_PATH.write_text(
        f'github_token = "{token}"\ngist_id = "{gist_id}"\n',
        encoding="utf-8",
    )


def load_credentials() -> tuple[str, str]:
    """Try to read saved credentials. Returns (token, gist_id) or ('', '')."""
    try:
        import streamlit as st
        token   = st.secrets.get("github_token", "")
        gist_id = st.secrets.get("gist_id", "")
        return token, gist_id
    except Exception:
        return "", ""
