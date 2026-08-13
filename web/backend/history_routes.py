import json, re
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()
HISTORY_DIR = Path("web/chat_history")

def _ensure_dir():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

def _valid_id(value: str) -> bool:
    return bool(value) and bool(re.fullmatch(r"[a-zA-Z0-9_\-]+", value))

def _session_path(session_id: str) -> Path:
    return HISTORY_DIR / f"{session_id}.json"

class Message(BaseModel):
    role: str
    text: str
    model: Optional[str] = None

class Session(BaseModel):
    id: str
    title: str
    messages: List[Message]
    created_at: float
    user_id: str
    shared: bool = False

@router.get("/history")
def get_my_history(user_id: str = Query(...)):
    if not _valid_id(user_id):
        raise HTTPException(400, "Invalid user_id")
    _ensure_dir()
    sessions = []
    for f in HISTORY_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("user_id") == user_id:
                sessions.append({
                    "id":         data["id"],
                    "title":      data["title"],
                    "created_at": data["created_at"],
                    "shared":     data.get("shared", False),
                })
        except Exception:
            continue
    return {"sessions": sessions}

@router.get("/history/community")
def get_community():
    _ensure_dir()
    sessions = []
    for f in HISTORY_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("shared"):
                sessions.append({
                    "id":         data["id"],
                    "title":      data["title"],
                    "created_at": data["created_at"],
                    "user_id":    data["user_id"],
                })
        except Exception:
            continue
    sessions.sort(key=lambda s: s["created_at"], reverse=True)
    return {"sessions": sessions}

@router.get("/history/{session_id}")
def get_session(session_id: str, user_id: str = Query(...)):
    """Load session đầy đủ — chỉ owner hoặc session shared mới đọc được."""
    if not _valid_id(session_id) or not _valid_id(user_id):
        raise HTTPException(400, "Invalid id")
    path = _session_path(session_id)
    if not path.exists():
        raise HTTPException(404, "Session not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("user_id") != user_id and not data.get("shared"):
        raise HTTPException(403, "Forbidden")
    return data

@router.post("/history/save")
def save_session(session: Session):
    """Lưu / cập nhật session. user_id trong body phải khớp file nếu đã tồn tại."""
    if not _valid_id(session.id) or not _valid_id(session.user_id):
        raise HTTPException(400, "Invalid id")
    _ensure_dir()
    path = _session_path(session.id)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("user_id") != session.user_id:
            raise HTTPException(403, "Forbidden")
    path.write_text(
        json.dumps(session.dict(), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return {"ok": True}

@router.patch("/history/{session_id}/share")
def toggle_share(session_id: str, user_id: str = Query(...)):
    if not _valid_id(session_id) or not _valid_id(user_id):
        raise HTTPException(400, "Invalid id")
    path = _session_path(session_id)
    if not path.exists():
        raise HTTPException(404, "Session not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("user_id") != user_id:
        raise HTTPException(403, "Forbidden")
    data["shared"] = not data.get("shared", False)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "shared": data["shared"]}

@router.delete("/history/{session_id}")
def delete_session(session_id: str, user_id: str = Query(...)):
    if not _valid_id(session_id) or not _valid_id(user_id):
        raise HTTPException(400, "Invalid id")
    path = _session_path(session_id)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("user_id") != user_id:
            raise HTTPException(403, "Forbidden")
        path.unlink()
    return {"ok": True}
