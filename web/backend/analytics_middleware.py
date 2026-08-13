import time, json, sqlite3, hashlib, asyncio
from pathlib import Path
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp
from datetime import datetime, timezone

DB_PATH = Path("web/analytics.db")

def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = _get_conn()

    conn.executescript("""
    PRAGMA journal_mode=WAL;

    CREATE TABLE IF NOT EXISTS requests (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          REAL    NOT NULL,
        date_str    TEXT    NOT NULL,
        hour        INTEGER NOT NULL,
        minute      INTEGER NOT NULL,
        method      TEXT    NOT NULL,
        path        TEXT    NOT NULL,
        status_code INTEGER NOT NULL,
        latency_ms  REAL    NOT NULL,
        user_id     TEXT,
        model_name  TEXT,
        ip          TEXT,
        user_agent  TEXT,
        request_size_bytes  INTEGER DEFAULT 0,
        response_size_bytes INTEGER DEFAULT 0,
        is_error    INTEGER DEFAULT 0,
        is_chat     INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS errors (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          REAL    NOT NULL,
        date_str    TEXT    NOT NULL,
        path        TEXT,
        method      TEXT,
        status_code INTEGER,
        error_msg   TEXT,
        user_id     TEXT,
        ip          TEXT
    );

    CREATE TABLE IF NOT EXISTS model_usage (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ts              REAL    NOT NULL,
        date_str        TEXT    NOT NULL,
        hour            INTEGER NOT NULL,
        model_name      TEXT    NOT NULL,
        user_id         TEXT,
        prompt_len      INTEGER DEFAULT 0,
        reply_len       INTEGER DEFAULT 0,
        tokens_in       INTEGER DEFAULT 0,
        tokens_out      INTEGER DEFAULT 0,
        tokens_per_sec  REAL    DEFAULT 0,
        latency_ms      REAL    DEFAULT 0,
        session_id      TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_req_ts        ON requests(ts);
    CREATE INDEX IF NOT EXISTS idx_req_date      ON requests(date_str);
    CREATE INDEX IF NOT EXISTS idx_req_path      ON requests(path);
    CREATE INDEX IF NOT EXISTS idx_req_chat      ON requests(is_chat);
    CREATE INDEX IF NOT EXISTS idx_model_date    ON model_usage(date_str);
    CREATE INDEX IF NOT EXISTS idx_model_name    ON model_usage(model_name);
    CREATE INDEX IF NOT EXISTS idx_model_user    ON model_usage(user_id);
    CREATE INDEX IF NOT EXISTS idx_model_session ON model_usage(session_id);
    """)

    migrations = [
        "ALTER TABLE requests    ADD COLUMN is_chat         INTEGER DEFAULT 0",
        "ALTER TABLE model_usage ADD COLUMN tokens_in       INTEGER DEFAULT 0",
        "ALTER TABLE model_usage ADD COLUMN tokens_out      INTEGER DEFAULT 0",
        "ALTER TABLE model_usage ADD COLUMN tokens_per_sec  REAL    DEFAULT 0",
        "ALTER TABLE model_usage ADD COLUMN session_id      TEXT",
    ]
    for sql in migrations:
        try:
            conn.execute(sql)
        except Exception:
            pass

    conn.commit()
    conn.close()

def _hash_ip(ip: str) -> str:
    return hashlib.sha256(ip.encode()).hexdigest()[:16]

def log_request(
    ts: float, method: str, path: str, status: int, latency_ms: float,
    user_id: str | None, model_name: str | None,
    ip: str, user_agent: str,
    req_size: int, resp_size: int,
    is_chat: bool = False,
):
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO requests
               (ts, date_str, hour, minute, method, path, status_code, latency_ms,
                user_id, model_name, ip, user_agent,
                request_size_bytes, response_size_bytes, is_error, is_chat)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ts,
                dt.strftime("%Y-%m-%d"),
                dt.hour,
                dt.minute,
                method, path, status, round(latency_ms, 2),
                user_id, model_name,
                _hash_ip(ip),
                user_agent[:256] if user_agent else "",
                req_size, resp_size,
                1 if status >= 400 else 0,
                1 if is_chat else 0,
            ),
        )
        if status >= 400:
            conn.execute(
                """INSERT INTO errors (ts, date_str, path, method, status_code, ip, user_id)
                   VALUES (?,?,?,?,?,?,?)""",
                (ts, dt.strftime("%Y-%m-%d"), path, method, status, _hash_ip(ip), user_id),
            )
        conn.commit()
    finally:
        conn.close()


def log_model_usage(ts: float, model_name: str, user_id: str | None, prompt_len: int, reply_len: int, latency_ms: float, tokens_in: int = 0, tokens_out: int = 0, session_id: str | None = None):
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    tps = 0.0
    if tokens_out > 0 and latency_ms > 0:
        tps = round(tokens_out / (latency_ms / 1000), 2)

    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO model_usage
               (ts, date_str, hour, model_name, user_id,
                prompt_len, reply_len,
                tokens_in, tokens_out, tokens_per_sec,
                latency_ms, session_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ts, dt.strftime("%Y-%m-%d"), dt.hour,
                model_name, user_id,
                prompt_len, reply_len,
                tokens_in, tokens_out, tps,
                round(latency_ms, 2),
                session_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()

class AnalyticsMiddleware(BaseHTTPMiddleware):
    SKIP_PATHS = {"/metrics", "/favicon.ico", "/health"}
    CHAT_PATHS = {"/chat"}

    def __init__(self, app: ASGIApp, skip_prefixes: list[str] | None = None):
        super().__init__(app)
        self._skip_prefixes = skip_prefixes or ["/static", "/assets"]

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path in self.SKIP_PATHS or any(path.startswith(p) for p in self._skip_prefixes):
            return await call_next(request)

        start = time.perf_counter()
        req_body = b""
        try:
            req_body = await request.body()
        except Exception:
            pass

        user_id = None
        model_name = None
        session_id = None
        if req_body:
            try:
                payload = json.loads(req_body)
                user_id    = payload.get("user_id")
                model_name = payload.get("model_name")
                session_id = payload.get("session_id")
            except Exception:
                pass

        if not user_id:
            user_id = request.query_params.get("user_id")

        is_chat = path in self.CHAT_PATHS

        response: Response = await call_next(request)

        latency_ms = (time.perf_counter() - start) * 1000
        ts = time.time()

        ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown")
        ip = ip.split(",")[0].strip()
        ua = request.headers.get("user-agent", "")

        asyncio.get_event_loop().run_in_executor(
            None,
            log_request,
            ts, request.method, path, response.status_code, latency_ms,
            user_id, model_name, ip, ua,
            len(req_body), int(response.headers.get("content-length", 0)),
            is_chat,
        )

        return response
    