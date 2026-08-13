import os, sqlite3, time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(prefix="/admin")

DB_PATH = Path("web/analytics.db")
ADMIN_KEY = os.environ.get("ADMIN_KEY", "changeme")

def _auth(request: Request):
    key = request.headers.get("X-Admin-Key") or request.query_params.get("admin_key")
    if key != ADMIN_KEY:
        raise HTTPException(401, "Unauthorized — set X-Admin-Key header")

def _conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise HTTPException(503, "Analytics DB chưa sẵn sàng")
    c = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def _since(days: int) -> str:
    return (datetime.now(tz=timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

@router.get("/api/web/overview")
def web_overview(request: Request, days: int = Query(7, ge=1, le=90)):
    _auth(request)
    conn = _conn()
    since = _since(days)

    total = conn.execute(
        "SELECT COUNT(*) as n FROM requests WHERE date_str >= ? AND is_chat=0", (since,)
    ).fetchone()["n"]

    errors = conn.execute(
        "SELECT COUNT(*) as n FROM requests WHERE date_str >= ? AND is_chat=0 AND is_error=1", (since,)
    ).fetchone()["n"]

    avg_lat = conn.execute(
        "SELECT AVG(latency_ms) as v FROM requests WHERE date_str >= ? AND is_chat=0", (since,)
    ).fetchone()["v"] or 0

    all_lats = conn.execute(
        "SELECT latency_ms FROM requests WHERE date_str >= ? AND is_chat=0 ORDER BY latency_ms ASC",
        (since,),
    ).fetchall()
    p95_val = 0
    if all_lats:
        idx = min(int(len(all_lats) * 0.95), len(all_lats) - 1)
        p95_val = all_lats[idx]["latency_ms"]

    unique_users = conn.execute(
        "SELECT COUNT(DISTINCT user_id) as n FROM requests WHERE date_str >= ? AND is_chat=0 AND user_id IS NOT NULL",
        (since,),
    ).fetchone()["n"]

    conn.close()
    return {
        "days": days,
        "total_requests": total,
        "error_requests": errors,
        "error_rate_pct": round(errors / total * 100, 2) if total else 0,
        "avg_latency_ms": round(avg_lat, 1),
        "p95_latency_ms": round(p95_val, 1),
        "unique_users": unique_users,
    }

@router.get("/api/web/daily")
def web_daily(request: Request, days: int = Query(30, ge=1, le=365)):
    _auth(request)
    conn = _conn()
    since = _since(days)
    rows = conn.execute(
        """SELECT date_str,
                  COUNT(*) as total,
                  SUM(is_error) as errors,
                  AVG(latency_ms) as avg_lat,
                  COUNT(DISTINCT user_id) as unique_users
           FROM requests WHERE date_str >= ? AND is_chat=0
           GROUP BY date_str ORDER BY date_str""",
        (since,),
    ).fetchall()
    conn.close()
    return {"data": [dict(r) for r in rows]}

@router.get("/api/web/hourly")
def web_hourly(request: Request, days: int = Query(7, ge=1, le=30)):
    _auth(request)
    conn = _conn()
    since = _since(days)
    rows = conn.execute(
        """SELECT date_str, hour,
                  COUNT(*) as total,
                  AVG(latency_ms) as avg_lat
           FROM requests WHERE date_str >= ? AND is_chat=0
           GROUP BY date_str, hour ORDER BY date_str, hour""",
        (since,),
    ).fetchall()
    conn.close()
    return {"data": [dict(r) for r in rows]}

@router.get("/api/web/top-paths")
def web_top_paths(request: Request, days: int = Query(7, ge=1, le=30), limit: int = Query(20, ge=5, le=100)):
    _auth(request)
    conn = _conn()
    since = _since(days)
    rows = conn.execute(
        """SELECT path, COUNT(*) as hits, AVG(latency_ms) as avg_lat,
                  SUM(is_error) as errors
           FROM requests WHERE date_str >= ? AND is_chat=0
           GROUP BY path ORDER BY hits DESC LIMIT ?""",
        (since, limit),
    ).fetchall()
    conn.close()
    return {"data": [dict(r) for r in rows]}

@router.get("/api/web/latency")
def web_latency(request: Request, days: int = Query(7, ge=1, le=30)):
    _auth(request)
    conn = _conn()
    since = _since(days)
    rows = conn.execute(
        "SELECT latency_ms FROM requests WHERE date_str >= ? AND is_chat=0 ORDER BY latency_ms ASC",
        (since,),
    ).fetchall()
    conn.close()
    lats = [r["latency_ms"] for r in rows]
    return _percentile_response(lats)

@router.get("/api/ai/overview")
def ai_overview(request: Request, days: int = Query(7, ge=1, le=90)):
    _auth(request)
    conn = _conn()
    since = _since(days)

    total_calls = conn.execute(
        "SELECT COUNT(*) as n FROM model_usage WHERE date_str >= ?", (since,)
    ).fetchone()["n"]

    unique_users = conn.execute(
        "SELECT COUNT(DISTINCT user_id) as n FROM model_usage WHERE date_str >= ? AND user_id IS NOT NULL",
        (since,),
    ).fetchone()["n"]

    avg_lat = conn.execute(
        "SELECT AVG(latency_ms) as v FROM model_usage WHERE date_str >= ?", (since,)
    ).fetchone()["v"] or 0

    # p95 latency
    all_lats = conn.execute(
        "SELECT latency_ms FROM model_usage WHERE date_str >= ? ORDER BY latency_ms ASC", (since,)
    ).fetchall()
    p95_val = 0
    if all_lats:
        idx = min(int(len(all_lats) * 0.95), len(all_lats) - 1)
        p95_val = all_lats[idx]["latency_ms"]

    # token stats
    tok = conn.execute(
        """SELECT
             AVG(tokens_in)      as avg_tok_in,
             AVG(tokens_out)     as avg_tok_out,
             SUM(tokens_in)      as total_tok_in,
             SUM(tokens_out)     as total_tok_out,
             AVG(CASE WHEN tokens_per_sec > 0 THEN tokens_per_sec END) as avg_tps
           FROM model_usage WHERE date_str >= ?""",
        (since,),
    ).fetchone()

    # req/user
    req_per_user = conn.execute(
        """SELECT AVG(cnt) as v FROM (
             SELECT COUNT(*) as cnt FROM model_usage
             WHERE date_str >= ? AND user_id IS NOT NULL
             GROUP BY user_id
           )""",
        (since,),
    ).fetchone()["v"] or 0

    conn.close()
    return {
        "days": days,
        "total_calls": total_calls,
        "unique_users": unique_users,
        "avg_latency_ms": round(avg_lat, 1),
        "p95_latency_ms": round(p95_val, 1),
        "avg_tokens_in": round(tok["avg_tok_in"] or 0, 1),
        "avg_tokens_out": round(tok["avg_tok_out"] or 0, 1),
        "total_tokens_in": int(tok["total_tok_in"] or 0),
        "total_tokens_out": int(tok["total_tok_out"] or 0),
        "avg_tokens_per_sec": round(tok["avg_tps"] or 0, 2),
        "avg_req_per_user": round(req_per_user, 2),
    }

@router.get("/api/ai/requests")
def ai_requests(request: Request, limit: int = Query(100, ge=1, le=500)):
    """100 request chat gần nhất, đủ chi tiết."""
    _auth(request)
    conn = _conn()
    rows = conn.execute(
        """SELECT id, ts, model_name, user_id, session_id,
                  prompt_len, reply_len,
                  tokens_in, tokens_out, tokens_per_sec,
                  latency_ms
           FROM model_usage ORDER BY id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["ts_human"] = datetime.fromtimestamp(d["ts"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        result.append(d)
    return {"data": result}

@router.get("/api/ai/daily")
def ai_daily(request: Request, days: int = Query(30, ge=1, le=365)):
    _auth(request)
    conn = _conn()
    since = _since(days)
    rows = conn.execute(
        """SELECT date_str, model_name,
                  COUNT(*) as calls,
                  AVG(latency_ms) as avg_lat,
                  AVG(tokens_in) as avg_tok_in,
                  AVG(tokens_out) as avg_tok_out,
                  AVG(CASE WHEN tokens_per_sec > 0 THEN tokens_per_sec END) as avg_tps,
                  COUNT(DISTINCT user_id) as unique_users
           FROM model_usage WHERE date_str >= ?
           GROUP BY date_str, model_name ORDER BY date_str""",
        (since,),
    ).fetchall()
    conn.close()
    return {"data": [dict(r) for r in rows]}

@router.get("/api/ai/token-stats")
def ai_token_stats(request: Request, days: int = Query(7, ge=1, le=90)):
    """Thống kê chi tiết token: histogram, percentile, phân bố theo model."""
    _auth(request)
    conn = _conn()
    since = _since(days)

    by_model = conn.execute(
        """SELECT model_name,
                  COUNT(*) as calls,
                  AVG(tokens_in) as avg_tok_in,
                  AVG(tokens_out) as avg_tok_out,
                  SUM(tokens_in) as total_tok_in,
                  SUM(tokens_out) as total_tok_out,
                  AVG(CASE WHEN tokens_per_sec > 0 THEN tokens_per_sec END) as avg_tps,
                  MAX(tokens_per_sec) as max_tps,
                  AVG(latency_ms) as avg_lat,
                  AVG(reply_len) as avg_reply_chars,
                  AVG(prompt_len) as avg_prompt_chars
           FROM model_usage WHERE date_str >= ?
           GROUP BY model_name ORDER BY calls DESC""",
        (since,),
    ).fetchall()

    # histogram tokens_out (phân bố độ dài reply)
    tok_out_rows = conn.execute(
        "SELECT tokens_out FROM model_usage WHERE date_str >= ? AND tokens_out > 0 ORDER BY tokens_out",
        (since,),
    ).fetchall()

    # histogram latency AI
    lat_rows = conn.execute(
        "SELECT latency_ms FROM model_usage WHERE date_str >= ? ORDER BY latency_ms",
        (since,),
    ).fetchall()

    conn.close()

    tok_out_vals = [r["tokens_out"] for r in tok_out_rows]
    lat_vals     = [r["latency_ms"]  for r in lat_rows]

    return {
        "by_model": [
            {k: (round(v, 2) if isinstance(v, float) else v)
             for k, v in dict(r).items()}
            for r in by_model
        ],
        "tokens_out_dist": _build_histogram(tok_out_vals, buckets=15),
        "latency_dist":    _percentile_response(lat_vals),
    }

@router.get("/api/ai/per-user")
def ai_per_user(request: Request, days: int = Query(7, ge=1, le=90), limit: int = Query(30, ge=5, le=200)):
    """Số request, token và tốc độ trung bình theo từng user."""
    _auth(request)
    conn = _conn()
    since = _since(days)
    rows = conn.execute(
        """SELECT user_id,
                  COUNT(*) as calls,
                  AVG(latency_ms) as avg_lat,
                  SUM(tokens_in) as total_tok_in,
                  SUM(tokens_out) as total_tok_out,
                  AVG(CASE WHEN tokens_per_sec > 0 THEN tokens_per_sec END) as avg_tps,
                  COUNT(DISTINCT session_id) as sessions,
                  MIN(ts) as first_seen,
                  MAX(ts) as last_seen
           FROM model_usage WHERE date_str >= ? AND user_id IS NOT NULL
           GROUP BY user_id ORDER BY calls DESC LIMIT ?""",
        (since, limit),
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        # Số req trung bình mỗi session
        d["req_per_session"] = round(d["calls"] / d["sessions"], 2) if d["sessions"] else 0
        d["first_seen_h"] = datetime.fromtimestamp(d["first_seen"], tz=timezone.utc).strftime("%Y-%m-%d") if d["first_seen"] else None
        d["last_seen_h"]  = datetime.fromtimestamp(d["last_seen"],  tz=timezone.utc).strftime("%Y-%m-%d") if d["last_seen"] else None
        result.append({k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()})
    return {"data": result}

@router.get("/api/ai/latency")
def ai_latency(request: Request, days: int = Query(7, ge=1, le=30)):
    _auth(request)
    conn = _conn()
    since = _since(days)
    rows = conn.execute(
        "SELECT latency_ms FROM model_usage WHERE date_str >= ? ORDER BY latency_ms ASC", (since,)
    ).fetchall()
    conn.close()
    return _percentile_response([r["latency_ms"] for r in rows])

@router.get("/api/overview")
def overview(request: Request, days: int = Query(7, ge=1, le=90)):
    _auth(request)
    conn = _conn()
    since = _since(days)
    total = conn.execute("SELECT COUNT(*) as n FROM requests WHERE date_str >= ?", (since,)).fetchone()["n"]
    errors = conn.execute("SELECT COUNT(*) as n FROM requests WHERE date_str >= ? AND is_error=1", (since,)).fetchone()["n"]
    avg_lat = conn.execute("SELECT AVG(latency_ms) as v FROM requests WHERE date_str >= ?", (since,)).fetchone()["v"] or 0
    all_lats = conn.execute("SELECT latency_ms FROM requests WHERE date_str >= ? ORDER BY latency_ms ASC", (since,)).fetchall()
    p95_val = 0
    if all_lats:
        idx = min(int(len(all_lats) * 0.95), len(all_lats) - 1)
        p95_val = all_lats[idx]["latency_ms"]
    unique_users = conn.execute("SELECT COUNT(DISTINCT user_id) as n FROM requests WHERE date_str >= ? AND user_id IS NOT NULL", (since,)).fetchone()["n"]
    total_chats = conn.execute("SELECT COUNT(*) as n FROM model_usage WHERE date_str >= ?", (since,)).fetchone()["n"]
    conn.close()
    return {"days": days, "total_requests": total, "error_requests": errors,
            "error_rate_pct": round(errors / total * 100, 2) if total else 0,
            "avg_latency_ms": round(avg_lat, 1), "p95_latency_ms": round(p95_val, 1),
            "unique_users": unique_users, "total_chats": total_chats}

@router.get("/api/requests/daily")
def requests_daily(request: Request, days: int = Query(30, ge=1, le=365)):
    _auth(request)
    conn = _conn()
    since = _since(days)
    rows = conn.execute("""SELECT date_str, COUNT(*) as total, SUM(is_error) as errors,
                  AVG(latency_ms) as avg_lat, COUNT(DISTINCT user_id) as unique_users
           FROM requests WHERE date_str >= ? GROUP BY date_str ORDER BY date_str""", (since,)).fetchall()
    conn.close()
    return {"data": [dict(r) for r in rows]}

@router.get("/api/requests/hourly")
def requests_hourly(request: Request, days: int = Query(7, ge=1, le=30)):
    _auth(request)
    conn = _conn()
    since = _since(days)
    rows = conn.execute("""SELECT date_str, hour, COUNT(*) as total, AVG(latency_ms) as avg_lat
           FROM requests WHERE date_str >= ? GROUP BY date_str, hour ORDER BY date_str, hour""", (since,)).fetchall()
    conn.close()
    return {"data": [dict(r) for r in rows]}

@router.get("/api/latency/distribution")
def latency_dist(request: Request, days: int = Query(7, ge=1, le=30), path: str = Query(None)):
    _auth(request)
    conn = _conn()
    since = _since(days)
    q = "SELECT latency_ms FROM requests WHERE date_str >= ?"
    params = [since]
    if path:
        q += " AND path = ?"
        params.append(path)
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return _percentile_response([r["latency_ms"] for r in rows])

@router.get("/api/requests/recent")
def recent_requests(request: Request, limit: int = Query(100, ge=1, le=500)):
    _auth(request)
    conn = _conn()
    rows = conn.execute("""SELECT id, ts, method, path, status_code, latency_ms,
                  user_id, model_name, user_agent, request_size_bytes, response_size_bytes, is_error
           FROM requests ORDER BY id DESC LIMIT ?""", (limit,)).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["ts_human"] = datetime.fromtimestamp(d["ts"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        result.append(d)
    return {"data": result}

@router.get("/api/requests/top-paths")
def top_paths(request: Request, days: int = Query(7, ge=1, le=30), limit: int = Query(20, ge=5, le=100)):
    _auth(request)
    conn = _conn()
    since = _since(days)
    rows = conn.execute("""SELECT path, COUNT(*) as hits, AVG(latency_ms) as avg_lat, SUM(is_error) as errors
           FROM requests WHERE date_str >= ? GROUP BY path ORDER BY hits DESC LIMIT ?""", (since, limit)).fetchall()
    conn.close()
    return {"data": [dict(r) for r in rows]}

@router.get("/api/models/usage")
def model_usage(request: Request, days: int = Query(7, ge=1, le=90)):
    _auth(request)
    conn = _conn()
    since = _since(days)
    by_model = conn.execute("""SELECT model_name, COUNT(*) as calls, AVG(latency_ms) as avg_lat,
                  AVG(reply_len) as avg_reply_len, SUM(prompt_len) as total_prompt_chars, SUM(reply_len) as total_reply_chars
           FROM model_usage WHERE date_str >= ? GROUP BY model_name ORDER BY calls DESC""", (since,)).fetchall()
    daily = conn.execute("""SELECT date_str, model_name, COUNT(*) as calls, AVG(latency_ms) as avg_lat
           FROM model_usage WHERE date_str >= ? GROUP BY date_str, model_name ORDER BY date_str""", (since,)).fetchall()
    conn.close()
    return {"by_model": [dict(r) for r in by_model], "daily": [dict(r) for r in daily]}

@router.get("/api/errors")
def error_log(request: Request, days: int = Query(7, ge=1, le=30), limit: int = Query(200, ge=10, le=1000)):
    _auth(request)
    conn = _conn()
    since = _since(days)
    rows = conn.execute("""SELECT id, ts, date_str, path, method, status_code, error_msg, user_id
           FROM errors WHERE date_str >= ? ORDER BY id DESC LIMIT ?""", (since, limit)).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["ts_human"] = datetime.fromtimestamp(d["ts"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        result.append(d)
    return {"data": result}

@router.get("/api/users/activity")
def user_activity(request: Request, days: int = Query(7, ge=1, le=30)):
    _auth(request)
    conn = _conn()
    since = _since(days)
    rows = conn.execute("""SELECT date_str, COUNT(DISTINCT user_id) as dau, COUNT(*) as requests
           FROM requests WHERE date_str >= ? AND user_id IS NOT NULL
           GROUP BY date_str ORDER BY date_str""", (since,)).fetchall()
    conn.close()
    return {"data": [dict(r) for r in rows]}

def _percentile_response(lats: list[float]) -> dict:
    if not lats:
        return {"count": 0, "percentiles": {}, "histogram": []}
    lats = sorted(lats)

    def pct(p):
        idx = int(len(lats) * p / 100)
        return round(lats[min(idx, len(lats) - 1)], 1)

    return {
        "count": len(lats),
        "percentiles": {
            "p50": pct(50), "p75": pct(75),
            "p90": pct(90), "p95": pct(95), "p99": pct(99),
            "min": round(lats[0], 1), "max": round(lats[-1], 1),
        },
        "histogram": _build_histogram(lats, buckets=20),
    }

def _build_histogram(vals: list[float], buckets: int = 20) -> list[dict]:
    if not vals:
        return []
    mn, mx = vals[0], vals[-1]
    bw = max((mx - mn) / buckets, 1)
    hist: dict[float, int] = {}
    for v in vals:
        b = round((int((v - mn) / bw) * bw) + mn, 1)
        hist[b] = hist.get(b, 0) + 1
    return [{"bucket": k, "count": v} for k, v in sorted(hist.items())]
