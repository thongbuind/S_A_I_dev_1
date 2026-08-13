import asyncio
import time
from collections import defaultdict
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from web.backend.analytics_middleware import AnalyticsMiddleware, init_db, log_model_usage
from web.backend.analytics_routes import router as analytics_router
from web.backend.history_routes import router as history_router
from web.backend.model_loader import load_model, generate_response, model_dir

app = FastAPI()

app.include_router(history_router)
app.include_router(analytics_router)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(AnalyticsMiddleware)

models: dict[str, object] = {}
tokenizers: dict[str, object] = {}
default_model_name: str = ""
_queue: asyncio.Queue = None

_last_request: dict[str, float] = defaultdict(float)
RATE_LIMIT_SEC = 2.0
frontend_dir = Path("web/frontend")

def _count_tokens(tokenizer, text: str) -> int:
    try:
        if tokenizer is None:
            return 0
        encoded = tokenizer.encode(text)
        ids = getattr(encoded, "ids", None) or getattr(encoded, "input_ids", None)
        if ids is not None:
            return len(ids)
        if isinstance(encoded, (list, tuple)):
            return len(encoded)
        return 0
    except Exception:
        return 0

async def _worker():
    while True:
        model_name, message, future = await _queue.get()
        try:
            m = models.get(model_name) or models.get(default_model_name)
            t = tokenizers.get(model_name) or tokenizers.get(default_model_name)
            actual_name = model_name if model_name in models else default_model_name

            start = time.perf_counter()
            reply = await asyncio.to_thread(generate_response, m, t, message)
            latency_ms = (time.perf_counter() - start) * 1000

            log_model_usage(
                ts=time.time(),
                model_name=actual_name,
                user_id=getattr(future, "_user_id", None),
                prompt_len=len(message),
                reply_len=len(reply) if reply else 0,
                latency_ms=latency_ms,
                tokens_in=_count_tokens(t, message),
                tokens_out=_count_tokens(t, reply) if reply else 0,
                session_id=getattr(future, "_session_id", None),
            )

            future.set_result({"reply": reply, "model": actual_name})
        except Exception as e:
            future.set_exception(e)
        finally:
            _queue.task_done()

@app.on_event("startup")
async def startup():
    global _queue, default_model_name

    init_db()
    _queue = asyncio.Queue()

    pts = sorted([f for f in model_dir.glob("*.pt")
                  if "pretrain" not in f.stem.lower()])
    if not pts:
        print("Không tìm thấy model nào!")
        return

    for pt in pts:
        print(f"Loading {pt.name}...")
        m, t = await asyncio.to_thread(load_model, pt)
        models[pt.name] = m
        tokenizers[pt.name] = t
        print(f"  ✓ {pt.name}")

    default_model_name = pts[0].name
    asyncio.create_task(_worker())
    print(f"Server ready — {len(models)} model(s) loaded")

class ChatRequest(BaseModel):
    message: str
    user_id: str
    model_name: Optional[str] = None
    session_id: Optional[str] = None

@app.get("/models")
def list_models():
    return {"models": list(models.keys()), "current": default_model_name}

@app.post("/chat")
async def chat(req: ChatRequest):
    if time.time() - _last_request[req.user_id] < RATE_LIMIT_SEC:
        raise HTTPException(429, "Gửi quá nhanh, thử lại sau.")
    _last_request[req.user_id] = time.time()

    loop = asyncio.get_event_loop()
    future = loop.create_future()
    future._user_id = req.user_id
    future._session_id = req.session_id
    await _queue.put((req.model_name or default_model_name, req.message, future))

    return await future

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(frontend_dir / "index.html")

@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa(full_path: str):
    file = frontend_dir / full_path
    if file.exists() and file.is_file():
        return FileResponse(file)
    index = frontend_dir / full_path / "index.html"
    if index.exists():
        return FileResponse(index)
    return FileResponse(frontend_dir / "index.html")
