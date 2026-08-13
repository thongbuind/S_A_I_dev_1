import subprocess
import sys
import os
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

server_cmd = [
    sys.executable,
    "-m",
    "uvicorn",
    "web.backend.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
]

print("Starting AI server (port 8000)...")
server = subprocess.Popen(
    server_cmd,
    cwd=ROOT,
    env={**os.environ, "PYTHONPATH": str(ROOT)}
)

print("\nSystem running:")
print("Web + API → http://127.0.0.1:8000")
print("Admin     → http://127.0.0.1:8000/admin.html")
print("\nPress CTRL+C to stop\n")

def shutdown(sig, frame):
    print("\nStopping...")
    server.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
server.wait()
