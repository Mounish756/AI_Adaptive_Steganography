"""
Unified launcher: starts the Stego API server AND a static file server for the frontend.

Usage:
  python start.py

Then open: http://127.0.0.1:8080
"""

import os
import sys
import time
import signal
import threading
import subprocess
import http.server
import socketserver
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

API_PORT      = 5050
FRONTEND_PORT = 8080
FRONTEND_DIR  = PROJECT_ROOT / "frontend"


class SilentHandler(http.server.SimpleHTTPRequestHandler):
    """Serve static files with CORS headers (for local development)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # Suppress noise


def serve_frontend():
    with socketserver.TCPServer(("127.0.0.1", FRONTEND_PORT), SilentHandler) as httpd:
        print(f"[+] Frontend server:  http://127.0.0.1:{FRONTEND_PORT}")
        httpd.serve_forever()


def run():
    print("=" * 60)
    print("  AI Adaptive Steganography Dashboard")
    print("=" * 60)
    print()

    # 1. Start API server in a subprocess
    api_proc = subprocess.Popen(
        [sys.executable, "api/server.py"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Wait for API server to be ready
    time.sleep(3)
    if api_proc.poll() is not None:
        print("[ERROR] API server failed to start. Check api/server.py.")
        sys.exit(1)

    print(f"[+] API server:       http://127.0.0.1:{API_PORT}")

    # 2. Start frontend server in a thread
    t = threading.Thread(target=serve_frontend, daemon=True)
    t.start()

    time.sleep(0.5)
    print()
    print("=" * 60)
    print(f"  Open your browser:  http://127.0.0.1:{FRONTEND_PORT}")
    print("  Press Ctrl+C to stop both servers.")
    print("=" * 60)

    # Try to open browser automatically
    try:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{FRONTEND_PORT}")
    except Exception:
        pass

    # Stream API server output
    try:
        for line in api_proc.stdout:
            print(f"[API] {line}", end="")
    except KeyboardInterrupt:
        pass
    finally:
        api_proc.terminate()
        print("\n[*] Servers stopped.")


if __name__ == "__main__":
    run()
