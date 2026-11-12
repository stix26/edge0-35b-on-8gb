#!/usr/bin/env python3
"""Tiny same-origin chat front end for edge0-infer serve.

Serves a single page on :8080 and proxies POST /v1/chat/completions to the
engine on :8000, relaying server-sent events so answers stream into the page.
Running the proxy on the same origin avoids CORS, which edge0-infer does not
implement.
"""

import http.client
import json
import os
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM_HOST = "127.0.0.1"
UPSTREAM_PORT = 8000
HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "index.html")

CONTROL_CLIENTS = []
CONTROL_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                with open(INDEX, "rb") as fh:
                    body = fh.read()
            except OSError:
                self.send_error(500, "index.html missing")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self._json(200, b'{"ok":true}')
        elif self.path == "/control":
            self._control_stream()
        else:
            self.send_error(404)

    def _control_stream(self):
        """Server-sent events channel the page listens on for prompts to type."""
        q = queue.Queue()
        with CONTROL_LOCK:
            CONTROL_CLIENTS.append(q)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while True:
                try:
                    msg = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")  # keep-alive
                    self.wfile.flush()
                    continue
                payload = json.dumps(msg)
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with CONTROL_LOCK:
                if q in CONTROL_CLIENTS:
                    CONTROL_CLIENTS.remove(q)

    def do_POST(self):
        if self.path == "/say":
            length = int(self.headers.get("Content-Length") or 0)
            try:
                msg = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                self._json(400, b'{"error":"bad json"}')
                return
            with CONTROL_LOCK:
                targets = list(CONTROL_CLIENTS)
            for q in targets:
                q.put(msg)
            self._json(200, json.dumps({"delivered": len(targets)}).encode())
            return

        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)

        conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=600)
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
            )
            resp = conn.getresponse()

            self.send_response(resp.status)
            self.send_header(
                "Content-Type", resp.getheader("Content-Type", "text/event-stream")
            )
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            while True:
                chunk = resp.read1(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser navigated away or hit stop
        except Exception as exc:  # noqa: BLE001 - report to the page
            try:
                self._json(502, ('{"error":{"message":%r}}' % str(exc)).encode())
            except Exception:  # noqa: BLE001
                pass
        finally:
            conn.close()

    def _json(self, code, payload):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    print(f"chat ui: http://127.0.0.1:{port}  -> engine {UPSTREAM_HOST}:{UPSTREAM_PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
