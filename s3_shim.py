#!/usr/bin/env python3
"""Minimal local S3-compatible store for the Neuronpedia graph flow.

The webapp presigns PUT URLs (AWS SDK), the graph server PUTs the generated graph JSON,
the webapp GETs it back and the browser later GETs GraphMetadata.url. This shim serves
that exact flow on http://127.0.0.1:9000 with filesystem storage: signatures and auth
are ignored (local, loopback only). Files land under s3_data/<bucket>/<key>.

Point the webapp at it with S3_ENDPOINT=http://127.0.0.1:9000 (see the S3Client sites).
"""
import os
import posixpath
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s3_data")
PORT = 9000


def safe_path(url_path: str) -> str | None:
    """Map /bucket/key to a filesystem path; reject traversal."""
    parts = [p for p in url_path.split("?")[0].split("/") if p]
    if not parts:
        return None
    norm = posixpath.normpath("/".join(parts)).split("/")
    if any(p == ".." or p.startswith("..") for p in norm):
        return None
    return os.path.join(ROOT, *norm)


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, PUT, DELETE, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_PUT(self):
        path = safe_path(self.path)
        if path is None:
            self.send_error(400)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(body)
        self.send_response(200)
        self._cors()
        self.send_header("ETag", '"local"')
        self.end_headers()

    def do_GET(self):
        path = safe_path(self.path)
        if path is None or not os.path.isfile(path):
            self.send_error(404)
            return
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        path = safe_path(self.path)
        if path is None or not os.path.isfile(path):
            self.send_error(404)
            return
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(os.path.getsize(path)))
        self.end_headers()

    def do_DELETE(self):
        path = safe_path(self.path)
        if path is None or not os.path.isfile(path):
            self.send_error(404)
            return
        os.remove(path)
        self.send_response(204)
        self._cors()
        self.end_headers()

    def log_message(self, fmt, *args):
        print(f"[s3-shim] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    print(f"[s3-shim] serving {ROOT} on http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
