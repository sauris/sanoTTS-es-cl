#!/usr/bin/env python3
"""Local server for the demo folder with cache disabled.

python -m http.server lets the browser heuristically cache index.html, which
stale-tabs the demo across fixes. This server sends `Cache-Control: no-store`
for everything and registers the right MIME types.
"""
import http.server
import socketserver

PORT = 8178


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".data": "application/octet-stream",
        ".js": "text/javascript",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"serving demo on http://localhost:{PORT} (no-store)")
    httpd.serve_forever()
