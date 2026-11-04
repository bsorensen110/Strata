#!/usr/bin/env python3
"""Transparent logging proxy: codex -> :8099/v1/* -> Strata :8086. Logs request bodies."""
import http.server, urllib.request, json, sys, os, threading

LOG = os.path.expanduser("~/.hermes/cache/scratch/amo-capture/requests.jsonl")
LOCK = threading.Lock()

class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        with LOCK:
            with open(LOG, "ab") as f:
                f.write(json.dumps({"path": self.path, "body": json.loads(body)}).encode() + b"\n")
        req = urllib.request.Request("http://localhost:8086" + self.path, data=body, method="POST")
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length"):
                req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = r.read()
                self.send_response(r.status)
                self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    def log_message(self, *a):
        pass

if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", 8099), H).serve_forever()
