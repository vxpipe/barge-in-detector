"""Annotation UI: uv run python -m barge_in_detector.review."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from barge_in_detector.review.store import ReviewStore


def make_handler(store, allowed_hosts=()):
    class Handler(BaseHTTPRequestHandler):
        def valid_host(self):
            return self.headers.get("Host", "").split(":")[0] in {
                "localhost",
                "127.0.0.1",
                *allowed_hosts,
            }

        def respond(self, status, body, content_type="application/json; charset=utf-8"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self.valid_host():
                self.respond(403, '{"error":"Use a configured review server address"}')
                return
            if self.path == "/":
                self.respond(
                    200,
                    Path(__file__).with_name("index.html").read_bytes(),
                    "text/html; charset=utf-8",
                )
            elif self.path == "/api/cases":
                self.respond(200, json.dumps(store.snapshot(), ensure_ascii=False))
            elif self.path == "/export.csv":
                self.respond(200, store.export_csv(), "text/csv; charset=utf-8")
            else:
                self.respond(404, '{"error":"Not found"}')

        def do_POST(self):
            if not self.valid_host():
                self.respond(403, '{"error":"Use a configured review server address"}')
                return
            if self.path != "/api/review":
                self.respond(404, '{"error":"Not found"}')
                return
            origin = self.headers.get("Origin")
            if (origin and origin != "http://" + self.headers.get("Host", "")) or self.headers.get(
                "Content-Type"
            ) != "application/json":
                self.respond(403, '{"error":"Use the local review page to save"}')
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(size))
                record = store.save(
                    data["case_id"], data["decision"], data["notes"], data["revision"]
                )
                self.respond(200, json.dumps(record))
            except (ValueError, KeyError, TypeError) as error:
                self.respond(400, json.dumps({"error": str(error)}))
            except OSError:
                self.respond(
                    500,
                    '{"error":"Could not save to disk. Your decision has not been saved. Try again."}',
                )

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument(
        "--allow-host", action="append", default=[], help="Additional allowed hostname or IP"
    )
    parser.add_argument(
        "--source", type=Path, default=Path("data/evaluation/english-challenge-v1.csv")
    )
    parser.add_argument(
        "--reviews", type=Path, default=Path("data/evaluation/reviews/english-challenge-v1.json")
    )
    args = parser.parse_args()
    store = ReviewStore(args.source, args.reviews)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store, args.allow_host))
    print(f"Review UI listening on {args.host}:{server.server_port}", flush=True)
    print(f"Decisions saved to {args.reviews}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
