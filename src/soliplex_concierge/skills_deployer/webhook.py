"""Wake the reconciler for authenticated Gitea push webhooks."""

import hashlib
import hmac
import http.server
import sys
import threading
import time

MAX_BODY_BYTES = 1024 * 1024


def create_server(
    listen: str, secret: bytes, event: threading.Event
) -> http.server.HTTPServer:
    """Bind a webhook server that signals authenticated requests."""
    if not secret.strip():
        message = "webhook secret must not be empty"
        raise ValueError(message)

    class Handler(http.server.BaseHTTPRequestHandler):
        """Authenticate webhook bodies and signal a reconciliation pass."""

        timeout = 5

        def __getattr__(self, name):
            if name.startswith("do_"):
                return self._handle
            raise AttributeError(name)

        def _handle(self):
            deadline = time.monotonic() + self.timeout
            if self.path != "/hooks/gitea":
                self._respond(404)
                return
            if self.command != "POST":
                self._respond(405)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._respond(400)
                return
            if length < 0:
                self._respond(400)
                return
            if length > MAX_BODY_BYTES:
                self._respond(413)
                return
            body = bytearray()
            try:
                while (
                    len(body) < length
                    and (remaining := deadline - time.monotonic()) > 0
                ):
                    self.connection.settimeout(remaining)
                    if not (chunk := self.rfile.read1(length - len(body))):
                        break
                    body.extend(chunk)
            except (TimeoutError, ConnectionResetError):
                pass
            if len(body) < length:
                return
            signature = self.headers.get("X-Gitea-Signature", "")
            expected = hmac.new(secret, body, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature.encode(), expected.encode()):
                self._respond(401)
                return
            event.set()
            self._respond(202)

        def _respond(self, status):
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_request(self, code="-", size="-"):
            """Pass only the response status to the request logger."""
            self.log_message("%s", code)

        def log_error(self, format, *args):
            """Write a fixed error message without request details."""
            print("request error", file=sys.stderr)

        def log_message(self, format, *args):
            """Write the method, path and status without request contents."""
            command = getattr(self, "command", None) or "-"
            path = getattr(self, "path", None) or "-"
            print(command, path, args[0], file=sys.stderr)

    host, port = listen.rsplit(":", 1)
    return http.server.HTTPServer((host, int(port)), Handler)
