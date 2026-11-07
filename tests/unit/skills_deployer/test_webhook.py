"""Test authenticated webhook requests and bounded HTTP handling."""

import contextlib
import hashlib
import hmac
import http.client
import http.server
import socket
import threading
import time

import pytest

from soliplex_concierge.skills_deployer import webhook

SECRET = b"webhook-secret-sentinel"
BODY = b"unparsed-body-sentinel"
SIGNATURE = hmac.new(SECRET, BODY, hashlib.sha256).hexdigest()


@pytest.fixture
def server():
    """Serve webhooks on a loopback port with a fresh wakeup event."""
    event = threading.Event()
    server = webhook.create_server("127.0.0.1:0", SECRET, event)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, event
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "signature, body, status",
    [
        (SIGNATURE, BODY, 202),
        (None, BODY, 401),
        ("wrong-signature-sentinel", BODY, 401),
        (SIGNATURE, b"different-body", 401),
        (SIGNATURE.upper(), BODY, 401),
        ("\xe9", BODY, 401),
        (hmac.new(SECRET, b"", hashlib.sha256).hexdigest(), b"", 202),
        (
            hmac.new(
                SECRET, b"x" * webhook.MAX_BODY_BYTES, hashlib.sha256
            ).hexdigest(),
            b"x" * webhook.MAX_BODY_BYTES,
            202,
        ),
    ],
    ids=[
        "valid",
        "missing",
        "wrong",
        "different-body",
        "uppercase",
        "non-ascii",
        "empty-body",
        "maximum-body",
    ],
)
def test_signature_verifies_raw_body(server, capfd, signature, body, status):
    httpd, event = server
    headers = {}
    if signature is not None:
        headers["X-Gitea-Signature"] = signature
    with contextlib.closing(
        http.client.HTTPConnection(*httpd.server_address, timeout=2)
    ) as conn:
        conn.request("POST", "/hooks/gitea", body, headers)
        response = conn.getresponse()
        assert response.status == status
        assert response.read() == b""
    assert event.is_set() == (status == 202)
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == f"POST /hooks/gitea {status}\n"
    assert SECRET.decode() not in captured.err
    assert SIGNATURE not in captured.err
    assert BODY.decode() not in captured.err


@pytest.mark.parametrize(
    "method, path, length, status",
    [
        ("POST", "/hooks/gitea", str(webhook.MAX_BODY_BYTES + 1), 413),
        ("POST", "/hooks/gitea", "-1", 400),
        ("POST", "/hooks/gitea", "invalid", 400),
        ("GET", "/hooks/gitea", "100", 405),
        ("HEAD", "/hooks/gitea", "100", 405),
        ("PUT", "/hooks/gitea", "100", 405),
        ("ARBITRARY", "/hooks/gitea", "100", 405),
        ("POST", "/other", "100", 404),
        ("GET", "/other", "100", 404),
    ],
)
def test_rejected_requests_do_not_read_body_or_signal(
    server, capfd, method, path, length, status
):
    httpd, event = server
    started = time.monotonic()
    with contextlib.closing(
        http.client.HTTPConnection(*httpd.server_address, timeout=2)
    ) as conn:
        conn.request(
            method, path, b"0123456789", headers={"Content-Length": length}
        )
        response = conn.getresponse()
        assert response.status == status
        assert response.read() == b""
    assert time.monotonic() - started < 1
    assert not event.is_set()
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == f"{method} {path} {status}\n"


@pytest.mark.parametrize("secret", [b"", b" \t\n"])
def test_create_server_rejects_empty_secret(secret):
    with pytest.raises(ValueError, match="secret"):
        webhook.create_server("127.0.0.1:0", secret, threading.Event())


def test_slow_client_times_out_without_signaling_and_next_request_succeeds(
    server,
):
    httpd, event = server
    assert type(httpd) is http.server.HTTPServer
    assert httpd.RequestHandlerClass.timeout == 5
    with contextlib.closing(
        http.client.HTTPConnection(*httpd.server_address, timeout=10)
    ) as conn:
        conn.putrequest("POST", "/hooks/gitea")
        conn.putheader("Content-Length", "100")
        conn.putheader("X-Gitea-Signature", SIGNATURE)
        conn.endheaders()
        started = time.monotonic()
        remaining = 8
        disconnected = False
        while remaining and not disconnected:
            try:
                conn.send(b"x")
                time.sleep(1)
            except OSError:
                disconnected = True
            remaining -= 1
    elapsed = time.monotonic() - started
    assert disconnected
    assert 4.5 <= elapsed < 6.5
    assert not event.is_set()
    with contextlib.closing(
        http.client.HTTPConnection(*httpd.server_address, timeout=2)
    ) as conn:
        conn.request(
            "POST",
            "/hooks/gitea",
            BODY,
            {"X-Gitea-Signature": SIGNATURE},
        )
        response = conn.getresponse()
        assert response.status == 202
        assert response.read() == b""
    assert event.is_set()


def test_short_body_close_handled_before_deadline_without_signaling(server):
    httpd, event = server
    started = time.monotonic()
    with contextlib.closing(
        http.client.HTTPConnection(*httpd.server_address, timeout=2)
    ) as conn:
        conn.putrequest("POST", "/hooks/gitea")
        conn.putheader("Content-Length", "100")
        conn.putheader("X-Gitea-Signature", SIGNATURE)
        conn.endheaders(b"0123456789")
        conn.sock.shutdown(socket.SHUT_WR)
        assert conn.sock.recv(1) == b""
    assert time.monotonic() - started < 1
    assert not event.is_set()


def test_malformed_request_line_returns_400_without_logging_line(
    server, capfd
):
    httpd, event = server
    request_line = "malformed-line-sentinel extra-token another-token HTTP/1.1"
    with socket.create_connection(httpd.server_address, timeout=2) as sock:
        sock.sendall(f"{request_line}\r\n\r\n".encode())
        response = sock.recv(4096)
    assert response.startswith(b"HTTP/1.0 400 ")
    assert not event.is_set()
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == "request error\n- - 400\n"
    for token in request_line.split():
        assert token not in captured.err


def test_missing_length_uses_empty_body(server):
    httpd, event = server
    signature = hmac.new(SECRET, b"", hashlib.sha256).hexdigest()
    with contextlib.closing(
        http.client.HTTPConnection(*httpd.server_address, timeout=2)
    ) as conn:
        conn.putrequest("POST", "/hooks/gitea")
        conn.putheader("X-Gitea-Signature", signature)
        conn.endheaders()
        response = conn.getresponse()
        assert response.status == 202
        assert response.read() == b""
    assert event.is_set()


def test_handler_preserves_missing_attribute_errors(server):
    httpd, _event = server
    handler = object.__new__(httpd.RequestHandlerClass)
    with pytest.raises(AttributeError, match="missing"):
        _ = handler.missing
