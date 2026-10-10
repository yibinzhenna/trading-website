"""
Request-size limit, enforced before anything reads the body.

FastAPI parses a JSON body in full before validation or any dependency runs,
so neither the schema nor the rate limiter can stop a huge one. Measured on
this app: one 18 MB request held the server for 19 s and peaked at 500 MB
of memory — on a host with 512 MB. One anonymous request could crash it.

Every legitimate request here is under 1 KB. This middleware sits outermost
and refuses anything larger than the limit: from Content-Length when the
client declares one, and by counting bytes as they arrive when it does not
(chunked uploads), so an omission does not get a body through.
"""

import json

from api import deps


async def _reject(send, status, detail):
    body = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                            (b"connection", b"close")]})
    await send({"type": "http.response.body", "body": body})


class BodySizeLimit:
    """Pure ASGI, so the limit applies to the raw byte stream rather than to
    a body some other layer has already buffered."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = deps.settings.max_body_bytes
        too_big = f"Request body too large (limit {limit:,} bytes)."

        for name, value in scope.get("headers", ()):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    return await _reject(send, 400, "Invalid Content-Length.")
                if declared > limit:
                    return await _reject(send, 413, too_big)

        # Undeclared length: count as it arrives. Past the limit, stop
        # handing bytes to the app (an empty final chunk ends its read) and
        # replace whatever it answers with a 413. Raising instead does not
        # work: FastAPI turns any error during a body read into its own 400.
        received = 0
        exceeded = False

        async def counted_receive():
            nonlocal received, exceeded
            if exceeded:
                return {"type": "http.request", "body": b"", "more_body": False}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    return {"type": "http.request", "body": b"",
                            "more_body": False}
            return message

        replaced = False

        async def guarded_send(message):
            nonlocal replaced
            if not exceeded:
                return await send(message)
            if message["type"] == "http.response.start" and not replaced:
                replaced = True
                await _reject(send, 413, too_big)
            # Drop the app's own response to a body it never fully got.

        await self.app(scope, counted_receive, guarded_send)
