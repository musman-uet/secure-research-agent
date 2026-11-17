"""Who is calling? Bearer token -> user.

Demo: tokens are listed in data/tokens.json and map to users in
data/directory.json, so n8n or Claude can connect with a header like
`Authorization: Bearer demo-bilal-...`.

Production: replace DemoTokenVerifier with a verifier that validates a
Microsoft Entra ID access token (signature, issuer, audience, expiry) and
takes the user's object ID and `groups` claim from it. Nothing else changes:
the agent only ever receives a resolved User.
"""

from __future__ import annotations

import hmac
import json
from pathlib import Path

from mcp.server.auth.provider import AccessToken

from securerag.directory import JsonDirectory, UnknownUserError
from securerag.models import User

SCOPE = "research"


class TokenStore:
    def __init__(self, path: Path, directory: JsonDirectory):
        self._tokens: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))["tokens"]
        self.directory = directory

    def user_for(self, token: str | None) -> User | None:
        if not token:
            return None
        for known, user_id in self._tokens.items():
            if hmac.compare_digest(known.encode(), token.encode()):  # constant-time compare
                try:
                    return self.directory.get_user(user_id)
                except UnknownUserError:
                    return None
        return None


class DemoTokenVerifier:
    """MCP SDK TokenVerifier backed by the TokenStore."""

    def __init__(self, store: TokenStore):
        self.store = store

    async def verify_token(self, token: str) -> AccessToken | None:
        user = self.store.user_for(token)
        if user is None:
            return None
        return AccessToken(token=token, client_id="demo-client", scopes=[SCOPE], subject=user.user_id)
