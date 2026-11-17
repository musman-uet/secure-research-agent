"""Resolve a user to the groups they belong to.

Production: read the user's group memberships from the Entra ID token
(`groups` claim) or Microsoft Graph `/me/transitiveMemberOf`.
Demo/tests: a JSON file of users and groups.
"""

from __future__ import annotations

import json
from pathlib import Path

from securerag.models import User


class UnknownUserError(KeyError):
    pass


class JsonDirectory:
    def __init__(self, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        self._users: dict[str, list[str]] = data["users"]

    def get_user(self, user_id: str) -> User:
        if user_id not in self._users:
            raise UnknownUserError(user_id)
        return User(user_id=user_id, groups=list(self._users[user_id]))

    def all_users(self) -> list[User]:
        return [self.get_user(u) for u in self._users]
