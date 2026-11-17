from __future__ import annotations

from typing import Literal, Protocol

from securerag.models import Chunk, SearchHit

SearchMode = Literal["keyword", "vector", "hybrid"]


class SearchBackend(Protocol):
    def index(self, chunks: list[Chunk]) -> None: ...

    def search(
        self,
        query: str,
        query_vector: list[float] | None,
        principals: list[str],
        top_k: int = 5,
        mode: SearchMode = "hybrid",
    ) -> list[SearchHit]:
        """Return the top_k chunks the given principals are allowed to read.

        Implementations MUST apply the permission filter inside the search
        engine (pre-filter), never by post-filtering a larger result list.
        An empty principal list returns nothing.
        """
        ...
