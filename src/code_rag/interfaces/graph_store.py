from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolRef:
    """Lightweight reference returned from graph queries.

    Kept intentionally flat (no Chunk) so the graph layer does not depend on
    the vector store. The caller joins on (path, symbol) when it needs text.
    """

    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class Edge:
    src_path: str
    src_symbol: str
    dst_path: str | None         # None when the callee is unresolved (external / dynamic)
    dst_symbol: str
    kind: str                    # "defines" | "calls" | "imports" | "references" | "contains"


class GraphStore(ABC):
    """Symbol/call/import graph. Keyed by (path, symbol)."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def upsert_symbols(self, symbols: Sequence[SymbolRef]) -> None: ...

    @abstractmethod
    def upsert_edges(self, edges: Sequence[Edge]) -> None: ...

    @abstractmethod
    def delete_by_path(self, path: str) -> None:
        """Remove every symbol and edge whose src_path == path. Used on file delete/rename."""

    @abstractmethod
    def callers_of(self, symbol: str, path: str | None = None) -> list[SymbolRef]: ...

    @abstractmethod
    def callees_of(self, symbol: str, path: str | None = None) -> list[SymbolRef]: ...

    @abstractmethod
    def find_symbol(self, symbol: str, path: str | None = None) -> list[SymbolRef]: ...
