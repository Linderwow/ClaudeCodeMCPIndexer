from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import chromadb

from code_rag.interfaces.vector_store import VectorStore
from code_rag.logging import get
from code_rag.models import Chunk, ChunkKind, IndexMeta, SearchHit
from code_rag.version import INDEX_SCHEMA_VERSION

log = get(__name__)


class ChromaVectorStore(VectorStore):
    """Persistent Chroma collection keyed by Chunk.id.

    IndexMeta is stored in a sibling JSON file (not inside Chroma) so we can
    validate it WITHOUT loading embeddings — a mismatch aborts before we
    touch the collection.
    """

    def __init__(self, persist_dir: Path, collection: str, meta_path: Path) -> None:
        self._dir = persist_dir
        self._collection_prefix = collection
        # Effective collection name is resolved in open() once we know the
        # embedder model/dim — see `_resolved_name`.
        self._collection_name: str = collection
        self._meta_path = meta_path
        self._client: Any = None
        self._coll: Any = None
        self._meta: IndexMeta | None = None

    @staticmethod
    def _resolved_name(prefix: str, meta: IndexMeta) -> str:
        """Namespace the Chroma collection by embedder model + dim.

        Defense-in-depth against IndexMeta tampering / deletion: even if the
        meta file is missing or hand-edited, vectors embedded with a different
        model land in a DIFFERENT collection and can't silently commingle.
        """
        import hashlib
        key = f"{meta.embedder_kind}|{meta.embedder_model}|{meta.embedder_dim}".encode()
        tag = hashlib.blake2b(key, digest_size=4).hexdigest()
        return f"{prefix}_{tag}"

    # ---- lifecycle ----------------------------------------------------------

    def open(self, meta: IndexMeta) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._validate_or_write_meta(meta)
        self._collection_name = self._resolved_name(self._collection_prefix, meta)
        self._client = chromadb.PersistentClient(path=str(self._dir))
        self._coll = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "chroma.opened",
            dir=str(self._dir),
            collection=self._collection_name,
            count=self._coll.count(),
        )

    def close(self) -> None:
        self._coll = None
        self._client = None

    # ---- meta enforcement ---------------------------------------------------

    def _validate_or_write_meta(self, meta: IndexMeta) -> None:
        if self._meta_path.exists():
            on_disk = IndexMeta.model_validate_json(self._meta_path.read_text("utf-8"))
            mismatches = []
            if on_disk.schema_version != meta.schema_version:
                mismatches.append(f"schema_version {on_disk.schema_version} != {meta.schema_version}")
            if on_disk.embedder_kind != meta.embedder_kind:
                mismatches.append(f"embedder_kind {on_disk.embedder_kind!r} != {meta.embedder_kind!r}")
            if on_disk.embedder_model != meta.embedder_model:
                mismatches.append(f"embedder_model {on_disk.embedder_model!r} != {meta.embedder_model!r}")
            if on_disk.embedder_dim != meta.embedder_dim:
                mismatches.append(f"embedder_dim {on_disk.embedder_dim} != {meta.embedder_dim}")
            if mismatches:
                raise RuntimeError(
                    "Index metadata mismatch. Delete the data directory to force a full rebuild.\n  "
                    + "\n  ".join(mismatches)
                )
            # Touch updated_at, keep created_at.
            meta = IndexMeta(
                schema_version=on_disk.schema_version,
                embedder_kind=on_disk.embedder_kind,
                embedder_model=on_disk.embedder_model,
                embedder_dim=on_disk.embedder_dim,
                created_at=on_disk.created_at,
                updated_at=datetime.now(UTC).isoformat(),
            )
        self._meta = meta
        self._meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def build_meta(embedder_kind: str, embedder_model: str, embedder_dim: int) -> IndexMeta:
        now = datetime.now(UTC).isoformat()
        return IndexMeta(
            schema_version=INDEX_SCHEMA_VERSION,
            embedder_kind=embedder_kind,
            embedder_model=embedder_model,
            embedder_dim=embedder_dim,
            created_at=now,
            updated_at=now,
        )

    # ---- writes -------------------------------------------------------------

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if not chunks:
            return
        assert len(chunks) == len(embeddings), "chunk/embedding count mismatch"
        coll = self._require()
        ids = [c.id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [self._meta_of(c) for c in chunks]
        # Chroma's upsert type is narrow (str|int|float|bool|list|None only) — our
        # metadata honors that, but the declared Mapping types don't accept our
        # concrete dict[str, Any]. Cast once at the boundary.
        coll.upsert(
            ids=ids,
            documents=docs,
            metadatas=cast(Any, metas),
            embeddings=cast(Any, [list(v) for v in embeddings]),
        )

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        self._require().delete(ids=list(ids))

    def delete_by_path(self, path: str) -> int:
        coll = self._require()
        before = int(coll.count())
        coll.delete(where=cast(Any, {"path": path}))
        after = int(coll.count())
        return max(0, before - after)

    # ---- reads --------------------------------------------------------------

    def query(
        self,
        embedding: Sequence[float],
        k: int,
        where: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        coll = self._require()
        res = coll.query(
            query_embeddings=cast(Any, [list(embedding)]),
            n_results=k,
            where=cast(Any, where) if where else None,
        )
        hits: list[SearchHit] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for i, _id in enumerate(ids):
            m: dict[str, Any] = dict(metas[i] or {}) if i < len(metas) else {}
            chunk = Chunk(
                id=_id,
                repo=str(m.get("repo", "")),
                path=str(m.get("path", "")),
                language=str(m.get("language", "")),
                symbol=(str(m["symbol"]) if m.get("symbol") else None),
                kind=ChunkKind(str(m.get("kind", "other"))),
                start_line=int(cast(int, m.get("start_line", 1))),
                end_line=int(cast(int, m.get("end_line", 1))),
                text=(docs[i] or "") if i < len(docs) else "",
            )
            # cosine distance -> similarity in (approx) [0, 1]. Defensive against
            # Chroma ever returning mismatched column lengths.
            score = 1.0 - float(dists[i]) if i < len(dists) else 0.0
            hits.append(SearchHit(
                chunk=chunk,
                score=score,
                source="vector",
                match_reason=f"vector cosine {score:.3f}",
            ))
        return hits

    def count(self) -> int:
        return int(self._require().count())

    def list_paths(self) -> set[str]:
        """All distinct paths currently in the store.

        Phase 26: used by the eval gate to filter ground-truth cases down to
        paths actually in the index. Mined transcripts reference files Claude
        opened from worktree clones, deleted files, or roots not currently
        configured — including them artificially depresses recall.

        Paginated because Chroma's `get` materializes one SQLite parameter per
        row internally; on 80K+ chunk collections an unpaginated call hits
        SQLITE_MAX_VARIABLE_NUMBER and raises "too many SQL variables". 5000
        per page keeps us well under the limit while only making ~20 round
        trips on a 100K-chunk index.
        """
        coll = self._require()
        out: set[str] = set()
        page = 5000
        offset = 0
        while True:
            res = coll.get(include=["metadatas"], limit=page, offset=offset)
            metas = res.get("metadatas") or []
            if not metas:
                break
            for m in metas:
                if not m:
                    continue
                p = m.get("path")
                if isinstance(p, str) and p:
                    out.add(p)
            if len(metas) < page:
                break
            offset += page
        return out

    # ---- internal -----------------------------------------------------------

    def _require(self) -> Any:
        if self._coll is None:
            raise RuntimeError("ChromaVectorStore not open; call .open(meta) first")
        return self._coll

    @staticmethod
    def _meta_of(c: Chunk) -> dict[str, Any]:
        return {
            "repo": c.repo,
            "path": c.path,
            "language": c.language,
            "symbol": c.symbol or "",
            "kind": c.kind.value,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "n_chars": c.n_chars,
        }
