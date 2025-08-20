from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from datetime import datetime, timezone
from bisect import bisect_right
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import faiss

from app.utils.logger import logger
from app.config.llm_config import embedder

PROJECT_STORAGE_PATH = Path("project_storage")
ANALYSIS_BASE_PATH = Path("project_analysis")

# Word-based chunking configuration
CHUNK_WORDS = int(os.environ.get("RAG_CHUNK_WORDS", "350"))
CHUNK_WORD_OVERLAP = int(os.environ.get("RAG_CHUNK_WORD_OVERLAP", "100"))

EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "64"))
MAX_WORKERS = int(os.environ.get("RAG_MAX_WORKERS", "8"))

EXT_LANG_MAP = {
    ".jcl": "jcl",
    ".cbl": "cobol",
    ".cob": "cobol",
    ".cpy": "copybook",
    ".rex": "rexx",
    ".rexx": "rexx",
    ".txt": "text",
    ".md": "markdown",
}


# ---------------- Path Helpers ----------------
def _get_project_source_path(project_id: UUID) -> Path:
    return PROJECT_STORAGE_PATH / str(project_id)


def _get_project_rag_dir(project_id: UUID) -> Path:
    return ANALYSIS_BASE_PATH / str(project_id) / "rag"


def get_faiss_index_path(project_id: UUID) -> Path:
    return _get_project_rag_dir(project_id) / "faiss.index"


def get_faiss_meta_path(project_id: UUID) -> Path:
    return _get_project_rag_dir(project_id) / "index_meta.json"


def get_embedding_status_path(project_id: UUID) -> Path:
    return _get_project_rag_dir(project_id) / "status.json"


# ---------------- Status Helpers ----------------
def _utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_status(
    project_id: UUID, status: str, extra: Optional[Dict[str, Any]] = None
) -> None:
    rag_dir = _get_project_rag_dir(project_id)
    rag_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "status": status,  # pending | embedding | ready | failed | stale
        "updated_at": _utc_now_iso(),
    }
    if extra:
        payload.update(extra)
    with open(get_embedding_status_path(project_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_embedding_status(project_id: UUID) -> Dict[str, Any]:
    p = get_embedding_status_path(project_id)
    if not p.exists():
        return {"status": "pending"}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------- File Helpers ----------------
def _iter_project_text_files(project_id: UUID) -> List[Path]:
    src = _get_project_source_path(project_id)
    if not src.exists():
        return []
    patterns = ["*.jcl", "*.cbl", "*.cob", "*.cpy", "*.rex*", "*.txt", "*.md"]
    paths = set()
    for pat in patterns:
        paths.update(src.rglob(pat))
    # Deduplicate and provide stable order
    return sorted(paths, key=lambda p: str(p).lower())


def _read_text(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


# ---------------- Chunking Helpers (word-based) ----------------
def _chunk_text_words(
    text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_WORD_OVERLAP,
) -> List[Tuple[int, int, int]]:
    """
    Split text into chunks of chunk_words words with overlap_words overlap.
    Returns list of (char_start, char_end, tokens) where tokens == number of words in the chunk.
    """
    if not text or not text.strip():
        return []

    chunk_words = max(1, chunk_words)
    overlap_words = max(0, min(overlap_words, chunk_words - 1))

    # Find word spans (non-whitespace sequences)
    words = list(re.finditer(r"\S+", text))
    n = len(words)
    if n == 0:
        return []

    spans: List[Tuple[int, int, int]] = []
    start_idx = 0
    while start_idx < n:
        end_idx = min(n, start_idx + chunk_words)
        s_char = words[start_idx].start()
        e_char = words[end_idx - 1].end()
        tokens = end_idx - start_idx
        spans.append((s_char, e_char, tokens))

        if end_idx >= n:
            break
        next_start = max(0, end_idx - overlap_words)
        if next_start <= start_idx:
            next_start = start_idx + 1
        start_idx = next_start

    return spans


# ---------------- Line Index Helpers ----------------
def _build_line_index(text: str) -> List[int]:
    # return positions of '\n'
    return [i for i, ch in enumerate(text) if ch == "\n"]


def _span_to_lines(newline_pos: List[int], start: int, end: int) -> Tuple[int, int]:
    # 1-based line numbers
    start_line = bisect_right(newline_pos, start - 1) + 1
    end_line = bisect_right(newline_pos, end - 1) + 1
    return start_line, end_line


# ---------------- Embedding Helpers ----------------
def _normalize_embeddings(embs: np.ndarray) -> np.ndarray:
    embs = embs.astype("float32", copy=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    return embs / norms


def _embed_texts(texts: List[str]) -> np.ndarray:
    """
    Retained for potential ad-hoc use (not used in main build now).
    WARNING: Allocates full matrix in memory.
    """
    if not texts:
        return np.zeros((0, 1), dtype="float32")
    vecs: List[np.ndarray] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        arr = embedder(batch)  # returns numpy array
        vecs.append(arr)
    embs = np.vstack(vecs)
    return _normalize_embeddings(embs)


def _build_faiss_index_stream(texts: List[str]) -> faiss.IndexFlatIP:
    """
    Stream embeddings batch-by-batch directly into a FAISS index to avoid
    holding an (N x D) float32 matrix in memory at once.
    """
    index: Optional[faiss.IndexFlatIP] = None
    total = len(texts)
    if total == 0:
        # Return empty 1-dim index to keep downstream simple
        return faiss.IndexFlatIP(1)
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        if not batch:
            continue
        embs = embedder(batch)
        embs = _normalize_embeddings(embs)
        if index is None:
            index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        if (i // EMBED_BATCH_SIZE) % 10 == 0:
            logger.debug(f"[RAG] Embedded {min(i + len(batch), total)}/{total} chunks")
    return index if index is not None else faiss.IndexFlatIP(1)


# ---------------- Up-to-date Check ----------------
def embeddings_up_to_date(project_id: UUID) -> bool:
    meta_path = get_faiss_meta_path(project_id)
    if not meta_path.exists():
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        src_root = _get_project_source_path(project_id)
        files_map: Dict[str, Dict[str, Any]] = meta.get("files", {})

        if files_map:
            # Compare current files against recorded files
            current_paths = _iter_project_text_files(project_id)
            current_set = {p.relative_to(src_root).as_posix() for p in current_paths}
            if set(files_map.keys()) != current_set:
                return False
            for rel, info in files_map.items():
                path = src_root / Path(rel)
                if not path.exists() or _file_sha256(path) != info.get("sha256", ""):
                    return False
            return True
        else:
            # Fallback using per-chunk items (slower)
            seen: Dict[str, bool] = {}
            for item in meta.get("items", []):
                rel = item["file"]
                if rel in seen:
                    continue
                seen[rel] = True
                path = src_root / Path(rel)
                if not path.exists() or _file_sha256(path) != item.get("sha256", ""):
                    return False
            current_paths = _iter_project_text_files(project_id)
            current_set = {p.relative_to(src_root).as_posix() for p in current_paths}
            if set(seen.keys()) != current_set:
                return False
            return True
    except Exception:
        return False

# ---------------- Parallel File Processing ----------------
def _process_file(path: Path, src_root: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    rel = path.relative_to(src_root).as_posix()
    text = _read_text(path)
    if not text.strip():
        return [], []

    spans = _chunk_text_words(text)
    newline_pos = _build_line_index(text)

    # Determine if file is code vs documentation by extension
    ext = path.suffix.lower()
    is_code = ext not in (".txt", ".md")

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for (s, e, tokens) in spans:
        chunk = text[s:e]
        if not chunk.strip():
            continue
        line_start, line_end = _span_to_lines(newline_pos, s, e)
        texts.append(chunk)
        metas.append(
            {
                "file": rel,
                "char_start": s,
                "char_end": e,
                "tokens": tokens,        # number of words in this chunk
                "is_code": is_code,      # True for code files, False for docs
                "line_start": line_start,
                "line_end": line_end,
            }
        )
    return texts, metas

# ---------------- Main Build Function ----------------
def build_embeddings_for_project(project_id: UUID) -> None:
    logger.info(f"[RAG] Building/updating embeddings for {project_id}")
    _write_status(
        project_id,
        "embedding",
        {"started_at": _utc_now_iso()},
    )

    try:
        rag_dir = _get_project_rag_dir(project_id)
        rag_dir.mkdir(parents=True, exist_ok=True)

        faiss_path = get_faiss_index_path(project_id)
        meta_path = get_faiss_meta_path(project_id)

        # Load existing meta (if any)
        existing_items: List[Dict[str, Any]] = []
        existing_files: Dict[str, Dict[str, Any]] = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
                existing_items = prev.get("items", [])
                existing_files = prev.get("files", {})

        files = _iter_project_text_files(project_id)
        src_root = _get_project_source_path(project_id)

        # Compute current file hashes (normalize rel paths to POSIX)
        current_files: Dict[str, Dict[str, Any]] = {}
        for p in files:
            rel = p.relative_to(src_root).as_posix()
            current_files[rel] = {"sha256": _file_sha256(p), "mtime": int(p.stat().st_mtime)}

        # Detect changes: additions, deletions, or hash changes
        existing_set = set(existing_files.keys())
        current_set = set(current_files.keys())
        deletions = len(existing_set - current_set) > 0
        additions = len(current_set - existing_set) > 0
        changes = any(
            existing_files.get(rel, {}).get("sha256") != data["sha256"]
            for rel, data in current_files.items()
            if rel in existing_set
        )
        up_to_date = meta_path.exists() and not deletions and not additions and not changes

        if up_to_date:
            vectors = len(existing_items)
            _write_status(
                project_id,
                "ready",
                {
                    "vectors": vectors,
                    "files": len(files),
                    "completed_at": _utc_now_iso(),
                    "index_path": str(faiss_path) if faiss_path.exists() else None,
                },
            )
            logger.info(f"[RAG] No changes detected for project {project_id}.")
            return

        # Rebuild the index from scratch to keep FAISS ids aligned with metadata
        new_texts: List[str] = []
        new_metas: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_process_file, path, src_root): path for path in files}
            for future in as_completed(futures):
                texts, metas = future.result()
                new_texts.extend(texts)
                new_metas.extend(metas)

        if not new_texts:
            # No embeddable content (e.g., empty files only)
            if faiss_path.exists():
                try:
                    faiss_path.unlink()
                except Exception:
                    pass
            meta = {
                "dimension": 0,
                "count": 0,
                "project_id": str(project_id),
                "created_at": _utc_now_iso(),
                "items": [],
                "files": current_files,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            _write_status(
                project_id,
                "ready",
                {
                    "vectors": 0,
                    "files": len(files),
                    "completed_at": _utc_now_iso(),
                    "index_path": None,
                },
            )
            logger.info(f"[RAG] No embeddable content for project {project_id}.")
            return

        # Stream embeddings directly into FAISS to reduce peak RAM
        index = _build_faiss_index_stream(new_texts)
        faiss.write_index(index, str(faiss_path))

        meta = {
            "dimension": index.d,
            "count": len(new_metas),
            "project_id": str(project_id),
            "created_at": _utc_now_iso(),
            "items": new_metas,     # per-chunk, 1:1 with FAISS ids
            "files": current_files, # per-file sha/mtime map for change detection
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        _write_status(
            project_id,
            "ready",
            {
                "vectors": len(new_metas),
                "files": len(files),
                "completed_at": _utc_now_iso(),
                "index_path": str(faiss_path),
            },
        )
        logger.info(
            f"[RAG] Rebuilt embeddings (streamed): {len(new_metas)} vectors for project {project_id}"
        )

    except Exception as e:
        logger.error(
            f"[RAG] Failed to build embeddings for project {project_id}: {e}",
            exc_info=True,
        )
        _write_status(
            project_id,
            "failed",
            {"error": str(e), "completed_at": _utc_now_iso()},
        )

# ---------------- Retrieval (Cached and Thread-safe) ----------------
class _ProjectSearcher:
    """
    Thread-safe cached searcher that keeps FAISS index, metadata and file cache
    in memory, and auto-reloads if underlying files change.
    """
    def __init__(self, project_id: UUID, faiss_path: Path, meta_path: Path):
        self.project_id = project_id
        self.faiss_path = faiss_path
        self.meta_path = meta_path
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if not self.faiss_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("FAISS index or metadata not found.")

        self.index = faiss.read_index(str(self.faiss_path))
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.src_root = _get_project_source_path(self.project_id)
        self.file_cache: Dict[str, str] = {}
        self._mtimes = self._current_mtimes()

    def _current_mtimes(self) -> Tuple[int, int]:
        return (
            int(self.faiss_path.stat().st_mtime),
            int(self.meta_path.stat().st_mtime),
        )

    def _stale(self) -> bool:
        try:
            return self._current_mtimes() != self._mtimes
        except Exception:
            # If we cannot stat, assume stale to trigger reload attempt
            return True

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            if self._stale():
                # Try to reload if index/meta changed on disk
                self._load()

            query_emb = _normalize_embeddings(embedder([query]))
            scores, ids = self.index.search(query_emb, top_k)

            results: List[Dict[str, Any]] = []
            items = self.meta.get("items", [])
            for idx, score in zip(ids[0], scores[0]):
                if 0 <= idx < len(items):
                    item = items[idx]
                    rel: str = item["file"]
                    if rel not in self.file_cache:
                        try:
                            self.file_cache[rel] = _read_text(self.src_root / rel)
                        except Exception:
                            self.file_cache[rel] = ""
                    text = self.file_cache.get(rel, "")
                    s = int(item.get("char_start", 0))
                    e = int(item.get("char_end", 0))
                    snippet = text[s:e] if 0 <= s <= e <= len(text) else ""
                    title = f"{rel} L{item.get('line_start', '?')}-{item.get('line_end', '?')}"
                    results.append(
                        {
                            "score": float(score),
                            "file": rel,
                            "line_start": item.get("line_start"),
                            "line_end": item.get("line_end"),
                            "is_code": item.get("is_code"),
                            "title": title,
                            "content": snippet,
                        }
                    )
            return results


_SEARCHERS: Dict[str, _ProjectSearcher] = {}
_SEARCHERS_LOCK = threading.RLock()


def get_project_searcher(project_id: UUID) -> _ProjectSearcher:
    key = str(project_id)
    with _SEARCHERS_LOCK:
        searcher = _SEARCHERS.get(key)
        if searcher is None:
            searcher = _ProjectSearcher(
                project_id, get_faiss_index_path(project_id), get_faiss_meta_path(project_id)
            )
            _SEARCHERS[key] = searcher
        return searcher


def clear_project_searcher_cache(project_id: Optional[UUID] = None) -> None:
    """
    Clear cached searchers. If project_id is None, clear all.
    """
    with _SEARCHERS_LOCK:
        if project_id is None:
            _SEARCHERS.clear()
        else:
            _SEARCHERS.pop(str(project_id), None)


def search_project_cached(project_id: UUID, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Cached, thread-safe search API.
    """
    return get_project_searcher(project_id).search(query, top_k)


# Backwards-compatible API that now uses the cached searcher
def search_project(project_id: UUID, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    return search_project_cached(project_id, query, top_k)