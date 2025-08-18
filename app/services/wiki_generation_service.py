from pathlib import Path
from uuid import UUID
from typing import Dict, Any, List
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy

from app.services.rag_service import get_project_searcher
from app.utils.logger import logger
from app.agents.wiki_generation_agent import create_wiki_generation_agent
from app.utils.wiki_utils import (
    format_wiki_tree,
    iter_leaf_paths,
    iter_paths,
    load_full_analysis_json,
    page_path_to_str,
    get_project_wiki_dir,
)


# Concurrency for wiki generation
WIKI_MAX_WORKERS = int(os.environ.get("WIKI_MAX_WORKERS", "4"))


def generate_wiki_page(
    project_id: UUID,
    page_title: str,
    page_path: str,
    wiki_tree: Dict[str, Any],
    react_agent: dspy.ReAct = None
) -> str:
    """
    Generates a single wiki page using a DSPy ReAct agent and the RAG search tool.
    If react_agent is provided, it will be reused.
    """
    wiki_context = format_wiki_tree(wiki_tree)
    analysis_raw = load_full_analysis_json(project_id)
    if analysis_raw:
        wiki_context = f"{wiki_context}\n\n=== analysis.json ===\n{analysis_raw}"

    if react_agent is None:
        # Use cached RAG searcher for faster lookups
        try:
            searcher = get_project_searcher(project_id)
        except Exception:
            searcher = None
        agent = create_wiki_generation_agent(project_id, searcher=searcher)
    else:
        agent = react_agent

    result = agent(
        page_title=page_title,
        page_path=page_path,
        wiki_context=wiki_context,
    )
    return result.content if hasattr(result, "content") else str(result)


def _collect_page_titles(wiki_tree: Dict[str, Any]) -> List[str]:
    """
    Flattens the LEAF wiki tree keys into a list of titles.
    Titles may repeat across different branches; used mostly for logging.
    """
    titles: List[str] = []
    for segs in iter_leaf_paths(wiki_tree):
        if segs:
            titles.append(segs[-1])
    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def generate_and_log_wiki_pages(project_id: UUID, wiki_tree: Dict[str, Any], leaves_only: bool = True) -> Dict[str, str]:
    """
    Generates pages for the wiki and logs them.
    By default, only leaf pages are generated (leaves_only=True), so intermediate nodes like
    'rexx', 'jcl', 'cobol', 'copybooks' are skipped when they have children.
    Returns a dict of {title: content}. Note: if leaf titles repeat in different branches,
    later pages overwrite earlier ones in this return mapping.
    """
    if leaves_only:
        titles = _collect_page_titles(wiki_tree)
    else:
        # fallback to all nodes if explicitly requested
        titles = []
        for segs in iter_paths(wiki_tree):
            if segs:
                titles.append(segs[-1])

    logger.info(f"[WikiGen] Starting wiki generation (log-only) for project {project_id}. {len(titles)} titles detected. leaves_only={leaves_only}")

    results: Dict[str, str] = {}
    # Shared cached RAG searcher
    try:
        searcher = get_project_searcher(project_id)
    except Exception:
        searcher = None

    wiki_context = format_wiki_tree(wiki_tree)
    analysis_raw = load_full_analysis_json(project_id)
    if analysis_raw:
        wiki_context = f"{wiki_context}\n\n=== analysis.json ===\n{analysis_raw}"

    if WIKI_MAX_WORKERS <= 1:
        react_agent = create_wiki_generation_agent(project_id, searcher=searcher)
        for idx, title in enumerate(titles, start=1):
            try:
                logger.info(f"[WikiGen] ({idx}/{len(titles)}) Generating page (by title): {title!r}")
                result = react_agent(
                    page_title=title,
                    page_path=title,
                    wiki_context=wiki_context
                )
                content = result.content if hasattr(result, "content") else str(result)
                results[title] = content
                logger.info(
                    f"[WikiGen] Completed (title): {title!r} (chars={len(content)}).\n"
                    f"----- {title} -----\n{content}\n----- END {title} -----"
                )
            except Exception as e:
                logger.exception(f"[WikiGen] Failed to generate page for title {title!r}: {e}")
    else:
        def worker(title: str) -> tuple[str, str]:
            agent = create_wiki_generation_agent(project_id, searcher=searcher)
            result = agent(page_title=title, page_path=title, wiki_context=wiki_context)
            content = result.content if hasattr(result, "content") else str(result)
            return title, content

        with ThreadPoolExecutor(max_workers=WIKI_MAX_WORKERS) as ex:
            futures = {ex.submit(worker, title): title for title in titles}
            for i, fut in enumerate(as_completed(futures), start=1):
                title = futures[fut]
                try:
                    t, content = fut.result()
                    results[t] = content
                    logger.info(
                        f"[WikiGen] Completed (title): {t!r} (chars={len(content)}).\n"
                        f"----- {t} -----\n{content}\n----- END {t} -----"
                    )
                except Exception as e:
                    logger.exception(f"[WikiGen] Failed to generate page for title {title!r}: {e}")

    logger.info(f"[WikiGen] Completed (log-only) wiki generation for project {project_id}. Successful: {len(results)}/{len(titles)}")
    return results


def generate_persist_and_log_wiki(project_id: UUID, wiki_tree: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a page for every LEAF path in the wiki tree and writes each to:
      project_wiki/{project_id}/{path}.md

    Also logs the generated content. Returns a manifest:
      {
        "project_id": "...",
        "base_dir": "project_wiki/<id>",
        "pages": [
            {"path": "a/b/c", "file": "project_wiki/<id>/a/b/c.md", "chars": 1234}
        ]
      }
    """
    paths: List[List[str]] = list(iter_leaf_paths(wiki_tree))
    base_dir = get_project_wiki_dir(project_id)
    base_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[WikiGen] Starting full wiki generation for project {project_id}. {len(paths)} leaf pages detected.")
    logger.info(f"[WikiGen] Output directory: {base_dir.resolve()}")

    # Shared cached RAG searcher
    try:
        searcher = get_project_searcher(project_id)
    except Exception:
        searcher = None

    wiki_context = format_wiki_tree(wiki_tree)
    analysis_raw = load_full_analysis_json(project_id)
    if analysis_raw:
        wiki_context = f"{wiki_context}\n\n=== analysis.json ===\n{analysis_raw}"

    manifest_pages: List[Dict[str, Any]] = []

    if WIKI_MAX_WORKERS <= 1:
        react_agent = create_wiki_generation_agent(project_id, searcher=searcher)
        for idx, segs in enumerate(paths, start=1):
            page_title = segs[-1] if segs else "untitled"
            page_path_str = page_path_to_str(segs)
            try:
                logger.info(f"[WikiGen] ({idx}/{len(paths)}) Generating page: {page_path_str!r}")
                result = react_agent(
                    page_title=page_title,
                    page_path=page_path_str,
                    wiki_context=wiki_context,
                )
                content = result.content if hasattr(result, "content") else str(result)

                # Persist to file path project_wiki/{project_id}/{path}.md
                file_path = (base_dir / Path(*segs)).with_suffix(".md")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                logger.info(
                    f"[WikiGen] Completed page: {page_path_str!r} "
                    f"(chars={len(content)}, file={file_path})\n"
                    f"----- {page_path_str} -----\n{content}\n----- END {page_path_str} -----"
                )

                manifest_pages.append({
                    "path": page_path_str,
                    "file": str(file_path),
                    "chars": len(content),
                })

            except Exception as e:
                logger.exception(f"[WikiGen] Failed to generate/persist page {page_path_str!r}: {e}")
    else:
        def worker(segs: List[str]) -> Dict[str, Any]:
            page_title = segs[-1] if segs else "untitled"
            page_path_str = page_path_to_str(segs)
            agent = create_wiki_generation_agent(project_id, searcher=searcher)

            result = agent(page_title=page_title, page_path=page_path_str, wiki_context=wiki_context)
            content = result.content if hasattr(result, "content") else str(result)

            file_path = (base_dir / Path(*segs)).with_suffix(".md")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(
                f"[WikiGen] Completed page: {page_path_str!r} "
                f"(chars={len(content)}, file={file_path})\n"
                f"----- {page_path_str} -----\n{content}\n----- END {page_path_str} -----"
            )

            return {
                "path": page_path_str,
                "file": str(file_path),
                "chars": len(content),
            }

        with ThreadPoolExecutor(max_workers=WIKI_MAX_WORKERS) as ex:
            futures = {ex.submit(worker, segs): segs for segs in paths}
            for fut in as_completed(futures):
                segs = futures[fut]
                page_path_str = page_path_to_str(segs)
                try:
                    manifest_pages.append(fut.result())
                except Exception as e:
                    logger.exception(f"[WikiGen] Failed to generate/persist page {page_path_str!r}: {e}")

    summary = {
        "project_id": str(project_id),
        "base_dir": str(base_dir),
        "pages": manifest_pages,
        "success_count": len(manifest_pages),
        "total_count": len(paths),
    }
    logger.info(
        f"[WikiGen] Finished full wiki generation for project {project_id}. "
        f"Success: {summary['success_count']}/{summary['total_count']}. "
        f"Base dir: {summary['base_dir']}"
    )
    return summary