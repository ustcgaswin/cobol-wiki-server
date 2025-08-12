from pathlib import Path
from uuid import UUID
from typing import Dict, Any, List, Iterable
import dspy

from app.services.rag_service import search_project
from app.utils.logger import logger

# Root for persisted wiki output relative to the current workspace
PROJECT_WIKI_BASE_PATH = Path("project_wiki")
ANALYSIS_BASE_PATH = Path("project_analysis")


def _get_project_analysis_json_path(project_id: UUID) -> Path:
    """
    Returns project_analysis/<id>/analysis/analysis.json without importing analysis_service (avoid cycles).
    """
    return ANALYSIS_BASE_PATH / str(project_id) / "analysis" / "analysis.json"


def _load_full_analysis_json(project_id: UUID) -> str:
    """
    Loads the raw analysis.json content as a string; returns "" if missing.
    """
    p = _get_project_analysis_json_path(project_id)
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
    except Exception:
        return ""


def _make_rag_search_fn(project_id: UUID):
    """
    Returns a callable tool that searches the project's RAG index and formats results.
    """
    def rag_search(query: str = "", top_k: int = 20, **kwargs) -> str:
        # Be tolerant of different call styles from the agent.
        q = query or kwargs.get("query") or kwargs.get("input") or kwargs.get("text") or ""
        q = str(q).strip()
        try:
            tk = int(kwargs.get("top_k", top_k))
        except Exception:
            tk = top_k

        if not q:
            return "No query provided."

        try:
            results = search_project(project_id, q, tk)
        except FileNotFoundError:
            return "No RAG index found."
        except Exception as e:
            return f"RAG search error: {e}"

        if not results:
            return "No results found."

        lines: List[str] = []
        for i, r in enumerate(results):
            if isinstance(r, dict):
                title = r.get("title", "")
                content = r.get("content", "")
            else:
                title = ""
                content = str(r)
            lines.append(f"[{i+1}] {title}\n{content}")
        return "\n\n".join(lines)
    return rag_search


def _make_mermaid_fn():
    """
    Returns a callable tool that wraps a Mermaid diagram body in fences.
    """
    def mermaid(description: str = "", **kwargs) -> str:
        # Accept missing or differently named inputs.
        desc = description or kwargs.get("description") or kwargs.get("input") or kwargs.get("text") or ""
        desc = str(desc).strip()
        if not desc:
            return "Mermaid description is empty."
        if desc.startswith("```mermaid"):
            return desc
        return f"```mermaid\n{desc}\n```"
    return mermaid


def _format_wiki_tree(wiki_tree: Dict[str, Any]) -> str:
    """
    Converts the wiki tree dict into a readable string for context.
    """
    def recurse(tree, depth=0):
        lines = []
        for key, value in (tree or {}).items():
            lines.append("  " * depth + f"- {key}")
            if isinstance(value, dict):
                lines.extend(recurse(value, depth + 1))
        return lines

    return "\n".join(recurse(wiki_tree or {}))


def _iter_paths(wiki_tree: Dict[str, Any]) -> Iterable[List[str]]:
    """
    Yields all page paths as lists of segments, e.g., ["jcl", "file-a", "job-1"].
    """
    def recurse(tree: Dict[str, Any], prefix: List[str]):
        for key, value in (tree or {}).items():
            current = prefix + [str(key)]
            yield current
            if isinstance(value, dict):
                yield from recurse(value, current)

    yield from recurse(wiki_tree or {}, [])


def _iter_leaf_paths(wiki_tree: Dict[str, Any]) -> Iterable[List[str]]:
    """
    Yields only leaf page paths (nodes with no children).
    A leaf is either a non-dict value or an empty dict {}.
    """
    def recurse(tree: Dict[str, Any], prefix: List[str]):
        for key, value in (tree or {}).items():
            current = prefix + [str(key)]
            if isinstance(value, dict):
                if value:  # has children -> dive deeper
                    yield from recurse(value, current)
                else:      # empty dict -> leaf
                    yield current
            else:
                yield current

    yield from recurse(wiki_tree or {}, [])


def _page_path_to_str(segments: List[str]) -> str:
    return "/".join(segments)


def _get_project_wiki_dir(project_id: UUID) -> Path:
    """
    Returns the base folder where the project's wiki pages are persisted.
    """
    return PROJECT_WIKI_BASE_PATH / str(project_id)


def _make_react_agent(project_id: UUID) -> dspy.ReAct:
    """
    Creates a DSPy ReAct agent with callable tools for generating
    production-grade technical wiki pages, with file-type-specific parsing
    for COBOL, JCL, and other source files.
    """
    instructions = """
You are a Technical Wiki Page Generator with expertise in documenting
mainframe-related source files, including COBOL programs and JCL scripts.

Goal:
Produce a single, polished Markdown page for the given page_title and page_path.
Do not include reasoning or tool-call transcripts; only output the final Markdown content.

Step 1: Detect File Type
- If the file contains COBOL keywords (IDENTIFICATION DIVISION, PROCEDURE DIVISION, FD, WORKING-STORAGE, PERFORM, MOVE, etc.), treat it as a COBOL program.
- If the file contains JCL syntax (//JOB, //STEP, //DD, EXEC PGM=, PROC, PEND), treat it as a JCL script.
- Otherwise, treat it as a generic source/configuration file.

Step 2: Apply the Correct Documentation Template

=== COBOL Template ===
- Document all COBOL divisions and sections:
  1. IDENTIFICATION DIVISION
  2. ENVIRONMENT DIVISION
     - CONFIGURATION SECTION
     - INPUT-OUTPUT SECTION (FILE-CONTROL)
  3. DATA DIVISION
     - FILE SECTION (FD entries, record layouts, copybooks)
     - WORKING-STORAGE SECTION
     - LINKAGE SECTION
  4. PROCEDURE DIVISION
     - Paragraphs, sections, PERFORM flow, file I/O, subprogram calls
- Include file definitions, data flow, copybook usage, and external dependencies.
- If a section is missing, explicitly state "Not present in source".

=== JCL Template ===
- Job Overview:
  - Job name, purpose, scheduling info (if available)
- Steps:
  - EXEC statements (programs/procs executed)
  - DD statements (datasets, disposition, space allocation)
  - PROC usage and overrides
- Dataset Flow:
  - Input datasets, output datasets, temporary datasets
- External Dependencies:
  - Called programs, utilities (SORT, IDCAMS, IEBGENER, etc.)
- Execution Flow Diagram (Mermaid graph TD or LR)
- Error Handling:
  - COND codes, IF/THEN/ELSE blocks

=== Generic Template ===
- Title and Summary
- Purpose and Scope
- Inputs and Outputs
- Key Components
- Execution Flow
- Configuration and Parameters
- Error Handling
- Dependencies and Relationships
- How to Run / Examples

Step 3: Research
- Use wiki_context and rag_search to extract facts.
- Call rag_search at least once with varied queries (include file-type-specific keywords).
- Only include facts supported by sources; if missing, state "Not available".

Step 4: Diagrams (Mermaid)
- For COBOL: file I/O flow, data structure diagrams.
- For JCL: job step flow, dataset lineage.
- For generic: relevant architecture or flow diagrams.
- Pass only the diagram body to the mermaid tool (no code fences).

Output Contract:
- Return only the final Markdown page as the content.
"""

    signature = dspy.Signature(
        "page_title: str, page_path: str, wiki_context: str -> content: str",
        instructions,
    )

    rag_search = _make_rag_search_fn(project_id)
    mermaid = _make_mermaid_fn()

    return dspy.ReAct(
        signature,
        tools=[rag_search, mermaid],
        max_iters=16,
    )


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
    wiki_context = _format_wiki_tree(wiki_tree)
    analysis_raw = _load_full_analysis_json(project_id)
    if analysis_raw:
        wiki_context = f"{wiki_context}\n\n=== analysis.json ===\n{analysis_raw}"
    agent = react_agent or _make_react_agent(project_id)

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
    for segs in _iter_leaf_paths(wiki_tree):
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
        for segs in _iter_paths(wiki_tree):
            if segs:
                titles.append(segs[-1])

    logger.info(f"[WikiGen] Starting wiki generation (log-only) for project {project_id}. {len(titles)} titles detected. leaves_only={leaves_only}")

    results: Dict[str, str] = {}
    react_agent = _make_react_agent(project_id)
    wiki_context = _format_wiki_tree(wiki_tree)
    analysis_raw = _load_full_analysis_json(project_id)
    if analysis_raw:
        wiki_context = f"{wiki_context}\n\n=== analysis.json ===\n{analysis_raw}"

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
    paths: List[List[str]] = list(_iter_leaf_paths(wiki_tree))
    base_dir = _get_project_wiki_dir(project_id)
    base_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[WikiGen] Starting full wiki generation for project {project_id}. {len(paths)} leaf pages detected.")
    logger.info(f"[WikiGen] Output directory: {base_dir.resolve()}")

    react_agent = _make_react_agent(project_id)
    wiki_context = _format_wiki_tree(wiki_tree)
    analysis_raw = _load_full_analysis_json(project_id)
    if analysis_raw:
        wiki_context = f"{wiki_context}\n\n=== analysis.json ===\n{analysis_raw}"

    manifest_pages: List[Dict[str, Any]] = []

    for idx, segs in enumerate(paths, start=1):
        page_title = segs[-1] if segs else "untitled"
        page_path_str = _page_path_to_str(segs)

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