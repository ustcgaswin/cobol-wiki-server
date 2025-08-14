from pathlib import Path
from uuid import UUID
from typing import Dict, Any, List, Iterable
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy

from app.services.rag_service import search_project, get_project_searcher
from app.utils.logger import logger

# Root for persisted wiki output relative to the current workspace
PROJECT_WIKI_BASE_PATH = Path("project_wiki")
ANALYSIS_BASE_PATH = Path("project_analysis")

# Concurrency for wiki generation
WIKI_MAX_WORKERS = int(os.environ.get("WIKI_MAX_WORKERS", "1"))


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


def _make_rag_search_fn(project_id: UUID, searcher=None):
    """
    Returns a callable tool that searches the project's RAG index and formats results.
    Each result includes a machine-parseable citation tag:
      <cite file="rel/path.ext" lines="start-end" score="0.000" />
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
            if searcher is not None:
                results = searcher.search(q, tk)
            else:
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
                file_ = r.get("file", "")
                ls = r.get("line_start", "")
                le = r.get("line_end", "")
                score = r.get("score", 0.0)
                cite = f'<cite file="{file_}" lines="{ls}-{le}" score="{score:.3f}" />'
                lines.append(f"[{i+1}] {title}\n{cite}\n{content}")
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


def _make_react_agent(project_id: UUID, searcher=None) -> dspy.ReAct:
    """
    Creates a DSPy ReAct agent with callable tools for generating
    production-grade technical wiki pages, with file-type-specific parsing
    for COBOL, JCL, and other source files.
    """
    
    instructions = """
You are a Technical Wiki Page Generator with deep expertise in documenting
mainframe-related source files (COBOL, Copybooks, JCL, REXX, plus generic
code or configuration artifacts).

Goal
Produce one clear, well-structured Markdown page for the given page_title and
page_path.  The page is the ONLY thing that must be returned; do not emit
chain-of-thought, tool calls, JSON, or any extra text.

Key Restrictions
- You never see the file directly; gather facts exclusively through the
  rag_search tool and the supplied wiki_context.
- Absolutely no hallucination: every substantive statement MUST be supported
  by at least one <cite …/> tag returned from rag_search.
- Do not sprinkle raw <cite …/> tags throughout the prose.  They may appear
  only in the section-ending “Sources:” line as defined below.

Citations Policy (MUST FOLLOW)
1. No inline citations inside prose or bullet points.  
2. After finishing each H2/H3 section, add a single line that begins
   exactly with `Sources:` followed by one or more cite tags, comma-separated.
   Example:
   Sources: <cite>src/PGM001.cbl:15-89</cite>, <cite>includes/FILEAUTO.cpy:3-47</cite>
3. Every cite tag must correspond to a *real* tag (`<cite path:start-end/>`)
   returned by your prior rag_search calls.
   - Bare filenames, folder names, or guessed line numbers are INVALID.
4. Within a section, deduplicate citations and list them in first-appearance
   order.
5. If you truly have no sourced facts for a section, omit the Sources line
   entirely.

Mandatory Section — “Changelog / Revision History”
- Always search for maintenance headers or comment blocks that mention
  “CHANGE LOG”, “HISTORY”, “REVISION”, “Version”, “Modified by”, dates,
  or ticket numbers.
- If found, create an H2 section named `Changelog / Revision History`
  summarizing the entries in reverse-chronological order (newest first).
  Each bullet should include date, author/ID if present, and a concise
  description.
- If nothing is located, omit this section.

Workflow
Step 1 – Retrieve Source Material
- First rag_search query: `"Full source code for {page_path}"`.
- If that returns ≥500 lines, switch to targeted queries (e.g.,
  `"IDENTIFICATION DIVISION in {page_path}"`) to keep token usage sane.
- Perform at least four distinct, well-targeted rag_search calls until you can:
  a. Determine file type confidently and  
  b. Populate every planned section with fact-backed details.

Step 2 – Detect File Type
- COBOL Program : DIVISION headers plus PROCEDURE DIVISION present
- Copybook      : COBOL level numbers/PIC clauses but NO PROCEDURE DIVISION
- JCL           : //JOB, //STEP, EXEC, DD cards, PROC/PEND
- REXX          : /* REXX */, SAY, PARSE, DO/END, etc.
- Otherwise     : treat as Generic

Step 3 – Decide Page Outline
- Start with `# {page_title}`
- One-paragraph summary (what it is, why it matters).
- Then choose sections according to file type (templates below).  
  Only include sections that add meaningful value; merge or drop irrelevant
  ones.
- Include “Changelog / Revision History” when data is available.
- Add Mermaid diagrams where they help.

Step 4 – Populate Sections
- Extract precise facts with rag_search; paraphrase without embellishment.
- Where the source lacks information, state “Not documented in source”.
- After completing a section, append its Sources line per the policy.

Step 5 – Validate Before Returning
- Every Sources line follows the exact pattern:
  Sources: <cite>path:start-end</cite>, <cite>other/file:10-34</cite>
- No dangling or fake citations.  No raw <cite …/> inside prose.
- No lowercase “sources:”.  No extra commentary outside the Markdown page.

Template Hints
=== COBOL Programs ===
H2 Overview  
H2 Environment Division  
H2 Data Division  
H2 Procedure Division  
H2 External Dependencies  
H2 Changelog / Revision History  
H2 Mermaid Diagrams (optional)

=== Copybooks ===
Overview  
Data Structures  
Field Descriptions  
Dependencies  
Changelog / Revision History  

=== JCL ===
Job Overview  
Steps & Programs  
DD Statements & Dataset Flow  
PROC Overrides  
Error Handling / COND Logic  
Changelog / Revision History  
Mermaid Diagram – Job Flow

=== REXX / Generic ===
Overview  
Inputs & Parameters  
Control Flow / Key Routines  
External Calls & Dependencies  
Error Handling  
Changelog / Revision History  

Mermaid Guidance
- Diagrams may be as large as necessary; ensure readability.
  If a flow is huge, consider splitting into multiple diagrams rather than
  cramming everything into one.
- Embed in Markdown like:
  ```mermaid
  graph TD;
  ...
  ```
- Use clear labels and directional arrows.

Remember
Our users demand reliability and traceability.  When in doubt, fetch more
evidence with rag_search; never invent details.

"""

    signature = dspy.Signature(
        "page_title: str, page_path: str, wiki_context: str -> content: str",
        instructions,
    )

    rag_search = _make_rag_search_fn(project_id, searcher=searcher)
    

    return dspy.ReAct(
        signature,
        tools=[rag_search],
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

    if react_agent is None:
        # Use cached RAG searcher for faster lookups
        try:
            searcher = get_project_searcher(project_id)
        except Exception:
            searcher = None
        agent = _make_react_agent(project_id, searcher=searcher)
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
    # Shared cached RAG searcher
    try:
        searcher = get_project_searcher(project_id)
    except Exception:
        searcher = None

    wiki_context = _format_wiki_tree(wiki_tree)
    analysis_raw = _load_full_analysis_json(project_id)
    if analysis_raw:
        wiki_context = f"{wiki_context}\n\n=== analysis.json ===\n{analysis_raw}"

    if WIKI_MAX_WORKERS <= 1:
        react_agent = _make_react_agent(project_id, searcher=searcher)
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
            agent = _make_react_agent(project_id, searcher=searcher)
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
    paths: List[List[str]] = list(_iter_leaf_paths(wiki_tree))
    base_dir = _get_project_wiki_dir(project_id)
    base_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[WikiGen] Starting full wiki generation for project {project_id}. {len(paths)} leaf pages detected.")
    logger.info(f"[WikiGen] Output directory: {base_dir.resolve()}")

    # Shared cached RAG searcher
    try:
        searcher = get_project_searcher(project_id)
    except Exception:
        searcher = None

    wiki_context = _format_wiki_tree(wiki_tree)
    analysis_raw = _load_full_analysis_json(project_id)
    if analysis_raw:
        wiki_context = f"{wiki_context}\n\n=== analysis.json ===\n{analysis_raw}"

    manifest_pages: List[Dict[str, Any]] = []

    if WIKI_MAX_WORKERS <= 1:
        react_agent = _make_react_agent(project_id, searcher=searcher)
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
    else:
        def worker(segs: List[str]) -> Dict[str, Any]:
            page_title = segs[-1] if segs else "untitled"
            page_path_str = _page_path_to_str(segs)
            agent = _make_react_agent(project_id, searcher=searcher)

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
                page_path_str = _page_path_to_str(segs)
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