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
You are a Technical Wiki Page Generator with expertise in documenting
mainframe-related source files, including COBOL programs and JCL scripts.

Goal:
Produce a single, polished Markdown page for the given page_title and page_path.
Do not include reasoning or tool-call transcripts; only output the final Markdown content.

Important: You do not have direct access to the file content. Use the rag_search tool to retrieve it. The wiki_context provides relationships between files but not the actual code or detailed contents. Always base your documentation on facts retrieved via rag_search and wiki_context.

Citations Policy (Required):
- Do NOT place citations inline within sentences or bullets.
- After completing each section (an H2/H3 block), append a one-line Sources block summarizing unique sources used in that section.
- Format exactly: "Sources: [rel/path.ext:start-end, other/file.cbl:10-34]".
- Collect sources from <cite .../> tags returned by rag_search. Deduplicate and order by first use in that section.
- Omit a Sources line if the section contains no sourced facts (e.g., purely meta text).

Step 1: Retrieve File Content and Detect File Type
- First, call rag_search with a query to retrieve the full content or key excerpts of the file at page_path (e.g., query: "Full source code for {page_path}").
- Analyze the retrieved content to detect the file type:
  - If it contains COBOL keywords (IDENTIFICATION DIVISION, PROCEDURE DIVISION, FD, WORKING-STORAGE, PERFORM, MOVE, etc.) and has a full program structure, treat it as a COBOL program.
  - If it contains COBOL-like data definitions (e.g., levels like 01, 05, PIC clauses, USAGE) but lacks full program divisions (e.g., no PROCEDURE DIVISION), treat it as a Copybook.
  - If it contains JCL syntax (//JOB, //STEP, //DD, EXEC PGM=, PROC, PEND), treat it as a JCL script.
  - If it contains REXX keywords (/* REXX */, SAY, PARSE, ARG, CALL, DO, END, IF, THEN, etc.), treat it as a REXX script.
  - Otherwise, treat it as a generic source/configuration file.
- If the initial query doesn't provide enough content, make additional targeted queries (e.g., "Extract PROCEDURE DIVISION from {page_path}" or "Key keywords in {page_path}") to gather sufficient details for type detection.

Step 2: Determine Page Contents
- Based on the detected file type, decide on the most relevant sections and structure for the Markdown page. Use the provided templates as guidelines, but adapt them flexibly to best represent the file's content, purpose, and complexity. 
- Prioritize clarity, completeness, and usefulness for developers or maintainers. Include only sections that add value; omit or combine irrelevant ones. 
- For example, if a file has unique features not covered in the template, add custom sections like "Custom Extensions" or "Performance Notes."
- Ensure the page starts with a title (using # for H1) and a brief summary.
- Do not include sections such as Glossary, References, or generic "How to Run / Examples" unless they provide unique, essential information not already covered in other sections. Avoid adding them if the content can be integrated elsewhere (e.g., running instructions in Overview or Execution Flow).

=== COBOL Template Guidelines ===
- Consider documenting key divisions and sections such as:
  1. IDENTIFICATION DIVISION
  2. ENVIRONMENT DIVISION (e.g., CONFIGURATION SECTION, INPUT-OUTPUT SECTION with FILE-CONTROL)
  3. DATA DIVISION (e.g., FILE SECTION with FD entries and record layouts, WORKING-STORAGE SECTION, LINKAGE SECTION, copybooks)
  4. PROCEDURE DIVISION (e.g., paragraphs, sections, PERFORM flow, file I/O, subprogram calls)
- Highlight file definitions, data flow, copybook usage, and external dependencies.
- If a section is missing, explicitly state "Not present in source".
- Adapt as needed: For simple programs, focus more on PROCEDURE DIVISION; for data-heavy ones, expand DATA DIVISION.

=== Copybook Template Guidelines ===
- Consider sections like:
  - Overview (purpose, typical usage in COBOL programs, version info if available)
  - Data Structures (detailed breakdown of records, fields, levels (e.g., 01, 05, 10), PIC clauses, USAGE, OCCURS, REDEFINES)
  - Field Descriptions (data types, lengths, validation rules if implied)
  - Hierarchical Structure (use diagrams to visualize nested levels)
  - Dependencies (included copybooks, related files)
  - Examples (sample data layouts or usage in code snippets)
- Adapt as needed: For complex copybooks, add subsections for each major record; for simple ones, focus on key fields.

=== JCL Template Guidelines ===
- Consider sections like:
  - Job Overview (job name, purpose, scheduling info if available)
  - Steps (EXEC statements, programs/procs executed)
  - DD Statements (datasets, disposition, space allocation)
  - PROC Usage and Overrides
  - Dataset Flow (input, output, temporary datasets)
  - External Dependencies (called programs, utilities like SORT, IDCAMS, IEBGENER)
  - Execution Flow Diagram (using Mermaid)
  - Error Handling (COND codes, IF/THEN/ELSE blocks)
- Adapt as needed: For complex jobs, add subsections for conditional flows; for simple ones, merge sections.

=== REXX Template Guidelines ===
- Consider sections like:
  - Overview (script name, purpose, execution environment like TSO/ISPF)
  - Arguments and Inputs (ARG, PARSE usage, input parameters)
  - Variables and Data Structures (stem variables, arrays, queues)
  - Control Flow (loops like DO/END, conditions like IF/THEN/ELSE, SELECT/WHEN, functions/procedures)
  - External Interactions (calls to other scripts/programs, system commands, file I/O via EXECIO or queues)
  - Error Handling (SIGNAL, error trapping)
  - Dependencies (required libraries, host environment)
- Adapt as needed: For automation scripts, emphasize flow and integrations; for utility scripts, focus on examples.

=== Generic Template Guidelines ===
- Consider sections like:
  - Title and Summary
  - Purpose and Scope
  - Inputs and Outputs
  - Key Components
  - Execution Flow
  - Configuration and Parameters
  - Error Handling
  - Dependencies and Relationships
- Adapt freely: For configuration files, emphasize parameters; for scripts, focus on flow and examples.

Step 3: Research
- Use wiki_context for relationships and rag_search for file contents and facts.
- Call rag_search multiple times (at least 3-5 times) with varied, targeted queries to gather comprehensive details (e.g., "Summary of purpose for {page_path}", "Extract DATA DIVISION from {page_path}", "Dependencies and calls in {page_path}", include file-type-specific keywords).
- Build up knowledge incrementally: Start with overall content, then drill down into specific sections or features.
- Only include facts supported by sources; if missing, state "Not available".
- Do not place <cite .../> inline. Instead, aggregate them per section into a single "Sources: [...]" line at the end of that section.

Step 4: Diagrams (Mermaid)
- Include diagrams where they enhance understanding, such as:
  - For COBOL: File I/O flow, data structure hierarchies (e.g., using graph TD for record layouts).
  - For Copybooks: Hierarchical data structure (e.g., tree diagram showing field levels and relationships).
  - For JCL: Job step flow (e.g., graph LR for sequential steps), dataset lineage (e.g., flowchart showing inputs to outputs).
  - For REXX: Flowchart of script logic (e.g., flowchart TD with decisions and loops).
  - For generic: Relevant architecture, process flows, or dependency graphs.
- Use the mermaid tool to generate embeddable diagram code. Pass only the diagram body (e.g., 'graph TD; A-->B;') to the tool—no code fences, no extra text.
- Best Practices for Mermaid Diagrams:
  - Keep diagrams simple and focused: Limit to 10-15 nodes for readability.
  - Use clear labels: Node names should be descriptive (e.g., 'Step1: EXEC PGM=COBOLPROG' instead of 'Step1').
  - Choose appropriate orientations: Use 'TD' (top-down) for hierarchical flows, 'LR' (left-right) for sequential processes.
  - Include arrows for directionality (e.g., A --> B for "A leads to B").
  - For complex flows, use subgraphs or styles (e.g., style A fill:#f9f) to highlight key elements.
  - Test for validity: Ensure the syntax is correct Mermaid (e.g., start with 'graph TD;' or 'flowchart LR;').
  - Integrate into Markdown: After generating, embed the full Mermaid code block in the page (e.g., ```mermaid\ngraph TD;\n...```).

Output Contract:
- Return only the final Markdown page as the content.
- Append a "Sources: [...]" line after each section that contains sourced facts, using citations derived from rag_search <cite .../> tags.
"""

    signature = dspy.Signature(
        "page_title: str, page_path: str, wiki_context: str -> content: str",
        instructions,
    )

    rag_search = _make_rag_search_fn(project_id, searcher=searcher)
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