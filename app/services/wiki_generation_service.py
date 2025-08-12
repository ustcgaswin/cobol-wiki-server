from pathlib import Path
from uuid import UUID
from typing import Dict, Any, List, Iterable
import dspy

from app.services.rag_service import search_project
from app.utils.logger import logger

# Root for persisted wiki output relative to the current workspace
PROJECT_WIKI_BASE_PATH = Path("project_wiki")


def _make_rag_search_fn(project_id: UUID):
    """
    Returns a callable tool that searches the project's RAG index and formats results.
    """
    def rag_search(query: str, top_k: int = 20) -> str:
        try:
            results = search_project(project_id, query, top_k)
        except FileNotFoundError:
            return "No RAG index found."
        except Exception as e:
            return f"RAG search error: {e}"

        if not results:
            return "No results found."
        return "\n\n".join(
            f"[{i+1}] {r.get('title', '')}\n{r.get('content', '')}"
            for i, r in enumerate(results)
        )
    return rag_search


def _make_mermaid_fn():
    """
    Returns a callable tool that wraps a Mermaid diagram body in fences.
    """
    def mermaid(description: str) -> str:
        desc = (description or "").strip()
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
    Creates a DSPy ReAct agent with callable tools.
    """
    instructions = """
You are a **Technical Wiki Page Generator**.

Your task: Create a **clear, accurate, and thorough Markdown page** for the given
`page_title` and `page_path`.

---

## **Core Requirements**
1. **Markdown Output**
   - Always use Markdown with meaningful, hierarchical section headings.
   - Follow the provided **Page Template** (see below) but adapt if needed.

2. **Information Gathering**
   - Before writing, call the `rag_search` tool **at least twice** with queries derived from:
     • `page_title`
     • `page_path`
     • likely synonyms, program names, dataset names, or job steps.
   - Use the most relevant and factual details from `rag_search` results.
   - Number the results as `[RAG 1]`, `[RAG 2]`, etc., in the order retrieved.

3. **References**
   - Include a **References** section at the end.
   - Cite sources inline using `[RAG X]` where X matches the numbered search result.

4. **Diagrams**
   - If there is an execution flow, component interaction, or data lineage:
     • Use the `mermaid` tool to create at least one diagram.
     • Choose the most appropriate diagram type (flowchart, sequence, etc.).

5. **Wiki Context**
   - Use the provided `wiki_context` (Wiki Tree Structure) to:
     • Cross-link related pages using **relative paths**.
     • Maintain consistency in terminology and structure.

6. **Factual Accuracy**
   - Prefer details from `rag_search` over assumptions.
   - Avoid speculation; if data is missing, state it clearly.

---

## **Page Template**
(Adapt as needed based on the topic)

1. **Title and Summary**
   - Page title as H1
   - 2–3 sentence summary of the topic

2. **Purpose and Scope**
   - Why this exists and what it covers

3. **Inputs and Outputs**
   - Files, datasets, DB tables, messages, or APIs

4. **Key Components**
   - Programs, jobs, modules, copybooks with brief descriptions

5. **Execution Flow**
   - Step-by-step process
   - Include a **Mermaid diagram** if applicable

6. **Configuration and Parameters**
   - JCL parameters, environment variables, job cards, scheduler settings

7. **Error Handling and Edge Cases**
   - Known failure modes and recovery steps

8. **Dependencies and Relationships**
   - Cross-link related wiki pages using relative paths

9. **How to Run / Examples**
   - Commands, JCL snippets, sample invocations

10. **References**
    - List `[RAG X]` citations with their corresponding snippet summaries

---
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