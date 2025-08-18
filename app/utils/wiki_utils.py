from pathlib import Path
from typing import Any, Dict, Iterable, List
from uuid import UUID

# Common base paths
PROJECT_WIKI_BASE_PATH = Path("project_wiki")
ANALYSIS_BASE_PATH = Path("project_analysis")


def get_project_analysis_dir(project_id: UUID) -> Path:
    """
    Returns the path to the project's analysis directory.
    e.g., project_analysis/{project_id}/analysis
    """
    return ANALYSIS_BASE_PATH / str(project_id) / "analysis"


def get_project_wiki_dir(project_id: UUID) -> Path:
    """
    Returns the base folder where the project's wiki pages are persisted.
    e.g., project_wiki/{project_id}
    """
    return PROJECT_WIKI_BASE_PATH / str(project_id)


def load_full_analysis_json(project_id: UUID) -> str:
    """
    Loads the raw analysis.json content as a string; returns "" if missing.
    """
    p = get_project_analysis_dir(project_id) / "analysis.json"
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, Exception):
        return ""


def format_wiki_tree(wiki_tree: Dict[str, Any]) -> str:
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


def iter_paths(wiki_tree: Dict[str, Any]) -> Iterable[List[str]]:
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


def iter_leaf_paths(wiki_tree: Dict[str, Any]) -> Iterable[List[str]]:
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
                else:  # empty dict -> leaf
                    yield current
            else:
                yield current

    yield from recurse(wiki_tree or {}, [])


def page_path_to_str(segments: List[str]) -> str:
    """Joins path segments into a single string path."""
    return "/".join(segments)


def slugify(text: str) -> str:
    """
    Normalizes a string into a URL-friendly slug.
    """
    s = "".join(ch.lower() if ch.isalnum() else "-" for ch in (text or ""))
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-") or "untitled"


def insert_path(tree: Dict[str, Any], path: str) -> None:
    """
    Insert a nested path like 'a/b/c' into a dict as nested keys.
    """
    parts = [p for p in (path or "").split("/") if p]
    cur = tree
    for p in parts:
        cur = cur.setdefault(p, {})