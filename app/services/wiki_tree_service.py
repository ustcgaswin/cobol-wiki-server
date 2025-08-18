import json
from pathlib import Path
from typing import Any, Dict
from uuid import UUID

from app.utils.logger import logger
from app.utils.wiki_utils import (
    get_project_analysis_dir,
    insert_path,
    slugify,
)


def get_wiki_structure_file_path(project_id: UUID) -> Path:
    """Path for persisted wiki tree."""
    return get_project_analysis_dir(project_id) / "wiki_structure.json"


# ---------------- Wiki structure helpers (deterministic, no LLM) ----------------


def build_pages_tree(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a nested pages tree only. No nav, no page payload.
    Leaves are empty dicts. Supports nested pages.
    """
    tree: Dict[str, Any] = {}

    # Overview
    insert_path(tree, "overview")

    # JCL
    for j in analysis.get("jcl", []) or []:
        file_label = j.get("file") or "unknown.jcl"
        file_slug = slugify(file_label)
        # Jobs
        for job in j.get("jobs") or []:
            job_name = job.get("name") or "UNNAMED_JOB"
            job_slug = slugify(job_name)
            insert_path(tree, f"jcl/{file_slug}/{job_slug}")
        # Relationships page per file
        insert_path(tree, f"jcl/{file_slug}/relationships")

    # COBOL
    for c in analysis.get("cobol", []) or []:
        file_label = c.get("file") or "unknown.cbl"
        insert_path(tree, f"cobol/{slugify(file_label)}")

    # Copybooks
    for cb in analysis.get("copybooks", []) or []:
        file_label = cb.get("file") or "unknown.cpy"
        insert_path(tree, f"copybooks/{slugify(file_label)}")

    # REXX
    for rx in analysis.get("rexx", []) or []:
        file_label = rx.get("file") or "unknown.rex"
        file_slug = slugify(file_label)
        insert_path(tree, f"rexx/{file_slug}")
        subs = (
            ((rx.get("data") or {}).get("control_flow") or {}).get("subroutines") or []
        )
        for s in subs:
            sub_slug = slugify(s.get("name") or "subroutine")
            insert_path(tree, f"rexx/{file_slug}/sub/{sub_slug}")

    return tree


# # Back-compat: keep old name but return the minimal tree
# def build_wiki_structure_from_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Return only the nested pages tree (no nav, no page payload).
#     """
#     return build_pages_tree(analysis)


def get_wiki_structure_for_project(project_id: UUID) -> Dict[str, Any]:
    """
    Return the nested pages tree for a project.
    Prefer reading a persisted wiki_structure.json; otherwise build from analysis.json.
    Always log the tree.
    """
    tree_file = get_wiki_structure_file_path(project_id)
    if tree_file.exists():
        with open(tree_file, "r", encoding="utf-8") as tf:
            tree = json.load(tf)
        logger.info(
            f"Wiki structure (tree) from disk for project {project_id}:\n{json.dumps(tree, indent=2)}"
        )
        return tree

    analysis_file = get_project_analysis_dir(project_id) / "analysis.json"
    if not analysis_file.exists():
        raise FileNotFoundError(f"analysis.json not found at {analysis_file}")
    with open(analysis_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    tree = build_pages_tree(payload)
    logger.info(
        f"Wiki structure (tree) for project {project_id}:\n{json.dumps(tree, indent=2)}"
    )

    # Persist for future reads
    try:
        tree_file.parent.mkdir(parents=True, exist_ok=True)
        with open(tree_file, "w", encoding="utf-8") as wf:
            json.dump(tree, wf, indent=2)
        logger.info(f"Wrote wiki structure to {tree_file}")
    except Exception as e:
        logger.error(
            f"Failed to persist wiki structure for project {project_id}: {e}",
            exc_info=True,
        )

    return tree