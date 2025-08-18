from uuid import UUID
from typing import List, Any
import subprocess
import os
import dspy
import pandas as pd
import re
import pathlib
import json

from app.services.rag_service import search_project
from app.services.project_service import get_project_by_id
from app.services.db_population_service import get_db_engine



ANALYSIS_BASE_PATH = pathlib.Path("project_analysis")
def _analysis_file_path(project_id: UUID) -> pathlib.Path:
    return ANALYSIS_BASE_PATH / str(project_id) / "analysis" / "analysis.json"


def _get_project_path(project_id: UUID) -> str:
    """
    Resolve absolute filesystem path for a project.

    Current storage layout: project_storage/<project_id>/
    Previously relied on Project.path (now absent).
    """
    project = get_project_by_id(project_id)
    if not project:
        raise FileNotFoundError(f"Project with ID {project_id} not found.")

    # If a path attribute (or similar) was reintroduced, honor it first.
    for attr in ("path", "root_path", "project_path", "directory"):
        val = getattr(project, attr, None)
        if isinstance(val, str) and val.strip():
            return os.path.abspath(val)

    # Fallback: build from PROJECT_STORAGE_ROOT env var (default: project_storage)
    storage_root = os.getenv("PROJECT_STORAGE_ROOT", "project_storage")
    candidate = os.path.abspath(os.path.join(storage_root, str(project_id)))

    if not os.path.exists(candidate):
        raise FileNotFoundError(
            f"Derived project path '{candidate}' does not exist. "
            f"Set PROJECT_STORAGE_ROOT or add a path attribute to Project."
        )
    return candidate


def make_rag_search_tool(project_id: UUID, searcher: Any = None):
    """
    Returns a callable tool that searches the project's RAG index and formats results.
    Each result includes a machine-parseable citation tag:
      <cite file="rel/path.ext" lines="start-end" score="0.000" />
    """
    def rag_search(query: str = "", top_k: int = 20, **kwargs) -> str:
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



def make_git_log_tool(project_id: UUID):
    """
    Returns a callable tool that gets the git commit history for a specific file.
    Supports omission of the file extension by inferring a unique match.
    """
    def _safe_norm(rel: str) -> str:
        rel = os.path.normpath(rel).replace("\\", "/").lstrip("/")
        if rel.startswith(".."):
            raise ValueError("Illegal relative path traversal.")
        return rel

    def _find_stem_matches(project_path: str, norm_path: str) -> list[str]:
        """
        Find files whose basename (without extension) matches the stem provided.
        If a directory was specified, only search that directory.
        Otherwise, search the entire project tree (bounded).
        """
        base_dir, leaf = os.path.split(norm_path)
        if "." in leaf:
            return []  # Already has an extension, no inference needed.
        stem = leaf.lower()

        matches: list[str] = []
        max_walk = 10000  # safety cap on files visited

        def add_match(full_path: str):
            rel = os.path.relpath(full_path, project_path).replace("\\", "/")
            matches.append(rel)

        if base_dir:
            candidate_dir = os.path.join(project_path, base_dir)
            if not os.path.isdir(candidate_dir):
                return []
            try:
                for entry in os.listdir(candidate_dir):
                    full_entry = os.path.join(candidate_dir, entry)
                    if os.path.isfile(full_entry):
                        name, ext = os.path.splitext(entry)
                        if name.lower() == stem and ext:
                            add_match(full_entry)
            except Exception:
                return []
        else:
            visited = 0
            for root, dirs, files in os.walk(project_path):
                for f in files:
                    visited += 1
                    if visited > max_walk:
                        return []  # Abort on large repos to avoid cost
                    name, ext = os.path.splitext(f)
                    if name.lower() == stem and ext:
                        add_match(os.path.join(root, f))
        return matches

    def git_log(file_path: str = "", **kwargs) -> str:
        raw_path = file_path or kwargs.get("file_path") or kwargs.get("path") or ""
        raw_path = str(raw_path).strip()
        if not raw_path:
            return "No file_path provided."

        try:
            project_path = _get_project_path(project_id)
            try:
                norm_path = _safe_norm(raw_path)
            except ValueError as ve:
                return f"Error: {ve}"

            target_abs = os.path.join(project_path, norm_path)

            if not os.path.exists(target_abs):
                # Try extension inference only if no dot in final segment
                if "." not in os.path.basename(norm_path):
                    inferred = _find_stem_matches(project_path, norm_path)
                    if len(inferred) == 1:
                        norm_path = inferred[0]
                        target_abs = os.path.join(project_path, norm_path)
                    elif len(inferred) > 1:
                        return ("Error: Ambiguous file stem without extension. "
                                "Matches: " + ", ".join(inferred))
                    else:
                        return f"Error: File not found at {raw_path} (no matching stem with any extension)."
                else:
                    return f"Error: File not found at {raw_path}"

            # Run git log using the repository-relative normalized path
            cmd = ["git", "log", "--pretty=format:%h - %an, %ar : %s", "--", norm_path]
            result = subprocess.run(
                cmd,
                cwd=project_path,
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout.strip() or "No git history found for this file."
            if norm_path != raw_path:
                output = f"(auto-inferred path: {norm_path})\n{output}"
            return output

        except FileNotFoundError:
            return f"Project with ID {project_id} not found."
        except subprocess.CalledProcessError as e:
            return f"Git log error: {e.stderr}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"
    return git_log




def make_project_tree_tool(project_id: UUID):
    """
    Returns a callable tool that shows the project's file and directory structure.
    """
    def project_tree(**kwargs) -> str:
        try:
            root_path = pathlib.Path(_get_project_path(project_id))
        except FileNotFoundError:
            return f"Project with ID {project_id} not found."
        except Exception as e:
            return f"Error resolving project path: {e}"

        fmt = (kwargs.get("format") or kwargs.get("mode") or "tree").lower()
        max_depth = int(kwargs.get("max_depth", 6))
        max_entries = int(kwargs.get("max_entries", 500))
        include_hidden = bool(kwargs.get("include_hidden", False))
        show_counts = bool(kwargs.get("show_counts", True))
        sort_mode = kwargs.get("sort", "name")

        counts = {"dirs": 0, "files": 0}
        emitted = 0
        truncated = False

        def is_hidden(p: pathlib.Path) -> bool:
            return p.name.startswith(".")

        def children(dir_path: pathlib.Path):
            try:
                entries = list(dir_path.iterdir())
            except Exception:
                return []
            if not include_hidden:
                entries = [e for e in entries if not is_hidden(e)]
            if sort_mode == "type":
                entries.sort(key=lambda p: (p.is_file(), p.name.lower()))
            else:
                entries.sort(key=lambda p: p.name.lower())
            return entries

        if fmt == "json":
            def build(node_path: pathlib.Path, depth: int, rel: str):
                nonlocal emitted, truncated
                if depth > max_depth or truncated:
                    return None
                obj = {
                    "name": node_path.name if rel != "." else node_path.name,
                    "path": rel,
                    "type": "directory" if node_path.is_dir() else "file",
                }
                if node_path.is_file():
                    try:
                        stat = node_path.stat()
                        obj["size"] = stat.st_size
                        obj["modified_ts"] = int(stat.st_mtime)
                    except Exception:
                        pass
                    obj["ext"] = node_path.suffix.lower()
                    counts["files"] += 1
                    emitted += 1
                else:
                    counts["dirs"] += 1 if rel != "." else 0
                    emitted += 1
                    if emitted >= max_entries:
                        truncated = True
                        obj["children"] = []
                        return obj
                    ch = []
                    for c in children(node_path):
                        if emitted >= max_entries:
                            truncated = True
                            break
                        rel_child = c.name if rel == "." else f"{rel}/{c.name}"
                        built = build(c, depth + 1, rel_child)
                        if built:
                            ch.append(built)
                    obj["children"] = ch
                return obj

            root_obj = build(root_path, 0, ".")
            if root_obj is None:
                root_obj = {
                    "name": root_path.name,
                    "path": ".",
                    "type": "directory",
                    "children": []
                }
            root_obj["summary"] = {
                "dirs": counts["dirs"],
                "files": counts["files"],
                "truncated": truncated,
                "max_depth": max_depth,
                "max_entries": max_entries,
            }
            try:
                return json.dumps(root_obj, indent=2)
            except Exception as e:
                return f"Error serializing JSON: {e}"

        lines = []
        lines.append(f".  (project root: {root_path.name})")

        def render(dir_path: pathlib.Path, prefix: str, depth: int):
            nonlocal emitted, truncated
            if depth > max_depth or truncated:
                return
            ch = children(dir_path)
            for idx, c in enumerate(ch):
                if emitted >= max_entries:
                    truncated = True
                    return
                last = idx == len(ch) - 1
                connector = "└─" if last else "├─"
                if c.is_dir():
                    counts["dirs"] += 1
                    lines.append(f"{prefix}{connector} {c.name}/")
                    emitted += 1
                    extension_prefix = "   " if last else "│  "
                    render(c, prefix + extension_prefix, depth + 1)
                else:
                    counts["files"] += 1
                    lines.append(f"{prefix}{connector} {c.name}")
                    emitted += 1

        try:
            render(root_path, "", 0)
        except Exception as e:
            return f"Error while building tree: {e}"

        if truncated:
            lines.append(f"... (truncated after {max_entries} entries)")
        if show_counts:
            lines.append(f"\nSummary: {counts['dirs']} dirs, {counts['files']} files"
                         + (", truncated" if truncated else ""))
        return "\n".join(lines)

    return project_tree


class TextToSQL(dspy.Signature):
    """Given a database schema and a question, generate a valid SQLite query."""
    context = dspy.InputField(desc="A description of the database schema (tables and columns).")
    question = dspy.InputField(desc="A natural language question about the data.")
    query = dspy.OutputField(desc="A valid SQLite query that answers the question.")


def make_db_query_tool(project_id: UUID):
    """
    Returns a callable tool that converts a natural language question into a
    SQL query using a hardcoded schema, executes it, and returns the result.
    """
    schema_context = """
You have access to a SQLite database with the following tables and columns.
Use these to answer questions about the project's source code.
When joining tables, you must explicitly reference the foreign key relationships.

Table `sourcefile`: Stores information about each source file.
- id (INTEGER PRIMARY KEY)
- relative_path (TEXT)
- file_name (TEXT)
- file_type (TEXT)
- content (TEXT)
- status (TEXT)

Table `filerelationship`:
- id (INTEGER PRIMARY KEY)
- source_file_id (INTEGER) FK -> sourcefile.id
- target_file_name (TEXT)
- relationship_type (TEXT)
- statement (TEXT)
- line_number (INTEGER)

Table `cobolprogram`:
- id (INTEGER PRIMARY KEY)
- file_id (INTEGER) FK -> sourcefile.id
- program_id_name (TEXT)
- program_type (TEXT)

Table `cobolstatement`:
- id (INTEGER PRIMARY KEY)
- program_id (INTEGER) FK -> cobolprogram.id
- statement_type (TEXT)
- target (TEXT)
- content (TEXT)
- line_number (INTEGER)

Table `jcljob`:
- id (INTEGER PRIMARY KEY)
- file_id (INTEGER) FK -> sourcefile.id
- job_name (TEXT)

Table `jclstep`:
- id (INTEGER PRIMARY KEY)
- job_id (INTEGER) FK -> jcljob.id
- step_name (TEXT)
- exec_type (TEXT)
- exec_target (TEXT)

Table `jclddstatement`:
- id (INTEGER PRIMARY KEY)
- job_id (INTEGER) FK -> jcljob.id
- step_id (INTEGER) FK -> jclstep.id
- dd_name (TEXT)
- dataset_name (TEXT)
- disposition (TEXT)
"""

    def db_query(question: str = "", **kwargs) -> str:
        q = question or kwargs.get("question") or kwargs.get("query") or ""
        q = str(q).strip()
        if not q:
            return "No question provided for the database query tool."

        try:
            text_to_sql_predictor = dspy.Predict(TextToSQL)
            result = text_to_sql_predictor(context=schema_context, question=q)
            sql_query = result.query.replace("```sql", "").replace("```", "").strip()

            engine = get_db_engine(project_id)
            with engine.connect() as connection:
                df = pd.read_sql_query(sql_query, connection)

            if df.empty:
                return "Query executed successfully, but returned no results."
            return df.to_markdown(index=False)

        except FileNotFoundError:
            return f"Database for project with ID {project_id} not found."
        except Exception as e:
            error_sql = "not generated"
            if 'sql_query' in locals():
                error_sql = sql_query
            return f"An error occurred during DB query execution. Query was: `{error_sql}`. Error: {e}"

    return db_query


def make_analysis_graph_tool(project_id: UUID):
    """
    Returns a callable tool that reads the project's analysis.json file
    and generates a Mermaid dependency graph (validated).
    """

    def get_analysis_graph(**kwargs) -> str:
        try:
            analysis_file = _analysis_file_path(project_id)
            if not analysis_file.exists():
                return "Error: analysis.json not found. Please run the project analysis first."

            with open(analysis_file, "r", encoding="utf-8") as f:
                analysis = json.load(f)

            nodes, edges = _extract_graph_data(analysis)
            if not nodes and not edges:
                return "No dependencies found in analysis.json to generate a graph."

            mermaid_text = _to_mermaid_syntax(nodes, edges)

            # Validate Mermaid syntax
            try:
                validator_path = pathlib.Path(__file__).resolve().parents[2] / "mermaid_validator.js"
                if not validator_path.exists():
                    return ("Mermaid graph generated, but validator script not found at "
                            f"{validator_path}.\n\n{mermaid_text}")

                proc = subprocess.run(
                    ["node", str(validator_path)],
                    input=mermaid_text,
                    text=True,
                    capture_output=True,
                )
                if proc.returncode != 0:
                    line_info = proc.stdout.strip()
                    err_msg = proc.stderr.strip()
                    return (f"Mermaid validation failed (exit {proc.returncode}). "
                            f"Line {line_info or '?'}: {err_msg}\n\n--- Invalid Graph ---\n{mermaid_text}")
                return mermaid_text
            except FileNotFoundError:
                return ("Mermaid graph generated, but Node.js not found in PATH for validation.\n\n"
                        f"{mermaid_text}")
            except Exception as e:
                return (f"Mermaid graph generated, but validation errored: {e}\n\n{mermaid_text}")

        except json.JSONDecodeError:
            return "Error: Could not parse analysis.json. The file may be corrupt."
        except Exception as e:
            return f"An unexpected error occurred while generating the analysis graph: {e}"

    def _extract_graph_data(analysis: dict) -> tuple[list[dict], list[dict]]:
        nodes = []
        edges = []

        # JCL
        for jcl in analysis.get("jcl", []):
            jcl_name = jcl.get("file")
            if not jcl_name:
                continue
            nodes.append({"id": jcl_name, "type": "JCL"})

            for inv in jcl.get("relationships", {}).get("program_invocations", []):
                program = inv.get("program")
                if program:
                    edges.append({"from": jcl_name, "to": program, "relationship": "executes"})

            for ds in jcl.get("relationships", {}).get("dataset_references", []):
                dsn = ds.get("resolved_dsn") or ds.get("dsn")
                if dsn:
                    nodes.append({"id": dsn, "type": "DATASET"})
                    edges.append({"from": jcl_name, "to": dsn, "relationship": "uses"})

        # COBOL
        for cobol in analysis.get("cobol", []):
            cobol_name = cobol.get("file")
            if not cobol_name:
                continue
            nodes.append({"id": cobol_name, "type": "COBOL"})
            deps = cobol.get("data", {}).get("dependencies", {})
            for copybook_obj in deps.get("copybooks", []):
                copybook_name = copybook_obj.get("copybook") if isinstance(copybook_obj, dict) else copybook_obj
                if copybook_name:
                    edges.append({"from": cobol_name, "to": copybook_name, "relationship": "includes"})
            for file_obj in deps.get("files", []):
                file_name = file_obj.get("file") if isinstance(file_obj, dict) else file_obj
                if file_name:
                    edges.append({"from": cobol_name, "to": file_name, "relationship": "accesses"})
            for program_obj in deps.get("programs", []):
                program_name = program_obj.get("program") if isinstance(program_obj, dict) else program_obj
                if program_name:
                    edges.append({"from": cobol_name, "to": program_name, "relationship": "calls"})

        # Copybooks
        for copybook in analysis.get("copybooks", []):
            copy_name = copybook.get("file")
            if not copy_name:
                continue
            nodes.append({"id": copy_name, "type": "COPYBOOK"})

        # REXX
        for rexx in analysis.get("rexx", []):
            rexx_name = rexx.get("file")
            if not rexx_name:
                continue
            nodes.append({"id": rexx_name, "type": "REXX"})
            deps = rexx.get("data", {}).get("dependencies", {})
            for dep_obj in deps.get("programs", []):
                dep_name = dep_obj.get("program") if isinstance(dep_obj, dict) else dep_obj
                if dep_name:
                    edges.append({"from": rexx_name, "to": dep_name, "relationship": "calls"})

        # Deduplicate nodes
        unique_nodes = list({node["id"]: node for node in nodes}.values())
        return unique_nodes, edges

    def _to_mermaid_syntax(nodes: list[dict], edges: list[dict]) -> str:
        lines = ["graph TD"]

        style_map = {
            "JCL": "fill:#f9f,stroke:#333,stroke-width:2px",
            "COBOL": "fill:#bbf,stroke:#333,stroke-width:2px",
            "COPYBOOK": "fill:#fb9,stroke:#333,stroke-width:2px",
            "REXX": "fill:#9fb,stroke:#333,stroke-width:2px",
            "DATASET": "fill:#ddd,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5",
        }

        def sanitize_id(raw: str) -> str:
            return re.sub(r"[^a-zA-Z0-9_]", "_", raw)

        for node in nodes:
            node_id = node["id"]
            node_type = node["type"]
            clean_id = sanitize_id(node_id)
            label = f"{node_type}<br>{node_id}"
            lines.append(f"  {clean_id}[\"{label}\"]")
            if node_type in style_map:
                lines.append(f"  style {clean_id} {style_map[node_type]}")

        for edge in edges:
            from_node = sanitize_id(edge["from"])
            to_node = sanitize_id(edge["to"])
            rel = edge["relationship"]
            lines.append(f"  {from_node} -->|{rel}| {to_node}")

        return "\n".join(lines)

    return get_analysis_graph