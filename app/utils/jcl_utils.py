"""
JCL (Job Control Language) Parsing Utility

Parses JCL files/strings into a structured JSON format focusing on relationships:
- Jobs, EXEC steps, DD statements
- IF/THEN/ELSE blocks (structure preserved)
- INCLUDE MEMBER= expansion (recursive, guarded)
- PROC definitions (captured; not expanded)
- DD back-reference resolution (e.g., DSN=*.STEP1.DD1)

No LLM usage. Outputs a deterministic relationships summary.
"""
import re
import os
from typing import List, Dict, Any, Optional, Set
from app.utils.logger import logger


# --------- Helpers: text processing ----------

def _split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Simple chunker (kept for compatibility; not used for LLM)."""
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks


def _stitch_continuation_lines(lines: List[str]) -> List[str]:
    """
    Pre-processes JCL lines to handle and stitch continuations.
    Joins lines when a statement ends with a comma outside quotes and the next card
    is a continuation (starts with '//' and does not start a new JOB/EXEC/DD/PROC/IF/ELSE/ENDIF/INCLUDE).
    """
    if not lines:
        return []

    def ends_with_unquoted_comma(s: str) -> bool:
        s = s.rstrip()
        in_quotes = False
        for ch in s:
            if ch == "'":
                in_quotes = not in_quotes
        return (not in_quotes) and s.endswith(',')

    stitched: List[str] = []
    buffer: Optional[str] = None

    for raw in lines:
        line = raw.rstrip()
        if not line:
            if buffer is not None and ends_with_unquoted_comma(buffer):
                # Still expecting continuation; skip blank line
                continue
            else:
                if buffer is not None:
                    stitched.append(buffer)
                    buffer = None
                continue

        if line.startswith('//*'):
            # Comment card; flush current buffer first
            if buffer is not None:
                stitched.append(buffer)
                buffer = None
            stitched.append(line)
            continue

        if buffer is None:
            buffer = line
            continue

        if ends_with_unquoted_comma(buffer):
            # Treat current line as continuation content if it looks like a JCL card
            cont_match = re.match(r"^//\s*(.*)$", line)
            if cont_match:
                tail = cont_match.group(1).lstrip()
                # If tail starts a new statement keyword, do not continue
                if re.match(r"(?i)^(JOB|EXEC|DD|PROC|IF|ELSE|ENDIF|INCLUDE)\b", tail):
                    stitched.append(buffer)
                    buffer = line
                else:
                    buffer = buffer.rstrip().rstrip(',') + tail
            else:
                stitched.append(buffer)
                buffer = line
        else:
            stitched.append(buffer)
            buffer = line

    if buffer is not None:
        stitched.append(buffer)
    return stitched


def _expand_include_statements(
    lines: List[str],
    jcl_lib_dir: Optional[str],
    processed_members: Optional[Set[str]] = None
) -> List[str]:
    """
    Recursively expands `INCLUDE MEMBER=` statements within a list of JCL lines.
    """
    if processed_members is None:
        processed_members = set()

    expanded_lines = []
    for line in lines:
        match = re.match(r"//\s*INCLUDE\s+MEMBER=(\S+)", line, re.IGNORECASE)
        if match:
            member_name = match.group(1)
            if member_name in processed_members:
                expanded_lines.append(f"//* SKIPPING RECURSIVE INCLUDE OF MEMBER={member_name}")
                continue

            if not jcl_lib_dir:
                expanded_lines.append(f"//* WARNING: CANNOT EXPAND INCLUDE MEMBER={member_name}. NO JCL LIBRARY DIR PROVIDED.")
                continue

            try:
                # Assume member name is the filename. Common extensions could be .jcl, .inc
                include_path = os.path.join(jcl_lib_dir, member_name)
                if not os.path.exists(include_path):
                    include_path = os.path.join(jcl_lib_dir, f"{member_name}.jcl")

                with open(include_path, "r", encoding='utf-8', errors='ignore') as f:
                    included_content = f.readlines()

                processed_members.add(member_name)
                # Recursively expand includes within the new content
                stitched_included = _stitch_continuation_lines(included_content)
                recursively_expanded = _expand_include_statements(stitched_included, jcl_lib_dir, processed_members)
                expanded_lines.extend(recursively_expanded)

            except FileNotFoundError:
                expanded_lines.append(f"//* ERROR: INCLUDE MEMBER={member_name} NOT FOUND IN {jcl_lib_dir}.")
            except Exception as e:
                expanded_lines.append(f"//* ERROR: FAILED TO EXPAND INCLUDE MEMBER={member_name}: {e}")
        else:
            expanded_lines.append(line)
    return expanded_lines


# --------- Helpers: parameter parsing ----------

def _parse_jcl_list(list_str: str) -> List[Any]:
    """
    Parses a JCL parameter sublist, handling nested parentheses.
    Example: "CYL,(150,50),RLSE" -> ['CYL', ['150', '50'], 'RLSE']
    """
    items = []
    current_item = ""
    paren_depth = 0
    in_quote = False
    for char in list_str:
        if char == "'":
            in_quote = not in_quote

        if not in_quote:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1

        if char == ',' and paren_depth == 0 and not in_quote:
            items.append(current_item.strip())
            current_item = ""
        else:
            current_item += char

    if current_item:
        items.append(current_item.strip())

    parsed_items = []
    for item in items:
        if item.startswith('(') and item.endswith(')'):
            parsed_items.append(_parse_jcl_list(item[1:-1]))
        elif item.startswith("'") and item.endswith("'"):
            parsed_items.append(item[1:-1])
        else:
            parsed_items.append(item)
    return parsed_items


def _parse_param_value(value_str: str) -> Any:
    """Parses a parameter's value, handling lists and quoted strings."""
    value_str = value_str.strip()
    if value_str.startswith("'") and value_str.endswith("'"):
        return value_str[1:-1]
    if value_str.startswith('(') and value_str.endswith(')'):
        return _parse_jcl_list(value_str[1:-1])
    return value_str


def _parse_parameters(param_str: str) -> Dict[str, Any]:
    """
    Parses a JCL parameter string into a dictionary, handling keyword,
    positional, and complex list-based parameters.
    """
    params: Dict[str, Any] = {}
    # First, split the parameter string by commas, respecting parentheses and quotes
    param_list = _parse_jcl_list(param_str)

    positional_params = []
    for item in param_list:
        if isinstance(item, str) and '=' in item:
            key, value_str = item.split('=', 1)
            params[key.upper()] = _parse_param_value(value_str)
        else:
            # It's a positional parameter or a malformed entry
            positional_params.append(item)

    if positional_params:
        params['positional'] = positional_params

    return params


# --------- Relationship enrichment ----------

def _resolve_dd_back_references(jobs: List[Dict[str, Any]]):
    """
    Post-processes parsed jobs to resolve DD back-references (e.g., DSN=*.STEP1.DD1).
    This operates in-place on the jobs data structure.
    """
    for job in jobs:
        # Build a map of all steps in the job for easy lookup.
        steps_map: Dict[str, Dict[str, Any]] = {}

        def find_all_steps(items: List[Dict[str, Any]]):
            for item in items:
                if item.get('type') == 'EXEC' and item.get('name'):
                    steps_map[item['name']] = item
                elif item.get('type') == 'IF':
                    find_all_steps(item.get('then_items', []))
                    find_all_steps(item.get('else_items', []))

        find_all_steps(job.get('items', []))

        # Iterate through DDs and resolve back-references.
        def resolve_in_items(items: List[Dict[str, Any]]):
            for item in items:
                if item.get('type') == 'EXEC':
                    for dd in item.get('dd_statements', []):
                        if dd.get('type') != 'DD':
                            continue

                        params = dd.get('parameters', {})
                        dsn = params.get('DSN')

                        if isinstance(dsn, str) and dsn.startswith('*.'):
                            parts = dsn.split('.')
                            if len(parts) == 3:  # Format: *.stepname.ddname
                                _, target_step_name, target_dd_name = parts

                                target_step = steps_map.get(target_step_name)
                                resolved_dsn = None
                                resolution_error = None

                                if target_step:
                                    target_dd = next(
                                        (d for d in target_step.get('dd_statements', [])
                                         if d.get('type') == 'DD' and d.get('name') == target_dd_name),
                                        None
                                    )
                                    if target_dd:
                                        resolved_dsn = target_dd.get('parameters', {}).get('DSN')
                                        if not resolved_dsn:
                                            resolution_error = f"Target DD '{target_dd_name}' in step '{target_step_name}' has no DSN parameter."
                                    else:
                                        resolution_error = f"Target DD '{target_dd_name}' not found in step '{target_step_name}'."
                                else:
                                    resolution_error = f"Target step '{target_step_name}' not found in job."

                                if resolved_dsn:
                                    params['resolved_dsn'] = resolved_dsn
                                if resolution_error:
                                    params['resolution_error'] = resolution_error

                elif item.get('type') == 'IF':
                    resolve_in_items(item.get('then_items', []))
                    resolve_in_items(item.get('else_items', []))

        resolve_in_items(job.get('items', []))


def _collect_relationships(jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize relationships across parsed jobs:
    - program_invocations: job/step -> PGM= or PROC
    - dataset_references: job/step/dd -> DSN (and resolved_dsn if back-referenced)
    - programs_by_job, datasets_by_job
    """
    program_invocations: List[Dict[str, Any]] = []
    dataset_references: List[Dict[str, Any]] = []
    programs_by_job: Dict[str, List[str]] = {}
    datasets_by_job: Dict[str, List[str]] = {}

    def iter_exec_steps(items: List[Dict[str, Any]]):
        for item in items:
            if item.get('type') == 'EXEC':
                yield item
            elif item.get('type') == 'IF':
                yield from iter_exec_steps(item.get('then_items', []))
                yield from iter_exec_steps(item.get('else_items', []))

    for job in jobs:
        job_name = job.get('name')
        job_programs: Set[str] = set()
        job_datasets: Set[str] = set()

        for step in iter_exec_steps(job.get('items', [])):
            step_name = step.get('name')
            params = step.get('parameters', {})
            # EXEC PGM=... or EXEC PROC=...
            if 'PGM' in params:
                pgm = params.get('PGM')
                program_invocations.append({
                    "job": job_name,
                    "step": step_name,
                    "exec_type": "PGM",
                    "program": pgm
                })
                if isinstance(pgm, str):
                    job_programs.add(pgm)
            elif 'PROC' in params:
                proc = params.get('PROC')
                program_invocations.append({
                    "job": job_name,
                    "step": step_name,
                    "exec_type": "PROC",
                    "proc": proc
                })
                if isinstance(proc, str):
                    job_programs.add(f"PROC:{proc}")

            for dd in step.get('dd_statements', []):
                if dd.get('type') != 'DD':
                    continue
                dd_name = dd.get('name')
                dd_params = dd.get('parameters', {})
                dsn = dd_params.get('DSN')
                resolved = dd_params.get('resolved_dsn')
                disp = dd_params.get('DISP')
                dataset_references.append({
                    "job": job_name,
                    "step": step_name,
                    "dd": dd_name,
                    "dsn": dsn,
                    "resolved_dsn": resolved,
                    "disp": disp
                })
                if isinstance(resolved, str):
                    job_datasets.add(resolved)
                elif isinstance(dsn, str):
                    job_datasets.add(dsn)

        programs_by_job[job_name] = sorted(job_programs)
        datasets_by_job[job_name] = sorted(job_datasets)

    return {
        "program_invocations": program_invocations,
        "dataset_references": dataset_references,
        "programs_by_job": programs_by_job,
        "datasets_by_job": datasets_by_job,
    }


# --------- Main parsing ----------

def parse_jcl_to_json(jcl_content: str, jcl_lib_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Parses a JCL string into a structured JSON format, identifying jobs,
    steps, and DD statements. It handles continuations, includes, PROCs,
    and IF/THEN/ELSE logic. Finally, it resolves DD back-references.

    Args:
        jcl_content: A string containing the JCL to be parsed.
        jcl_lib_dir: Optional path to a directory containing INCLUDE members.

    Returns:
        A dictionary with keys:
         - "jobs": list of parsed job objects
         - "relationships": cross-reference summary (programs, datasets, etc.)
    """
    raw_lines = jcl_content.splitlines()
    stitched_lines = _stitch_continuation_lines(raw_lines)
    lines = _expand_include_statements(stitched_lines, jcl_lib_dir)

    jobs: List[Dict[str, Any]] = []
    current_job: Optional[Dict[str, Any]] = None
    current_step: Optional[Dict[str, Any]] = None
    in_stream_data_dd: Optional[Dict[str, Any]] = None
    last_dd_name: Optional[str] = None
    if_stack: List[Dict[str, Any]] = []
    current_proc_def: Optional[Dict[str, Any]] = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Handle in-stream data collection
        if in_stream_data_dd:
            if line.strip() == '/*' or line.startswith('//'):
                in_stream_data_dd = None
            else:
                in_stream_data_dd['in_stream_data'].append(line)
                continue

        # Handle PROC definition
        if current_proc_def:
            if re.match(r"//\s*PEND\b", line, re.IGNORECASE):
                if current_job:
                    current_job["proc_definitions"][current_proc_def['name']] = current_proc_def
                current_proc_def = None
            else:
                current_proc_def['body'].append(line)
            continue

        if line.startswith('//*'):
            comment = {"type": "COMMENT", "text": line}
            container = if_stack[-1]['current_branch'] if if_stack else (current_job['items'] if current_job else None)
            if current_step:
                current_step.setdefault('dd_statements', []).append(comment)
            elif container is not None:
                container.append(comment)
            elif current_job:
                current_job.setdefault('comments', []).append(line)
            continue

        # Handle Control Statements
        if_match = re.match(r"//\s*IF\s+(.*)\s+THEN\b", line, re.IGNORECASE)
        else_match = re.match(r"//\s*ELSE\b", line, re.IGNORECASE)
        endif_match = re.match(r"//\s*ENDIF\b", line, re.IGNORECASE)

        if if_match:
            container = if_stack[-1]['current_branch'] if if_stack else (current_job['items'] if current_job else None)
            if container is not None:
                if_block = {
                    "type": "IF", "raw_card": line,
                    "condition_raw": if_match.group(1).strip('()'),
                    "then_items": [], "else_items": []
                }
                container.append(if_block)
                if_stack.append({"block": if_block, "current_branch": if_block['then_items']})
            continue
        if else_match and if_stack:
            if_stack[-1]['block']['raw_else_card'] = line
            if_stack[-1]['current_branch'] = if_stack[-1]['block']['else_items']
            continue
        if endif_match and if_stack:
            if_stack[-1]['block']['raw_endif_card'] = line
            if_stack.pop()
            continue

        # Handle JCL Statements (JOB, EXEC, DD, PROC)
        match = re.match(r"//(\S*)\s+(JOB|EXEC|DD|PROC)\s*(.*)", line, re.IGNORECASE)
        if not match:
            match = re.match(r"//\s+(DD)\s*(.*)", line, re.IGNORECASE)
            if not match:
                continue
            name, stmt_type, params_str = '', match.group(1).upper(), match.group(2)
        else:
            name, stmt_type, params_str = match.groups()
            stmt_type = stmt_type.upper()

        params = _parse_parameters(params_str)
        container = if_stack[-1]['current_branch'] if if_stack else (current_job['items'] if current_job else None)

        if stmt_type == 'JOB':
            current_job = {
                "name": name, "type": "JOB", "raw_card": line,
                "parameters": params, "comments": [], "items": [], "proc_definitions": {}
            }
            jobs.append(current_job)
            current_step = None
            last_dd_name = None
            if_stack = []

        elif stmt_type == 'PROC':
            if not current_job:
                continue
            current_proc_def = {
                "name": name, "type": "PROC", "raw_card": line,
                "parameters": params, "body": []
            }

        elif stmt_type == 'EXEC':
            if not current_job or container is None:
                continue
            current_step = {
                "name": name, "type": "EXEC", "raw_card": line,
                "parameters": params, "dd_statements": []
            }
            container.append(current_step)
            last_dd_name = None

        elif stmt_type == 'DD':
            if not current_step:
                continue
            is_concatenated = (name == '')
            dd_name = name if not is_concatenated else last_dd_name

            dd_statement = {
                "name": dd_name, "type": "DD", "raw_card": line,
                "parameters": params, "is_concatenated": is_concatenated
            }
            current_step['dd_statements'].append(dd_statement)

            if not is_concatenated:
                last_dd_name = name

            pos_params = dd_statement['parameters'].get('positional')
            if pos_params and pos_params[0] in ['*', 'DATA']:
                dd_statement['in_stream_data'] = []
                in_stream_data_dd = dd_statement

    # Resolve DD back-references
    _resolve_dd_back_references(jobs)

    # Build relationships summary (deterministic, no LLM)
    relationships = _collect_relationships(jobs)

    return {"jobs": jobs, "relationships": relationships}


def parse_jcl_file_to_json(jcl_path: str, jcl_lib_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Reads and parses a JCL file into a structured JSON format.

    Args:
        jcl_path: The path to the JCL file.
        jcl_lib_dir: Optional path to a directory for resolving INCLUDEs.
                     If not provided, the directory of the JCL file is used.

    Returns:
        A dictionary representing the parsed JCL and relationships, or an error dictionary.
    """
    try:
        if not jcl_lib_dir:
            jcl_lib_dir = os.path.dirname(jcl_path)
        with open(jcl_path, "r", encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return parse_jcl_to_json(content, jcl_lib_dir)
    except FileNotFoundError:
        return {"error": f"File not found: {jcl_path}"}
    except Exception as e:
        logger.error(f"An error occurred while processing {jcl_path}: {e}", exc_info=True)
        return {"error": f"An error occurred while processing {jcl_path}: {str(e)}"}