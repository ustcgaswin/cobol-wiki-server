import re
from typing import List, Dict, Any

# --- Helper Functions ---

def _preprocess_and_split_lines(rexx_code: str) -> List[str]:
    """
    Splits REXX code into a list of lines, handling multi-line comments.
    Note: uppercases for simpler pattern matching in downstream extractors.
    """
    lines = rexx_code.upper().splitlines()
    cleaned_lines = []
    in_comment_block = False
    for line in lines:
        if '/*' in line and '*/' in line:
            # non-greedy to avoid wiping too much when multiple /* */ are on the same line
            line = re.sub(r'/\*.*?\*/', '', line)
        elif '/*' in line:
            in_comment_block = True
            line = line.split('/*')[0]
        elif '*/' in line:
            in_comment_block = False
            line = line.split('*/')[1]
        
        if not in_comment_block and line.strip():
            cleaned_lines.append(line.strip())
            
    return cleaned_lines

def _extract_script_metadata(lines: List[str], script_name: str) -> Dict[str, Any]:
    """Extracts script metadata from comments and code patterns."""
    metadata = {
        "script_name": script_name,
        "script_type": "UNKNOWN",
        "description": "No description found in comments.",
        "lines_of_code": len(lines)
    }
    
    code_text = " ".join(lines)
    if "ADDRESS ISPEXEC" in code_text:
        metadata["script_type"] = "ISPF"
    elif "ADDRESS TSO" in code_text:
        metadata["script_type"] = "TSO"
    elif "EXEC CICS" in code_text:
        metadata["script_type"] = "CICS"
    elif "EXECSQL" in code_text:
        metadata["script_type"] = "DB2"
    
    first_line = lines[0] if lines else ""
    if "REXX" in first_line:
        comment_block = []
        in_comment = False
        for line in lines[:10]:  # Check first 10 lines
            if '/*' in line:
                in_comment = True
            if in_comment:
                comment_block.append(line)
            if '*/' in line:
                break
        match = re.search(r'\/\*\s*(.*?)\s*\*\/', " ".join(comment_block), re.DOTALL)
        if match:
            metadata["description"] = re.sub(r'\s+', ' ', match.group(1)).strip()

    return metadata

def _extract_subroutines(lines: List[str]) -> List[Dict[str, Any]]:
    """Extracts subroutines/labels from the REXX script."""
    subroutines = []
    for i, line in enumerate(lines):
        match = re.match(r'^([A-Z0-9_#@$.]+):', line)
        if match:
            name = match.group(1)
            if name.upper() != 'END':
                subroutines.append({
                    "name": name,
                    "line_number": i + 1,
                    "called_by": [],
                    "calls": []
                })
    return subroutines

def _extract_control_flow(lines: List[str]) -> Dict[str, List]:
    """Extracts calls, conditionals, and loops."""
    details = {"calls": [], "conditionals": [], "loops": []}
    for i, line in enumerate(lines):
        call_match = re.search(r'CALL\s+([A-Z0-9_#@$.]+)', line)
        if call_match:
            details["calls"].append({"target": call_match.group(1), "line_number": i + 1})
        
        func_match = re.findall(r'([A-Z0-9_#@$.]+)\(', line)
        for func in func_match:
            if func not in ['SAY', 'PULL', 'ARG', 'SUBSTR', 'LENGTH', 'POS', 'WORD', 'DATE', 'TIME', 'QUEUED']:
                 details["calls"].append({"target": func, "line_number": i + 1})

        if_match = re.search(r'IF\s+(.*)\s+THEN', line)
        if if_match:
            details["conditionals"].append({"type": "IF", "condition": if_match.group(1).strip(), "line_number": i + 1})
        
        select_match = re.search(r'SELECT\s*;?', line)
        if select_match:
            details["conditionals"].append({"type": "SELECT", "line_number": i + 1})

        loop_match = re.search(r'DO\s+(.*)', line)
        if loop_match and 'END' not in line:
             details["loops"].append({"type": "DO", "condition": loop_match.group(1).strip(), "line_number": i + 1})
    return details

def _extract_external_commands(lines: List[str]) -> List[Dict[str, Any]]:
    """Extracts TSO, ISPF, and other external commands."""
    commands: List[Dict[str, Any]] = []
    current_address = "TSO"
    for i, line in enumerate(lines):
        addr_match = re.search(r'ADDRESS\s+(\w+)', line)
        if addr_match:
            current_address = addr_match.group(1)

        # Capture quoted strings as candidate commands when not obviously a built-in statement
        if not re.match(r'^(SAY|PULL|IF|DO|CALL|SELECT|ARG|END|ELSE)\b', line.strip()):
            cmd_match = re.search(r'["\'](.*?)["\']', line)
            if cmd_match:
                commands.append({"environment": current_address, "command": cmd_match.group(1), "line_number": i + 1})
    return commands

def _extract_io_operations(lines: List[str]) -> List[Dict[str, Any]]:
    """Extracts I/O related operations like SAY, PULL, EXECIO."""
    io_ops: List[Dict[str, Any]] = []
    for i, line in enumerate(lines):
        if 'SAY' in line:
            io_ops.append({"verb": "SAY", "line_number": i + 1})
        if 'PULL' in line:
            io_ops.append({"verb": "PULL", "line_number": i + 1})
        if 'PUSH' in line:
            io_ops.append({"verb": "PUSH", "line_number": i + 1})
        if 'QUEUE' in line:
            io_ops.append({"verb": "QUEUE", "line_number": i + 1})
        if 'EXECIO' in line:
            io_ops.append({"verb": "EXECIO", "details": line.strip(), "line_number": i + 1})
    return io_ops

def _build_call_graph(subroutines: List[Dict[str, Any]], calls: List[Dict[str, str]]) -> Dict[str, Any]:
    """Builds a call graph from subroutines and CALL relationships."""
    call_graph: Dict[str, Any] = {"entry_points": [], "call_hierarchy": {}}
    if not subroutines:
        return call_graph

    sub_lookup = {s["name"]: s for s in subroutines}
    sorted_subs = sorted(subroutines, key=lambda s: s['line_number'])
    
    for call in calls:
        caller_name, target_name = None, call["target"]
        for sub in reversed(sorted_subs):
            if call["line_number"] >= sub["line_number"]:
                caller_name = sub["name"]
                break
        
        if caller_name and caller_name in sub_lookup and target_name not in sub_lookup[caller_name]["calls"]:
            sub_lookup[caller_name]["calls"].append(target_name)
        if target_name in sub_lookup and caller_name and caller_name not in sub_lookup[target_name]["called_by"]:
            sub_lookup[target_name]["called_by"].append(caller_name)
    
    for sub in subroutines:
        if not sub["called_by"]:
            call_graph["entry_points"].append(sub["name"])
        # ensure unique children
        children = sorted(set(sub["calls"]))
        call_graph["call_hierarchy"][sub["name"]] = {"children": children}
    
    return call_graph

def parse_rexx_script(rexx_code: str, script_name: str = "UNTITLED") -> Dict[str, Any]:
    """Main orchestrator function to parse a REXX script string."""
    lines = _preprocess_and_split_lines(rexx_code)
    if not lines:
        return {"error": "No valid REXX code found after cleaning."}

    metadata = _extract_script_metadata(lines, script_name)
    subroutines = _extract_subroutines(lines)
    control_flow = _extract_control_flow(lines)
    call_graph = _build_call_graph(subroutines, control_flow.get("calls", []))
    external_commands = _extract_external_commands(lines)
    io_operations = _extract_io_operations(lines)

    result: Dict[str, Any] = {
        "script_metadata": metadata,
        "script_summary": {
            "overall_purpose": metadata.get("description"),
            "main_functionality": [],
        },
        "io_operations": io_operations,
        "external_commands": external_commands,
        "control_flow": {
            "subroutines": subroutines,
            "call_graph": call_graph,
            "conditionals": control_flow.get("conditionals", []),
            "loops": control_flow.get("loops", [])
        },
        "dependencies": {
            "called_scripts": [
                call for call in control_flow.get("calls", []) 
                if call["target"] not in {sub['name'] for sub in subroutines}
            ]
        }
    }
    
    return result