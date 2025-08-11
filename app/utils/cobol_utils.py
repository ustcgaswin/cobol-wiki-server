import re
from typing import List, Dict, Any
from app.utils.logger import logger

def _clean_cobol_line(line: str) -> str:
    """
    Removes sequence numbers and handles comment lines for fixed-format COBOL.
    Returns an empty string for comment lines or the cleaned content.
    """
    # Fixed-format columns: 1-6 sequence, 7 indicator, 8-11 Area A, 12-72 Area B
    if not line:
        return ""
    raw = line.rstrip("\n\r")

    if len(raw) < 7:
        return raw.strip()

    indicator = raw[6]  # column 7
    # Comment (*) or page-eject (/) or debug (D) lines
    if indicator in ("*", "/", "D"):
        return ""

    # Content begins at column 8 (index 7) up to 72
    content = raw[7:72]

    # Remove inline comments starting with *>
    ic = content.find("*>")
    if ic != -1:
        content = content[:ic]

    return content.strip()

def _preprocess_and_split_lines(cobol_code: str) -> List[str]:
    """
    Cleans and splits COBOL code into a list of relevant, non-empty lines.
    """
    lines = cobol_code.upper().splitlines()
    cleaned_lines = [_clean_cobol_line(line) for line in lines]
    return [line for line in cleaned_lines if line]

def _extract_program_id(lines: List[str]) -> str | None:
    """Extracts the PROGRAM-ID from the IDENTIFICATION DIVISION."""
    for line in lines:
        match = re.search(r'PROGRAM-ID\.\s*([A-Z0-9-]+)', line)
        if match:
            return match.group(1)
    return None

def _extract_program_metadata(lines: List[str]) -> Dict[str, Any]:
    """Extracts program metadata from IDENTIFICATION DIVISION."""
    metadata = {
        "program_id": None,
        "program_type": "BATCH",  # Default
        "remarks": "",
        "lines_of_code": len(lines)
    }
    
    for line in lines:
        if match := re.search(r'PROGRAM-ID\.\s*([A-Z0-9-]+)', line):
            metadata["program_id"] = match.group(1)
        elif match := re.search(r'REMARKS\.\s*(.*)', line):
            metadata["remarks"] = match.group(1).strip()
    
    # Determine program type based on code patterns
    code_text = " ".join(lines)
    if 'EXEC CICS' in code_text:
        metadata["program_type"] = "ONLINE"
    elif any(keyword in code_text for keyword in ['LINKAGE SECTION', 'USING']):
        metadata["program_type"] = "SUBROUTINE"
    
    return metadata

def _extract_select_statements(lines: List[str]) -> List[Dict[str, Any]]:
    """Extracts SELECT statements from the ENVIRONMENT DIVISION."""
    selects = []
    full_text = " ".join(lines)
    pattern = re.compile(r'SELECT\s+(.*?)\s+ASSIGN\s+TO\s+(.*?)(?=\s+FILE\s+STATUS|\s+ORGANIZATION|\.)', re.DOTALL)
    
    for match in pattern.finditer(full_text):
        logical_name = re.sub(r'\s+', ' ', match.group(1)).strip()
        assign_to = re.sub(r'\s+', ' ', match.group(2)).strip()
        
        # Extract additional file attributes
        file_info = {
            "logical_name": logical_name,
            "physical_name": assign_to,
            "organization": "SEQUENTIAL",  # Default
            "access_mode": "SEQUENTIAL",
            "record_key": None,
            "file_status": None,
            "operations": [],
            "usage_pattern": "INPUT_OUTPUT"
        }
        
        # Look for additional attributes
        select_text = full_text[match.start():match.end() + 200]  # Look ahead for more details
        if org_match := re.search(r'ORGANIZATION\s+IS\s+(\w+)', select_text):
            file_info["organization"] = org_match.group(1)
        if access_match := re.search(r'ACCESS\s+MODE\s+IS\s+(\w+)', select_text):
            file_info["access_mode"] = access_match.group(1)
        if key_match := re.search(r'RECORD\s+KEY\s+IS\s+([A-Z0-9-]+)', select_text):
            file_info["record_key"] = key_match.group(1)
        if status_match := re.search(r'FILE\s+STATUS\s+IS\s+([A-Z0-9-]+)', select_text):
            file_info["file_status"] = status_match.group(1)
            
        selects.append(file_info)
    
    return selects

def _extract_detailed_data_structures(lines: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Extracts detailed data structures with hierarchy and field information."""
    structures = {"working_storage": [], "linkage_section": [], "file_section": []}
    in_data_division = False
    current_section = None
    current_01_item = None
    
    for i, line in enumerate(lines):
        if 'DATA DIVISION.' in line:
            in_data_division = True
            continue
        if not in_data_division:
            continue
        if 'PROCEDURE DIVISION' in line:
            break
            
        if 'WORKING-STORAGE SECTION.' in line:
            current_section = "working_storage"
            continue
        if 'LINKAGE SECTION.' in line:
            current_section = "linkage_section"
            continue
        if 'FILE SECTION.' in line:
            current_section = "file_section"
            continue
            
        if current_section:
            # Match data items with level numbers
            match = re.match(r'^\s*(\d{2})\s+([A-Z0-9-]+)(?:\s+(.*?))?\.?\s*$', line)
            if match:
                level = int(match.group(1))
                name = match.group(2)
                definition = match.group(3) if match.group(3) else ""
                
                item = {
                    "name": name,
                    "level": f"{level:02d}",
                    "line_number": i + 1,
                    "type": definition if 'PIC' in definition else None,
                    "usage": "DISPLAY",
                    "description": "",
                    "children": []
                }
                
                # Extract additional attributes
                if 'USAGE' in definition:
                    usage_match = re.search(r'USAGE\s+([\w-]+)', definition)
                    if usage_match:
                        item["usage"] = usage_match.group(1)
                
                if level == 1:
                    current_01_item = item
                    structures[current_section].append(item)
                elif current_01_item and level > 1:
                    # Simplified nesting
                    parent = current_01_item
                    if parent["children"] and int(parent["children"][-1]["level"]) < level:
                        parent = parent["children"][-1]
                    parent["children"].append(item)
    
    return structures

def _extract_copy_statements(lines: List[str]) -> List[Dict[str, Any]]:
    """Extracts COPY statements with additional details."""
    copybooks = []
    for i, line in enumerate(lines):
        match = re.search(r'COPY\s+([A-Z0-9-]+)', line)
        if match:
            copybook = {
                "name": match.group(1),
                "line_number": i + 1,
                "type": "DATA_STRUCTURE"  # Heuristic; could be refined
            }
            copybooks.append(copybook)
    return copybooks

def _extract_procedure_division_details(lines: List[str]) -> Dict[str, List]:
    """Extracts detailed procedure division information."""
    details = {
        "calls": [],            # External program calls (CALL "PROG")
        "performs": [],         # PERFORM paragraph edges (caller resolved later)
        "io_operations": [],    # Verb + target (READ/WRITE/etc.)
        "cics_commands": [],    # EXEC CICS ... END-EXEC blocks
        "conditional_logic": [],# IF/EVALUATE snippets
        "loops": []             # PERFORM ... UNTIL
    }
    
    in_cics_block = False
    current_cics_block = []
    cics_start_line = 0
    
    for i, line in enumerate(lines):
        # CICS commands
        if 'EXEC CICS' in line:
            in_cics_block = True
            cics_start_line = i + 1
            clean_part = re.sub(r'EXEC\s+CICS', '', line, 1).strip()
            current_cics_block.append(clean_part)
            if 'END-EXEC' in line:
                in_cics_block = False
                command_text = " ".join(current_cics_block).replace('END-EXEC', '').strip()
                
                cics_cmd = {
                    "command": command_text.split()[0] if command_text else "",
                    "line_number": cics_start_line,
                    "full_command": command_text
                }
                
                if 'MAP' in command_text:
                    map_match = re.search(r'MAP\s*\(\s*[\'"]?([A-Z0-9]+)[\'"]?\s*\)', command_text)
                    if map_match:
                        cics_cmd["map_name"] = map_match.group(1)
                        cics_cmd["purpose"] = f"{'Send' if 'SEND' in command_text else 'Receive'} map operation"
                
                details["cics_commands"].append(cics_cmd)
                current_cics_block = []
            continue
            
        if in_cics_block:
            current_cics_block.append(line.strip())
            if 'END-EXEC' in line:
                in_cics_block = False
                command_text = " ".join(current_cics_block).replace('END-EXEC', '').strip()
                details["cics_commands"].append({
                    "command": command_text.split()[0] if command_text else "",
                    "line_number": cics_start_line,
                    "full_command": command_text
                })
                current_cics_block = []
            continue
        
        # CALL statements (static)
        call_match = re.search(r'CALL\s+([\'"])([A-Z0-9-]+)\1', line)
        if call_match:
            details["calls"].append({
                "program_name": call_match.group(2),
                "line_number": i + 1,
                "call_type": "STATIC",
                "parameters": []
            })
            continue
        
        # PERFORM statements (target paragraph)
        perform_match = re.search(r'PERFORM\s+([A-Z0-9-]+)', line)
        if perform_match:
            details["performs"].append({
                "target": perform_match.group(1),
                "line_number": i + 1
            })
            continue
        
        # I/O operations
        io_match = re.search(r'(OPEN|CLOSE|READ|WRITE|REWRITE|DELETE|START)\s+([A-Z0-9-]+)', line)
        if io_match:
            details["io_operations"].append({
                "verb": io_match.group(1),
                "target": io_match.group(2),
                "line_number": i + 1
            })
            continue
        
        # Conditional logic
        if_match = re.search(r'IF\s+(.*?)(?:\s+THEN)?', line)
        if if_match:
            details["conditional_logic"].append({
                "type": "IF",
                "condition": if_match.group(1).strip(),
                "line_number": i + 1
            })
            continue
        
        evaluate_match = re.search(r'EVALUATE\s+(.*)', line)
        if evaluate_match:
            details["conditional_logic"].append({
                "type": "EVALUATE",
                "subject": evaluate_match.group(1).strip(),
                "line_number": i + 1
            })
            continue
        
        # Loops
        perform_until_match = re.search(r'PERFORM\s+.*\s+UNTIL\s+(.*)', line)
        if perform_until_match:
            details["loops"].append({
                "type": "PERFORM_UNTIL",
                "condition": perform_until_match.group(1).strip(),
                "line_number": i + 1
            })
            continue
    
    return details

def _extract_sql_commands(lines: List[str]) -> Dict[str, Any]:
    """Extracts detailed SQL commands from the PROCEDURE DIVISION."""
    sql_commands = []
    tables = []
    cursors = []
    
    in_sql_block = False
    current_sql_block = []
    sql_start_line = 0
    
    for i, line in enumerate(lines):
        if re.search(r'EXEC\s+SQL', line):
            in_sql_block = True
            sql_start_line = i + 1
            clean_part = re.sub(r'.*EXEC\s+SQL', '', line, 1).strip()
            current_sql_block.append(clean_part)
            if 'END-EXEC' in line:
                in_sql_block = False
                command_text = " ".join(current_sql_block).replace('END-EXEC', '').strip()
                if command_text:
                    sql_cmd = _parse_sql_command(command_text, sql_start_line)
                    sql_commands.append(sql_cmd)
                    
                    # Extract table information summary
                    if sql_cmd.get("table_name"):
                        table_found = False
                        for table in tables:
                            if table["table_name"] == sql_cmd["table_name"]:
                                if sql_cmd["operation"] not in table["operations"]:
                                    table["operations"].append(sql_cmd["operation"])
                                table_found = True
                                break
                        if not table_found:
                            tables.append({
                                "table_name": sql_cmd["table_name"],
                                "operations": [sql_cmd["operation"]],
                                "columns_referenced": sql_cmd.get("columns", []),
                                "join_relationships": []
                            })
                
                current_sql_block = []
            continue
            
        if in_sql_block:
            current_sql_block.append(line.strip())
            if 'END-EXEC' in line:
                in_sql_block = False
                command_text = " ".join(current_sql_block).replace('END-EXEC', '').strip()
                if command_text:
                    sql_cmd = _parse_sql_command(command_text, sql_start_line)
                    sql_commands.append(sql_cmd)
                current_sql_block = []
    
    return {
        "sql_commands": sql_commands,
        "sql_tables": tables,
        "cursors": cursors
    }

def _parse_sql_command(sql_text: str, line_number: int) -> Dict[str, Any]:
    """Parse individual SQL command for detailed information."""
    sql_text = sql_text.strip()
    operation = sql_text.split()[0].upper() if sql_text else ""
    
    cmd = {
        "command": sql_text,
        "operation": operation,
        "line_number": line_number,
        "table_name": None,
        "columns": []
    }
    
    # Extract table name
    if operation in ["SELECT", "INSERT", "UPDATE", "DELETE"]:
        if operation == "SELECT":
            from_match = re.search(r'FROM\s+([A-Z0-9_]+)', sql_text, re.IGNORECASE)
            if from_match:
                cmd["table_name"] = from_match.group(1)
        elif operation == "INSERT":
            into_match = re.search(r'INTO\s+([A-Z0-9_]+)', sql_text, re.IGNORECASE)
            if into_match:
                cmd["table_name"] = into_match.group(1)
        elif operation in ["UPDATE", "DELETE"]:
            table_match = re.search(rf'{operation}\s+([A-Z0-9_]+)', sql_text, re.IGNORECASE)
            if table_match:
                cmd["table_name"] = table_match.group(1)
    
    return cmd

def _extract_procedure_paragraphs(lines: List[str]) -> List[Dict[str, Any]]:
    """Extracts paragraph headers from the PROCEDURE DIVISION."""
    paragraphs = []
    reserved_keywords = {
        'PROCEDURE', 'DIVISION', 'SECTION', 'INPUT-OUTPUT', 'FILE-CONTROL',
        'DATA', 'WORKING-STORAGE', 'LINKAGE', 'IDENTIFICATION', 'ENVIRONMENT'
    }
    
    in_procedure_division = False
    for i, line in enumerate(lines):
        if 'PROCEDURE DIVISION' in line:
            in_procedure_division = True
            continue
        if not in_procedure_division:
            continue

        # A paragraph name starts in Area A and ends with a period.
        match = re.match(r'^([A-Z0-9-]+)\.', line.strip())
        if match:
            name = match.group(1)
            if name not in reserved_keywords and not line.strip().startswith('END '):
                para = {
                    "name": name,
                    "line_number": i + 1,
                    "type": "MAIN" if "MAIN" in name else "SUBROUTINE",
                    "complexity": 1,  # Placeholder
                    "called_by": [],
                    "calls": []
                }
                paragraphs.append(para)
    
    return paragraphs

def _build_call_graph(paragraphs: List[Dict[str, Any]], performs: List[Dict[str, str]]) -> Dict[str, Any]:
    """Builds a call graph from paragraph and PERFORM relationships."""
    call_graph = {
        "entry_points": [],
        "call_hierarchy": {}
    }
    
    if not paragraphs:
        return call_graph

    # Create paragraph lookup
    para_lookup = {p["name"]: p for p in paragraphs}
    sorted_paras = sorted(paragraphs, key=lambda p: p['line_number'])
    
    # Build relationships from PERFORM statements
    for perform in performs:
        caller = None
        target = perform["target"]
        
        # Find the calling paragraph by checking which paragraph the PERFORM line falls into.
        for para in reversed(sorted_paras):
            if perform["line_number"] >= para["line_number"]:
                caller = para["name"]
                break
        
        if caller and caller in para_lookup:
            if target not in para_lookup[caller]["calls"]:
                para_lookup[caller]["calls"].append(target)
        
        if target in para_lookup:
            if caller and caller not in para_lookup[target]["called_by"]:
                para_lookup[target]["called_by"].append(caller)
    
    # Identify entry points (paragraphs not called by others)
    for para in paragraphs:
        if not para["called_by"]:
            call_graph["entry_points"].append(para["name"])
        
        call_graph["call_hierarchy"][para["name"]] = {
            "children": para["calls"],
            "depth": 0  # Placeholder for hierarchy depth
        }
    
    return call_graph

def _aggregate_file_io(selects: List[Dict[str, Any]], io_ops: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate file I/O by logical file name using SELECT and I/O verbs found.
    Returns dict logical_name -> { physical_name, operations: {READ, WRITE, ...} }
    """
    result: Dict[str, Any] = {}
    # Initialize from SELECTs
    for sel in selects:
        lname = sel.get("logical_name", "").upper()
        if not lname:
            continue
        result[lname] = {
            "physical_name": sel.get("physical_name"),
            "operations": set()
        }
    # Add operations from I/O verbs
    for op in io_ops:
        target = op.get("target", "").upper()
        verb = op.get("verb", "").upper()
        if not target or not verb:
            continue
        if target not in result:
            result[target] = {
                "physical_name": None,
                "operations": set()
            }
        result[target]["operations"].add(verb)
    # Convert operation sets to sorted lists
    for lname in list(result.keys()):
        ops = sorted(list(result[lname]["operations"]))
        result[lname]["operations"] = ops
    return result

def parse_cobol_program(cobol_code: str) -> Dict[str, Any]:
    """
    Parse a COBOL program and extract relationships only (no LLM):
    - program metadata (id, type, LOC)
    - copybooks used
    - paragraph list and perform-based call graph
    - external program calls (CALL)
    - file I/O (SELECT and verbs) and SQL table usage
    - CICS commands
    """
    lines = _preprocess_and_split_lines(cobol_code)
    if not lines:
        return {"error": "No valid COBOL code found after cleaning."}

    # Extract program metadata
    program_metadata = _extract_program_metadata(lines)
    
    # Extract data structures
    data_structures = _extract_detailed_data_structures(lines)
    
    # Extract file information (ENVIRONMENT DIVISION)
    select_statements = _extract_select_statements(lines)
    
    # Extract copy statements
    copy_statements = _extract_copy_statements(lines)
    
    # Initialize defaults
    procedure_details = {"calls": [], "performs": [], "io_operations": [], "cics_commands": [], "conditional_logic": [], "loops": []}
    sql_info = {"sql_commands": [], "sql_tables": [], "cursors": []}
    paragraphs = []
    call_graph = {"entry_points": [], "call_hierarchy": {}}
    
    try:
        proc_div_start_line_num = next(i for i, line in enumerate(lines) if 'PROCEDURE DIVISION' in line)
        proc_div_lines = lines[proc_div_start_line_num:]
        
        procedure_details = _extract_procedure_division_details(proc_div_lines)
        sql_info = _extract_sql_commands(proc_div_lines)
        paragraphs = _extract_procedure_paragraphs(lines)  # Absolute line numbers
        call_graph = _build_call_graph(paragraphs, procedure_details.get("performs", []))

    except StopIteration:
        logger.warning("No PROCEDURE DIVISION found in COBOL program")

    # Aggregate relationships
    called_programs = sorted({c["program_name"] for c in procedure_details.get("calls", []) if c.get("program_name")})
    file_io = _aggregate_file_io(select_statements, procedure_details.get("io_operations", []))
    perform_edges = []
    for name, node in call_graph.get("call_hierarchy", {}).items():
        for child in node.get("children", []):
            perform_edges.append({"from": name, "to": child})

    result = {
        "program_metadata": program_metadata,

        "relationships": {
            "called_programs": called_programs,
            "perform_edges": perform_edges,
            "copybooks": [c["name"] for c in copy_statements],
            "files": file_io,  # logical_name -> {physical_name, operations[]}
            "sql_tables": sql_info.get("sql_tables", []),
            "cics_commands": procedure_details.get("cics_commands", [])
        },

        "data_structures": data_structures,

        "files_and_databases": {
            "files": select_statements,
            "databases": {
                "sql_tables": sql_info.get("sql_tables", []),
                "cursors": sql_info.get("cursors", [])
            }
        },

        "control_flow": {
            "paragraphs": paragraphs,
            "call_graph": call_graph,
            "conditional_logic": procedure_details.get("conditional_logic", []),
            "loops": procedure_details.get("loops", [])
        },

        "dependencies": {
            "copybooks": copy_statements,
            "external_resources": []  # placeholder for JCL/dataset correlations
        }
    }
    
    return result