import re
import os
from typing import List, Dict, Any, Optional, Set


def _extract_line_parts(line: str) -> tuple[str, str]:
    """
    Returns (content_area, indicator_char) for a COBOL copybook line.
    - indicator_char: column 7 (index 6) or space if not present
    - content_area: columns 8-72 (index 7..72) for fixed-format, else stripped line
    Strips inline comments starting with '*>'.
    """
    indicator = line[6] if len(line) > 6 else ' '
    # Comment, page eject, or debug lines in indicator column
    if indicator in ('*', '/', 'D'):
        return ("", indicator)

    # Fixed-format content area (exclude indicator col)
    content = line[7:72] if len(line) > 7 else line.strip()

    # Strip inline comments "*>"
    ic = content.find("*>")
    if ic != -1:
        content = content[:ic]

    return (content.rstrip(), indicator)


def preprocess_copybook_lines(
    lines: List[str],
    copybook_dir: Optional[str] = None,
    _visited: Optional[Set[str]] = None
) -> List[str]:
    """
    Preprocess copybook lines to handle COPY statements and continuation lines.
    Returns a list of clean, ready-to-parse COBOL statements.
    - Handles fixed-format indicator column and inline comments.
    - Handles continuation lines indicated by '-' in column 7.
    - Expands COPY statements from copybook_dir if provided (tries several name variants).
    - Prevents infinite recursion using a visited set.
    """
    processed: List[str] = []
    buffer: str = ""
    if _visited is None:
        _visited = set()

    def flush_buffer():
        nonlocal buffer
        stmt = buffer.strip()
        buffer = ""
        if not stmt:
            return

        upper = stmt.upper()
        if upper.startswith("COPY "):
            # Expand copybook include (ignoring REPLACING semantics for now)
            tokens = stmt.split()
            if len(tokens) >= 2:
                copybook_name = tokens[1].rstrip('.')
                if copybook_dir:
                    candidates = [
                        os.path.join(copybook_dir, copybook_name),
                        os.path.join(copybook_dir, f"{copybook_name}.cpy"),
                        os.path.join(copybook_dir, f"{copybook_name}.CPY"),
                        os.path.join(copybook_dir, f"{copybook_name}.cob"),
                        os.path.join(copybook_dir, f"{copybook_name}.COB"),
                    ]
                    for path in candidates:
                        if os.path.exists(path) and os.path.isfile(path):
                            try:
                                real = os.path.realpath(path)
                                if real in _visited:
                                    # Avoid infinite recursion
                                    processed.append(f"* Skipping recursive include: {copybook_name}")
                                    break
                                _visited.add(real)
                                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                                    processed.extend(preprocess_copybook_lines(f.readlines(), copybook_dir, _visited))
                                break
                            except Exception:
                                processed.append(f"* Failed to include copybook: {copybook_name}")
                                break
                    else:
                        processed.append(f"* Unable to include copybook: {copybook_name}")
            # Do not append the COPY line itself
        else:
            processed.append(stmt)

    for raw in lines:
        content, indicator = _extract_line_parts(raw)
        if not content:
            continue

        starts_with_level = bool(re.match(r'^\s*\d{2}\b', content))

        if indicator == '-':
            # Continuation line: append content to current buffer
            buffer = (buffer + " " + content.lstrip()) if buffer else content.lstrip()
        else:
            if starts_with_level:
                # New logical statement; flush previous if any
                if buffer:
                    flush_buffer()
                buffer = content
            else:
                # Continuation without indicator (tolerate)
                buffer = (buffer + " " + content.lstrip()) if buffer else content

        # Heuristic: end statement at terminal period
        if content.strip().endswith('.'):
            flush_buffer()

    # Flush remaining
    if buffer.strip():
        flush_buffer()

    return processed


def parse_copybook_lines(lines: List[str]) -> Dict[str, Any]:
    """
    Parse clean COBOL statements into a nested Python dictionary representing the hierarchy.
    Processes all 01-level records found. Captures 88-level condition names on the
    immediately preceding data item (group or scalar). Skips 66-level RENAMES.
    """
    stack: List[Dict[str, Any]] = []  # holds group nodes only
    output: Dict[str, Any] = {}
    filler_count = 0
    last_data_node: Optional[Dict[str, Any]] = None  # track last defined node for 88-level attachment

    def add_to_dict(level, name, attrs, is_group,
                    occurs=None, occurs_min=None, occurs_max=None,
                    redefines=None, depending_on=None) -> Dict[str, Any]:
        nonlocal filler_count
        if name.upper() == "FILLER":
            filler_count += 1
            name = f"FILLER_{filler_count}"

        node: Dict[str, Any] = {
            "level": level,
            "name": name,
            "is_group": is_group,
            "raw_attrs": " ".join(attrs)
        }
        if occurs is not None:
            node["occurs"] = occurs
        if occurs_min is not None:
            node["occurs_min"] = occurs_min
        if occurs_max is not None:
            node["occurs_max"] = occurs_max
        if redefines:
            node["redefines"] = redefines
        if depending_on:
            node["depending_on"] = depending_on
        if is_group:
            node["children"] = {}

        # Adjust stack to current level (simple nesting, not full COBOL scoping rules)
        while stack and stack[-1]["level"] >= level:
            stack.pop()

        if stack:
            stack[-1]["children"][name] = node
        else:
            output[name] = node

        if is_group:
            stack.append(node)

        return node

    def _parse_88_values(attrs: List[str]) -> List[str]:
        # Collect tokens after VALUE/VALUES; support quoted literals and numeric tokens
        vals: List[str] = []
        i = 0
        # Find VALUE or VALUES
        while i < len(attrs) and attrs[i].upper() not in ("VALUE", "VALUES"):
            i += 1
        if i >= len(attrs):
            return vals
        i += 1
        # Read the rest as values; keep quoted groups intact
        cur: List[str] = []
        in_quote = False
        quote_char = ""
        while i < len(attrs):
            tok = attrs[i]
            u = tok.upper()
            if u == ".":
                break
            if not in_quote and tok and tok[0] in ("'", '"'):
                in_quote = True
                quote_char = tok[0]
                cur = [tok]
            elif in_quote:
                cur.append(tok)
                if tok.endswith(quote_char):
                    vals.append(" ".join(cur).strip().strip("."))
                    cur = []
                    in_quote = False
                    quote_char = ""
            else:
                vals.append(tok.strip().strip("."))
            i += 1
        if cur:
            vals.append(" ".join(cur).strip().strip("."))
        return vals

    for line in lines:
        clean = line.strip()
        if not clean or clean.startswith("*"):
            continue

        tokens = clean.split()
        if not tokens or not tokens[0].isdigit():
            continue

        # Level and name
        level = int(tokens[0])
        if len(tokens) < 2:
            continue

        # Handle 88-level condition names: attach to last data node
        if level == 88:
            if last_data_node is not None:
                cond_name = tokens[1].rstrip('.')
                attrs = tokens[2:]
                last_data_node.setdefault("conditions", []).append({
                    "name": cond_name,
                    "values": _parse_88_values(attrs)
                })
            continue

        # Skip 66-level RENAMES (not modelled here)
        if level == 66:
            continue

        name = tokens[1].rstrip('.')
        attrs = tokens[2:]

        # Drop trailing period token content
        if attrs and attrs[-1].endswith('.'):
            attrs[-1] = attrs[-1][:-1]

        upper_attrs = [a.upper() for a in attrs]
        is_group = ("PIC" not in upper_attrs and "PICTURE" not in upper_attrs)

        # Parse occurs/redefines/depending-on
        occurs = None
        occurs_min = None
        occurs_max = None
        redefines = None
        depending_on = None

        i = 0
        while i < len(attrs):
            attr_u = attrs[i].upper()
            if attr_u == "OCCURS":
                # OCCURS n TIMES | OCCURS n TO m TIMES [DEPENDING ON var]
                j = i + 1
                first_num = None
                second_num = None
                if j < len(attrs) and re.search(r'\d', attrs[j]):
                    first_num = int(re.sub(r'\D', '', attrs[j]))
                    j += 1
                    if j < len(attrs) and attrs[j].upper() == "TO" and (j + 1) < len(attrs) and re.search(r'\d', attrs[j+1]):
                        second_num = int(re.sub(r'\D', '', attrs[j+1]))
                        j += 2
                    # Skip optional TIMES
                    if j < len(attrs) and attrs[j].upper().startswith("TIME"):
                        j += 1
                if first_num is not None:
                    occurs = first_num if second_num is None else second_num
                    occurs_min = first_num
                    occurs_max = second_num if second_num is not None else first_num
                i = j
                continue
            if attr_u == "REDEFINES" and i + 1 < len(attrs):
                redefines = attrs[i + 1].rstrip('.')
                i += 2
                continue
            if attr_u == "DEPENDING" and i + 2 < len(attrs) and attrs[i+1].upper() == "ON":
                depending_on = attrs[i+2].rstrip('.')
                i += 3
                continue
            i += 1

        node = add_to_dict(level, name, attrs, is_group, occurs, occurs_min, occurs_max, redefines, depending_on)
        last_data_node = node  # For potential following 88-level conditions

    return output


def parse_pic_and_usage(attrs_str: str) -> Dict[str, Any]:
    """
    Parse PIC and USAGE clauses to get detailed field information.
    Returns: storage_type, logical_type, length, decimal_places, bytes, pic
    storage_type: ch (DISPLAY), pd (COMP-3), bi (COMP/COMP-4/COMP-5), f4 (COMP-1), f8 (COMP-2), zd/zd+ (zoned)
    """
    result = {
        "storage_type": "ch",
        "logical_type": "string",
        "length": 0,
        "decimal_places": 0,
        "bytes": 0,
        "pic": ""
    }

    if not attrs_str:
        return result

    attrs = attrs_str.split()
    upper = attrs_str.upper()

    # Extract PIC literal (next token after PIC/PICTURE)
    pic_str = ""
    for i, tok in enumerate(attrs):
        if tok.upper() in ("PIC", "PICTURE") and i + 1 < len(attrs):
            pic_str = attrs[i + 1].rstrip('.')
            break
    pic_str_u = pic_str.upper()
    result["pic"] = pic_str_u

    # Determine storage type
    if "COMP-1" in upper:
        result["storage_type"] = "f4"
        result["logical_type"] = "number"
        result["bytes"] = 4
        return result
    if "COMP-2" in upper:
        result["storage_type"] = "f8"
        result["logical_type"] = "number"
        result["bytes"] = 8
        return result
    if "COMP-3" in upper or "PACKED-DECIMAL" in upper:
        result["storage_type"] = "pd"
    elif "COMP-5" in upper or "COMP-4" in upper or re.search(r'\bCOMP\b', upper):
        result["storage_type"] = "bi"
    elif pic_str_u.startswith("S"):
        result["storage_type"] = "zd+"
    elif pic_str_u.startswith("9"):
        result["storage_type"] = "zd"
    else:
        result["storage_type"] = "ch"

    # If no PIC, keep defaults for group
    if not pic_str_u:
        return result

    # Remove sign S prefix for counting
    pic_body = pic_str_u.lstrip('S')

    # Handle implied decimal V
    if 'V' in pic_body:
        int_part, dec_part = pic_body.split('V', 1)
    else:
        int_part, dec_part = pic_body, ""

    # Helper to count symbols with optional parentheses
    def _count(symbols: str, allowed: str) -> int:
        if not symbols:
            return 0
        # Expand repetitions like 9(5) -> 99999
        total = 0
        i = 0
        while i < len(symbols):
            ch = symbols[i]
            if ch == '(':
                # Shouldn't start with '(' here; skip
                i += 1
                continue
            if ch in allowed:
                # Check for (n)
                if i + 1 < len(symbols) and symbols[i+1] == '(':
                    j = i + 2
                    num = ""
                    while j < len(symbols) and symbols[j].isdigit():
                        num += symbols[j]
                        j += 1
                    # Skip ')'
                    if j < len(symbols) and symbols[j] == ')':
                        j += 1
                    count = int(num) if num else 1
                    total += count
                    i = j
                else:
                    total += 1
                    i += 1
            else:
                i += 1
        return total

    # Count integer/decimal digits or characters
    if any(c in int_part for c in ("9", "Z")):
        int_digits = _count(int_part, "9Z")
    else:
        int_digits = _count(int_part, "XA")

    dec_digits = _count(dec_part, "9")

    result["length"] = int_digits
    result["decimal_places"] = dec_digits

    total_digits = int_digits + dec_digits

    # Size in bytes
    if result["storage_type"] == "pd":
        # Packed decimal: ceil(digits/2) + 1 sign nibble
        result["bytes"] = (total_digits + 1) // 2 + 1
        result["logical_type"] = "number" if dec_digits > 0 else "integer"
    elif result["storage_type"] == "bi":
        # Approximate based on digits
        if total_digits <= 4:
            result["bytes"] = 2
        elif total_digits <= 9:
            result["bytes"] = 4
        else:
            result["bytes"] = 8
        result["logical_type"] = "number" if dec_digits > 0 else "integer"
    elif result["storage_type"] in ("zd", "zd+"):
        # Zoned decimal stored as one byte per digit
        result["bytes"] = total_digits
        result["logical_type"] = "number" if dec_digits > 0 else "integer"
    else:
        # DISPLAY alphanumeric/alpha
        if any(c in pic_body for c in ("X", "A")):
            result["bytes"] = int_digits
            result["logical_type"] = "string"
        else:
            # Fallback to digits
            result["bytes"] = total_digits
            result["logical_type"] = "number" if dec_digits > 0 else "integer"

    return result


def _build_json_schema_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively builds a JSON schema for a single node."""
    schema_node: Dict[str, Any] = {}
    if node["is_group"]:
        schema_node["type"] = "object"
        schema_node["title"] = node["name"]
        properties: Dict[str, Any] = {}
        for child_name, child_node in node.get("children", {}).items():
            # Skip redefined nodes in schema
            if child_node.get('redefines'):
                continue
            properties[child_name] = _build_json_schema_node(child_node)
        if properties:
            schema_node["properties"] = properties
    else:
        pic_info = node
        logical_type = pic_info.get("logical_type", "string")
        if logical_type == "string":
            schema_node["type"] = "string"
            if "length" in pic_info:
                schema_node["maxLength"] = pic_info["length"]
        elif logical_type == "integer":
            schema_node["type"] = "integer"
        elif logical_type == "number":
            schema_node["type"] = "number"
        else:
            schema_node["type"] = "string"
        if "pic" in pic_info and pic_info["pic"]:
            schema_node["description"] = f"COBOL PIC: {pic_info['pic']}"

    # OCCURS handling: prefer min/max if present
    if node.get("occurs_min") is not None or node.get("occurs_max") is not None:
        return {
            "type": "array",
            "items": schema_node,
            "minItems": int(node.get("occurs_min", 0)),
            "maxItems": int(node.get("occurs_max", node.get("occurs", 0)) or 0)
        }
    if node.get("occurs"):
        count = int(node["occurs"])
        return {"type": "array", "items": schema_node, "minItems": count, "maxItems": count}
    return schema_node


def generate_layout_details(parsed_copybook: Dict[str, Any]) -> Dict[str, Any]:
    """
    Walks the parsed copybook tree to decorate it with physical layout details (offset, size)
    and generates a corresponding JSON Schema for each 01-level record.
    """
    if not parsed_copybook:
        return {"error": "Empty or invalid copybook provided. No 01-level record found."}

    all_layouts = []

    for root_name, root_node in parsed_copybook.items():
        # Track offsets for REDEFINES groups
        redefines_offsets: Dict[str, int] = {}
        single_record_tree = {root_name: root_node}

        def _get_item_size(node: Dict[str, Any]) -> int:
            # Size of a single occurrence of this node
            if not node.get("is_group"):
                return parse_pic_and_usage(node.get("raw_attrs", "")).get("bytes", 0)

            child_offset, max_redefine_end = 0, 0
            temp_redefines_offsets: Dict[str, int] = {}
            for name, child in node.get("children", {}).items():
                redefines = child.get("redefines")
                occurs_count = int(child.get("occurs_max", child.get("occurs", 1)) or 1)
                start_pos = temp_redefines_offsets.get(redefines, child_offset) if redefines else max(child_offset, max_redefine_end)
                temp_redefines_offsets[name] = start_pos
                child_size = _get_item_size(child) * occurs_count
                if redefines:
                    max_redefine_end = max(max_redefine_end, start_pos + child_size)
                else:
                    child_offset = start_pos + child_size
            return max(child_offset, max_redefine_end)

        def walk_and_decorate(obj: Dict[str, Any], base_offset: int = 0) -> int:
            current_offset = base_offset
            max_redefine_end = base_offset

            for name, node in obj.items():
                redefines = node.get("redefines")
                occurs_count = int(node.get("occurs_max", node.get("occurs", 1)) or 1)

                field_offset = redefines_offsets.get(redefines, current_offset) if redefines else max(current_offset, max_redefine_end)
                redefines_offsets[name] = field_offset
                node["offset"] = field_offset

                # Decorate leaf with PIC/USAGE info
                if node.get("is_group"):
                    # Compute size of one occurrence before descending
                    single_item_size = _get_item_size(node)
                    node["byte_size"] = single_item_size * occurs_count
                    walk_and_decorate(node.get("children", {}), field_offset)
                else:
                    pic_info = parse_pic_and_usage(node.get("raw_attrs", ""))
                    node.update(pic_info)
                    node["byte_size"] = pic_info.get("bytes", 0) * occurs_count

                if redefines:
                    max_redefine_end = max(max_redefine_end, field_offset + node["byte_size"])
                else:
                    current_offset = field_offset + node["byte_size"]

            return max(current_offset, max_redefine_end)

        lrecl = walk_and_decorate(single_record_tree)

        json_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": root_name,
            **_build_json_schema_node(root_node)
        }

        record_layout = {
            "record_name": root_name,
            "record_byte_size": lrecl,
            "physical_layout": single_record_tree,
            "json_schema": json_schema
        }
        all_layouts.append(record_layout)

    return {"record_layouts": all_layouts}


def copybook_to_detailed_json(copybook_content: str, copybook_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    High-level function to convert a copybook string to a detailed JSON with logical and physical layout.
    """
    lines = copybook_content.splitlines()
    processed_lines = preprocess_copybook_lines(lines, copybook_dir)
    parsed_structure = parse_copybook_lines(processed_lines)
    detailed_json = generate_layout_details(parsed_structure)
    return detailed_json


def copybook_file_to_detailed_json(copybook_path: str, copybook_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    High-level function to convert a copybook file to a detailed JSON with logical and physical layout.
    """
    with open(copybook_path, "r", encoding="utf-8", errors="ignore") as finp:
        content = finp.read()
    if not copybook_dir:
        copybook_dir = os.path.dirname(copybook_path)
    return copybook_to_detailed_json(content, copybook_dir=copybook_dir)