import os
import re
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy
from app.config import llm_config
from app.utils.logger import logger

# --- Constants ---
LLM_SUMMARY_BATCH_SIZE = 5
SCRIPT_SUMMARY_CHUNK_SIZE = 1500
SCRIPT_SUMMARY_CHUNK_OVERLAP = 150

# Feature flag to enable/disable LLM work (set REXX_SUMMARIES_ENABLED=0 to disable)
REXX_SUMMARIES_ENABLED = str(os.getenv("REXX_SUMMARIES_ENABLED", "1")).strip().lower() not in ("0", "false", "no")

# --- DSPy Signatures for LLM Output ---

class RexxSubroutineSummary(dspy.Signature):
    """Defines the structured summary for a single REXX subroutine."""
    subroutine_name: str = dspy.OutputField(desc="The name of the REXX subroutine or label.")
    summary: str = dspy.OutputField(desc="A concise, one-to-two sentence summary of the subroutine's primary function and logic.")
    inputs: List[str] = dspy.OutputField(desc="Variables or parameters used as input by this subroutine.")
    outputs: List[str] = dspy.OutputField(desc="Variables that are modified or returned by this subroutine.")

class RexxSubroutineSummaryList(dspy.Signature):
    """A container for a list of subroutine summaries, used for batch processing."""
    summaries: List[RexxSubroutineSummary] = dspy.OutputField(desc="A list of REXX subroutine summaries.")

# New robust JSON-based batch signature
class BatchRexxSubroutineSummaries(dspy.Signature):
    """Summarize a batch of REXX subroutines and return JSON text."""
    subroutines_code = dspy.InputField(desc="Delimited batch of REXX subroutines to summarize.")
    summaries_json = dspy.OutputField(desc="JSON object: { 'summaries': [ {subroutine_name, summary, inputs, outputs}, ... ] }")

class RexxScriptSummary(dspy.Signature):
    """Defines the structured summary for an entire REXX script."""
    script_name: str = dspy.OutputField(desc="The name of the REXX script (if known).")
    overall_summary: str = dspy.OutputField(desc="A comprehensive summary of the entire script's functionality, derived from analyzing its code in chunks.")

# --- DSPy Predictors ---

class SummarizeRexxSubroutines(dspy.Module):
    """A DSPy module to summarize a batch of REXX subroutines."""
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(BatchRexxSubroutineSummaries)

    def forward(self, subroutines_code: str):
        prompt = (
            "Analyze the following batch of REXX subroutines. For each one, provide a concise summary, "
            "list its key input variables, and the key outputs it produces or modifies.\n"
            "Respond ONLY with a single JSON object containing a 'summaries' key with a list of JSON objects.\n\n"
            "--- BATCH OF SUBROUTINES ---\n"
            f"{subroutines_code}"
        )
        return self.predictor(subroutines_code=prompt)

class SummarizeRexxChunk(dspy.Signature):
    """Summarizes a chunk of a REXX script, refining an existing summary if provided."""
    script_name = dspy.InputField(desc="The name of the REXX script.")
    code_chunk = dspy.InputField(desc="A chunk of the REXX script's source code.")
    existing_summary = dspy.InputField(desc="The summary generated from previous chunks. Empty for the first chunk.")
    
    refined_summary = dspy.OutputField(
        desc="A new, refined summary that integrates the information from the new code chunk."
    )

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

def _split_code_into_chunks(code: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """A simple text splitter to chunk the REXX code."""
    if not code:
        return []
    
    chunks = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)  # guard against non-positive step
    while start < len(code):
        end = start + chunk_size
        chunks.append(code[start:end])
        start += step
    return chunks

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

def _extract_subroutine_code_blocks(lines: List[str], subroutines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extracts the full source code for each subroutine/label."""
    code_blocks = []
    if not subroutines:
        return code_blocks
    
    sub_line_map = {s['line_number']: s['name'] for s in subroutines}
    
    current_sub_name = None
    current_sub_code: List[str] = []

    for i, line in enumerate(lines):
        line_num_in_file = i + 1
        
        if line_num_in_file in sub_line_map:
            if current_sub_name:
                code_blocks.append({"name": current_sub_name, "code": "\n".join(current_sub_code)})
            
            current_sub_name = sub_line_map[line_num_in_file]
            current_sub_code = [line]
        elif current_sub_name:
            current_sub_code.append(line)

    if current_sub_name:
        code_blocks.append({"name": current_sub_name, "code": "\n".join(current_sub_code)})
        
    return code_blocks

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

def _parse_llm_summaries_json(raw: str) -> List[Dict[str, Any]]:
    """
    Attempts to parse LLM output into a list of summary dicts.
    Accepts:
      - {"summaries": [...]} 
      - [...] (list at the top-level)
      - Text with an embedded JSON object containing "summaries".
    """
    if not raw or not isinstance(raw, str):
        return []

    def _try_parse(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    obj = _try_parse(raw)
    if obj is None:
        # Try to extract the largest {...} block
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            obj = _try_parse(m.group(0))

    if isinstance(obj, dict) and "summaries" in obj and isinstance(obj["summaries"], list):
        return obj["summaries"]
    if isinstance(obj, list):
        return obj
    return []

def _get_llm_summary_for_batch(batch: List[Dict[str, Any]]) -> None:
    """Sends a batch of subroutine code to the LLM for summarization."""
    try:
        with dspy.settings.context(lm=llm_config.llm):
            summarizer = SummarizeRexxSubroutines()
            batch_code_str = "\n".join([f"---\nSUBROUTINE: {p['name']}\n---\n{p['code']}" for p in batch])
            result = summarizer(subroutines_code=batch_code_str)

            summaries = _parse_llm_summaries_json(getattr(result, "summaries_json", ""))
            if not summaries:
                logger.warning("DSPy REXX summarization returned no parsable summaries.")
            for summary in summaries:
                logger.info(f"LLM Summary for {summary.get('subroutine_name')}: {summary.get('summary')}")
    except Exception as e:
        logger.error(f"DSPy REXX summarization failed for batch: {e}", exc_info=True)

def _get_progressive_summary(script_name: str, rexx_code: str) -> None:
    """Generates a progressive summary of the entire REXX script."""
    try:
        logger.info(f"Starting progressive summarization for script: {script_name}")
        chunks = _split_code_into_chunks(rexx_code, SCRIPT_SUMMARY_CHUNK_SIZE, SCRIPT_SUMMARY_CHUNK_OVERLAP)
        if not chunks:
            return

        summary = ""
        with dspy.settings.context(lm=llm_config.llm):
            summarize_chunk = dspy.Predict(SummarizeRexxChunk)
            for i, chunk in enumerate(chunks):
                logger.info(f"Summarizing chunk {i+1}/{len(chunks)} for script {script_name}...")
                existing = summary if i > 0 else "This is the first part of a REXX script."
                response = summarize_chunk(script_name=script_name, code_chunk=chunk, existing_summary=existing)
                summary = response.refined_summary.strip() if getattr(response, "refined_summary", None) else summary
        logger.info(f"Final Progressive Summary for {script_name}: {summary}")
    except Exception as e:
        logger.error(f"Progressive REXX summarization failed for {script_name}: {e}", exc_info=True)

def trigger_background_summaries(script_name: str, rexx_code: str, lines: List[str], subroutines: List[Dict[str, Any]]) -> None:
    """Uses a thread pool to run all summarization tasks in the background."""
    if not REXX_SUMMARIES_ENABLED:
        logger.info("REXX LLM summaries are disabled via REXX_SUMMARIES_ENABLED.")
        return
    if not getattr(llm_config, "llm", None):
        logger.warning("REXX LLM summaries skipped: llm_config.llm is not configured.")
        return

    with ThreadPoolExecutor(max_workers=5, thread_name_prefix="RexxSummary") as executor:
        futures = [executor.submit(_get_progressive_summary, script_name, rexx_code)]
        
        sub_code_blocks = _extract_subroutine_code_blocks(lines, subroutines)
        if sub_code_blocks:
            batches = [sub_code_blocks[i:i + LLM_SUMMARY_BATCH_SIZE] for i in range(0, len(sub_code_blocks), LLM_SUMMARY_BATCH_SIZE)]
            for batch in batches:
                futures.append(executor.submit(_get_llm_summary_for_batch, batch))
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"A REXX LLM summarization thread failed: {e}", exc_info=True)

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

    # Fire-and-forget background LLM summaries (if enabled and configured)
    trigger_background_summaries(metadata["script_name"], rexx_code, lines, subroutines)

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