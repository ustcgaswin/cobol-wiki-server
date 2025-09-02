import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlmodel import Field, Session, SQLModel, create_engine

from app.utils.logger import logger

from sqlalchemy.orm import registry

analysis_registry = registry()
AnalysisBase = analysis_registry.generate_base()

# --- Database Setup ---

ANALYSIS_BASE_PATH = Path("project_analysis")


def get_db_path(project_id: UUID) -> Path:
    """Returns the path to the project's analysis SQLite database."""
    db_dir = ANALYSIS_BASE_PATH / str(project_id)
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "analysis.db"


def get_db_engine(project_id: UUID):
    """Creates a new SQLAlchemy engine for the project's database."""
    db_path = get_db_path(project_id)
    return create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})


# --- SQLModel Table Definitions ---

class SourceFile(AnalysisBase,SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    relative_path: str
    file_name: str
    file_type: str
    content: str
    status: str = "parsed"


class FileRelationship(AnalysisBase,SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source_file_id: int = Field(foreign_key="sourcefile.id")
    target_file_name: str
    relationship_type: str
    statement: str
    line_number: int


class CobolProgram(AnalysisBase,SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    file_id: int = Field(foreign_key="sourcefile.id")
    program_id_name: str
    program_type: str


class CobolFileControlEntry(AnalysisBase,SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    program_id: int = Field(foreign_key="cobolprogram.id")
    logical_name: str
    assign_to: str
    organization: Optional[str] = None
    access_mode: Optional[str] = None
    record_key: Optional[str] = None
    file_status_variable: Optional[str] = None
    raw_statement: str


class CobolParagraph(AnalysisBase,SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    program_id: int = Field(foreign_key="cobolprogram.id")
    name: str
    content: str
    start_line: int


class CobolStatement(AnalysisBase,SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    paragraph_id: Optional[int] = Field(default=None, foreign_key="cobolparagraph.id")
    program_id: int = Field(foreign_key="cobolprogram.id")
    statement_type: str
    target: Optional[str] = None
    content: str
    line_number: int


class JclJob(AnalysisBase,SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    file_id: int = Field(foreign_key="sourcefile.id")
    job_name: str
    raw_card: str


class JclStep(AnalysisBase,SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="jcljob.id")
    step_name: str
    exec_type: str
    exec_target: str
    condition: Optional[str] = None
    raw_card: str


class JclDdStatement(AnalysisBase,SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="jcljob.id")
    step_id: Optional[int] = Field(default=None, foreign_key="jclstep.id")
    dd_name: str
    dataset_name: Optional[str] = None
    disposition: Optional[str] = None
    is_in_stream: bool = False
    in_stream_data: Optional[str] = None # FIX: Changed field name to match parser output
    raw_card: str


class CopybookField(AnalysisBase,SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    file_id: int = Field(foreign_key="sourcefile.id")
    parent_id: Optional[int] = Field(default=None, foreign_key="copybookfield.id")
    level: int
    name: str
    pic: Optional[str] = None
    usage: Optional[str] = None
    byte_size: int
    offset: int
    raw_definition: str


class RexxScript(AnalysisBase,SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    file_id: int = Field(foreign_key="sourcefile.id")
    script_name: str
    script_type: str


class RexxSubroutine(AnalysisBase,SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    script_id: int = Field(foreign_key="rexxscript.id")
    name: str
    content: str
    start_line: int


class RexxStatement(AnalysisBase,SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    subroutine_id: Optional[int] = Field(default=None, foreign_key="rexxsubroutine.id")
    script_id: int = Field(foreign_key="rexxscript.id")
    statement_type: str
    target: Optional[str] = None
    content: str
    line_number: int


# --- Service Functions ---

def initialize_database(project_id: UUID):
    engine = get_db_engine(project_id)
    analysis_registry.metadata.create_all(engine)
    logger.info(f"Database tables created for project {project_id}")

    
def add_source_file(project_id: UUID, relative_path: str, file_type: str, content: str) -> Optional[SourceFile]:
    """Adds a source file record to the database and returns the model instance."""
    engine = get_db_engine(project_id)
    with Session(engine) as session:
        try:
            file_name = Path(relative_path).name
            source_file = SourceFile(
                relative_path=relative_path,
                file_name=file_name,
                file_type=file_type,
                content=content,
            )
            session.add(source_file)
            session.commit()
            session.refresh(source_file)
            return source_file
        except Exception as e:
            logger.error(f"Failed to add source file {relative_path} to DB for project {project_id}: {e}", exc_info=True)
            session.rollback()
            return None





def populate_jcl_data(project_id: UUID, source_file_id: int, parsed_data: Dict[str, Any]):
    """Populates JCL tables from parsed data."""
    engine = get_db_engine(project_id)
    with Session(engine) as session:
        for job_data in parsed_data.get("jobs", []):
            jcl_job = JclJob(
                file_id=source_file_id,
                job_name=job_data.get("name", "UNKNOWN_JOB"),
                raw_card=job_data.get("raw_card", "")
            )
            session.add(jcl_job)
            session.flush()

            # --- Process Job-Level DD statements (e.g., JOBLIB) ---
            for dd_data in job_data.get("job_level_dd_statements", []):
                dd_params = dd_data.get("parameters", {})
                is_in_stream = "in_stream_data" in dd_data
                jcl_dd = JclDdStatement(
                    job_id=jcl_job.id,
                    step_id=None,
                    dd_name=dd_data.get("name", "UNKNOWN_DD"),
                    dataset_name=dd_data.get("dataset_name"),
                    disposition=json.dumps(dd_params.get("DISP")),
                    is_in_stream=is_in_stream,
                    in_stream_data="\n".join(dd_data.get("in_stream_data", [])),
                    raw_card=dd_data.get("raw_card", "")
                )
                session.add(jcl_dd)

            def process_items(items: List[Dict[str, Any]], current_job_id: int):
                for item in items:
                    if item.get("type") == "EXEC":
                        params = item.get("parameters", {})
                        exec_type = "PGM" if "PGM" in params else "PROC"
                        exec_target = params.get("PGM") or params.get("PROC", "UNKNOWN")
                        
                        # FIX: Serialize the COND parameter if it's a list
                        cond_param = params.get("COND")
                        condition_str = json.dumps(cond_param) if cond_param is not None else None

                        jcl_step = JclStep(
                            job_id=current_job_id,
                            step_name=item.get("name", "UNKNOWN_STEP"),
                            exec_type=exec_type,
                            exec_target=str(exec_target),
                            condition=condition_str,
                            raw_card=item.get("raw_card", "")
                        )
                        session.add(jcl_step)
                        session.flush()
                        for dd_data in item.get("dd_statements", []):
                            dd_params = dd_data.get("parameters", {})
                            is_in_stream = "in_stream_data" in dd_data
                            jcl_dd = JclDdStatement(
                                job_id=current_job_id,
                                step_id=jcl_step.id,
                                dd_name=dd_data.get("name", "UNKNOWN_DD"),
                                dataset_name=dd_data.get("dataset_name"),
                                disposition=json.dumps(dd_params.get("DISP")),
                                is_in_stream=is_in_stream,
                                in_stream_data="\n".join(dd_data.get("in_stream_data", [])),
                                raw_card=dd_data.get("raw_card", "")
                            )
                            session.add(jcl_dd)
                    elif item.get("type") == "IF":
                        process_items(item.get("then_items", []), current_job_id)
                        process_items(item.get("else_items", []), current_job_id)
            process_items(job_data.get("items", []), jcl_job.id)
        session.commit()
    logger.info(f"Populated JCL data for file ID {source_file_id}")




def populate_cobol_data(project_id: UUID, source_file_id: int, parsed_data: Dict[str, Any]):
    """Populates COBOL tables from parsed data."""
    engine = get_db_engine(project_id)
    with Session(engine) as session:
        meta = parsed_data.get("program_metadata", {})
        program = CobolProgram(
            file_id=source_file_id,
            program_id_name=meta.get("program_id", "UNKNOWN_PROGRAM"),
            program_type=meta.get("program_type", "UNKNOWN")
        )
        session.add(program)
        session.flush()

        for fc_data in parsed_data.get("files_and_databases", {}).get("files", []):
            fc_entry = CobolFileControlEntry(
                program_id=program.id,
                logical_name=fc_data.get("logical_name"),
                assign_to=fc_data.get("physical_name"),
                organization=fc_data.get("organization"),
                access_mode=fc_data.get("access_mode"),
                record_key=fc_data.get("record_key"),
                file_status_variable=fc_data.get("file_status"),
                raw_statement=f"SELECT {fc_data.get('logical_name')} ASSIGN TO {fc_data.get('physical_name')}"
            )
            session.add(fc_entry)

        para_map = {}
        sorted_paras = sorted(parsed_data.get("control_flow", {}).get("paragraphs", []), key=lambda p: p['line_number'])
        for para_data in sorted_paras:
            para = CobolParagraph(
                program_id=program.id,
                name=para_data["name"],
                content=para_data.get("content", ""),
                start_line=para_data["line_number"]
            )
            session.add(para)
            session.flush()
            para_map[para.name] = para

        def get_para_id_for_line(line_num: int) -> Optional[int]:
            for para in reversed(sorted_paras):
                if line_num >= para["line_number"]:
                    return para_map[para["name"]].id
            return None

        details = parsed_data.get("control_flow", {})
        for call in details.get("calls", []):
            session.add(CobolStatement(program_id=program.id, paragraph_id=get_para_id_for_line(call['line_number']), statement_type="CALL", target=call['program_name'], content=f"CALL '{call['program_name']}'", line_number=call['line_number']))
        for perform in details.get("performs", []):
            session.add(CobolStatement(program_id=program.id, paragraph_id=get_para_id_for_line(perform['line_number']), statement_type="PERFORM", target=perform['target'], content=f"PERFORM {perform['target']}", line_number=perform['line_number']))
        for io in details.get("io_operations", []):
            session.add(CobolStatement(program_id=program.id, paragraph_id=get_para_id_for_line(io['line_number']), statement_type=io['verb'], target=io['target'], content=f"{io['verb']} {io['target']}", line_number=io['line_number']))
        
        rels = parsed_data.get("relationships", {})
        for call in rels.get("called_programs", []):
            session.add(FileRelationship(source_file_id=source_file_id, target_file_name=call, relationship_type="CALL", statement=f"CALL '{call}'", line_number=0))
        for copybook in rels.get("copybooks", []):
            session.add(FileRelationship(source_file_id=source_file_id, target_file_name=copybook, relationship_type="COPY", statement=f"COPY {copybook}", line_number=0))

        session.commit()
    logger.info(f"Populated COBOL data for file ID {source_file_id}")


def populate_copybook_data(project_id: UUID, source_file_id: int, parsed_data: Dict[str, Any]):
    """Populates Copybook tables from parsed data."""
    engine = get_db_engine(project_id)
    with Session(engine) as session:
        for layout in parsed_data.get("record_layouts", []):
            root_node_data = next(iter(layout.get("physical_layout", {}).values()), None)
            if not root_node_data:
                continue

            def process_node(node_data: Dict[str, Any], parent_id: Optional[int]):
                field = CopybookField(
                    file_id=source_file_id,
                    parent_id=parent_id,
                    level=int(node_data.get("level", 0)),
                    name=node_data.get("name", "FILLER"),
                    pic=node_data.get("pic"),
                    usage=node_data.get("storage_type"),
                    byte_size=node_data.get("byte_size", 0),
                    offset=node_data.get("offset", 0),
                    raw_definition=node_data.get("raw_attrs", "")
                )
                session.add(field)
                session.flush()
                for child_data in node_data.get("children", {}).values():
                    process_node(child_data, field.id)
            process_node(root_node_data, None)
        session.commit()
    logger.info(f"Populated Copybook data for file ID {source_file_id}")


def populate_rexx_data(project_id: UUID, source_file_id: int, parsed_data: Dict[str, Any]):
    """Populates REXX tables from parsed data."""
    engine = get_db_engine(project_id)
    with Session(engine) as session:
        meta = parsed_data.get("script_metadata", {})
        script = RexxScript(
            file_id=source_file_id,
            script_name=meta.get("script_name", "UNTITLED"),
            script_type=meta.get("script_type", "UNKNOWN")
        )
        session.add(script)
        session.flush()

        sub_map = {}
        sorted_subs = sorted(parsed_data.get("control_flow", {}).get("subroutines", []), key=lambda s: s['line_number'])
        for sub_data in sorted_subs:
            sub = RexxSubroutine(
                script_id=script.id,
                name=sub_data["name"],
                content="",
                start_line=sub_data["line_number"]
            )
            session.add(sub)
            session.flush()
            sub_map[sub.name] = sub

        def get_sub_id_for_line(line_num: int) -> Optional[int]:
            for sub in reversed(sorted_subs):
                if line_num >= sub["line_number"]:
                    return sub_map[sub["name"]].id
            return None

        flow = parsed_data.get("control_flow", {})
        for call in flow.get("calls", []):
            session.add(RexxStatement(script_id=script.id, subroutine_id=get_sub_id_for_line(call['line_number']), statement_type="CALL", target=call["target"], content=f"CALL {call['target']}", line_number=call["line_number"]))
        for cmd in parsed_data.get("external_commands", []):
            session.add(RexxStatement(script_id=script.id, subroutine_id=get_sub_id_for_line(cmd['line_number']), statement_type="EXTERNAL_COMMAND", target=cmd["environment"], content=cmd["command"], line_number=cmd["line_number"]))

        session.commit()
    logger.info(f"Populated REXX data for file ID {source_file_id}")