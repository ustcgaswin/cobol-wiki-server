import json
import threading
from pathlib import Path
from typing import Dict, List
from uuid import UUID
from datetime import datetime, timezone

from sqlmodel import Session

from app.config.db_config import engine
from app.models.project_model import Project, WikiStatus
from app.services import (
    db_population_service,
    rag_service,
    wiki_generation_service,
    wiki_tree_service,
)
from app.utils import cobol_utils, copybook_utils, jcl_utils, rexx_utils
from app.utils.logger import logger

PROJECT_STORAGE_PATH = Path("project_storage")
ANALYSIS_BASE_PATH = Path("project_analysis")

_RUNNING_PIPELINES: set[str] = set()
_RUNNING_LOCK = threading.Lock()


def _get_project_source_path(project_id: UUID) -> Path:
    return PROJECT_STORAGE_PATH / str(project_id)


def _get_project_analysis_dir(project_id: UUID) -> Path:
    return ANALYSIS_BASE_PATH / str(project_id) / "analysis"


def get_analysis_file_path(project_id: UUID) -> Path:
    return _get_project_analysis_dir(project_id) / "analysis.json"


def _project_exists(project_id: UUID) -> bool:
    with Session(engine) as db:
        return db.get(Project, project_id) is not None


def start_project_pipeline_background(project_id: UUID):
    pid = str(project_id)
    with _RUNNING_LOCK:
        if pid in _RUNNING_PIPELINES:
            logger.info(f"[PIPELINE] Already running for project {project_id}, skipping new launch.")
            return
        _RUNNING_PIPELINES.add(pid)

    def _runner():
        try:
            start_project_pipeline(project_id)
        finally:
            with _RUNNING_LOCK:
                _RUNNING_PIPELINES.discard(pid)

    t = threading.Thread(
        target=_runner,
        name=f"pipeline-{project_id}",
        daemon=True,
    )
    t.start()
    logger.info(f"[PIPELINE] Launched background pipeline thread {t.name} for project {project_id}")


def start_project_pipeline(project_id: UUID):
    """
    Pipeline: (optional) build RAG embeddings first (skipped if up-to-date),
    then run analysis (which logs/persists wiki tree), and finally generate the wiki.
    Aborts early if the project was deleted.
    """
    # Early existence check (project might have been deleted before thread started)
    if not _project_exists(project_id):
        logger.warning(f"[PIPELINE] Project {project_id} no longer exists. Aborting pipeline.")
        return

    embedding_failed = False
    try:
        logger.info(f"Starting project pipeline (RAG -> Analysis) for {project_id}")
        # Skip embedding rebuild if index is already current
        try:
            if hasattr(rag_service, "embeddings_up_to_date") and rag_service.embeddings_up_to_date(project_id):
                logger.info(f"[PIPELINE] Embeddings already up to date for {project_id}; skipping rebuild.")
            else:
                rag_service.build_embeddings_for_project(project_id)
        except Exception as e:
            embedding_failed = True
            logger.error(
                f"[PIPELINE] Embedding step failed for project {project_id}: {e}. Continuing with analysis.",
                exc_info=True,
            )
    except Exception as e:
        embedding_failed = True
        logger.error(
            f"[PIPELINE] Unexpected error during embedding phase for {project_id}: {e}. Continuing with analysis.",
            exc_info=True,
        )

    # Re-check existence before analysis (project could be deleted during embeddings)
    if not _project_exists(project_id):
        logger.warning(f"[PIPELINE] Project {project_id} deleted after embeddings phase. Aborting analysis.")
        return

    start_analysis_for_project(project_id)

    if embedding_failed:
        logger.warning(f"[PIPELINE] Project {project_id}: analysis completed but embeddings had failed earlier.")


def start_analysis_for_project(project_id: UUID):
    logger.info(f"Starting analysis for project_id: {project_id}")

    with Session(engine) as db:
        try:
            project = db.get(Project, project_id)
            if not project:
                logger.error(f"Project {project_id} not found in database for analysis (aborting).")
                return

            if project.wiki_status not in (WikiStatus.ANALYZING, WikiStatus.GENERATED):
                project.wiki_status = WikiStatus.ANALYZING

            project.analysis_start_time = datetime.now(timezone.utc)
            db.add(project)
            db.commit()
            db.refresh(project)

            source_path = _get_project_source_path(project.id)
            if not source_path.exists() or not source_path.is_dir():
                logger.error(f"Source directory not found for project '{project.name}' at {source_path}")
                project.wiki_status = WikiStatus.FAILED
                db.add(project)
                db.commit()
                return

            # Initialize the analysis database for this project
            try:
                db_population_service.initialize_database(project.id)
                logger.info(f"Initialized analysis database for project {project.id}")
            except Exception as e:
                logger.error(f"Failed to initialize analysis database for project {project.id}: {e}", exc_info=True)
                project.wiki_status = WikiStatus.FAILED
                db.add(project)
                db.commit()
                return

            jcl_lib_dir = source_path
            copybook_dir = source_path
            logger.info(f"Scanning for files in: {source_path}")

            jcl_results: List[Dict[str, object]] = []
            cobol_results: List[Dict[str, object]] = []
            copybook_results: List[Dict[str, object]] = []
            rexx_results: List[Dict[str, object]] = []

            # Gather file lists
            jcl_files = list(source_path.rglob("*.jcl"))
            cobol_files = list(source_path.rglob("*.cbl"))
            cpy_files = list(source_path.rglob("*.cpy"))
            rexx_files = list(source_path.rglob("*.rex*"))

            # Update counts if project still exists
            project = db.get(Project, project_id)
            if not project:
                logger.warning(f"[ANALYSIS] Project {project_id} deleted before counting files. Aborting.")
                return

            project.jcl_file_count = len(jcl_files)
            project.cobol_file_count = len(cobol_files)
            project.copybook_file_count = len(cpy_files)
            project.rexx_file_count = len(rexx_files)
            db.add(project)
            db.commit()

            # Helper to abort mid-way if project deleted externally
            def _ensure_present() -> bool:
                p = db.get(Project, project_id)
                if p is None:
                    logger.warning(f"[ANALYSIS] Project {project_id} deleted mid-analysis. Aborting.")
                    return False
                return True

            # Process JCL files
            for jcl_file in jcl_files:
                if not _ensure_present():
                    return
                logger.info(f"Parsing JCL file: {jcl_file.name}")
                try:
                    with open(jcl_file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    parsed_jcl = jcl_utils.parse_jcl_to_json(content, str(jcl_lib_dir))
                    jcl_results.append({"file": str(jcl_file.relative_to(source_path)), **parsed_jcl})
                    source_file_record = db_population_service.add_source_file(
                        project_id, str(jcl_file.relative_to(source_path)), "JCL", content
                    )
                    if source_file_record:
                        db_population_service.populate_jcl_data(project_id, source_file_record.id, parsed_jcl)
                except Exception as e:
                    logger.error(f"Failed to parse JCL file {jcl_file.name}: {e}", exc_info=True)

            # Process COBOL files
            for cobol_file in cobol_files:
                if not _ensure_present():
                    return
                logger.info(f"Parsing COBOL file: {cobol_file.name}")
                try:
                    with open(cobol_file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    parsed_cobol = cobol_utils.parse_cobol_program(content)
                    cobol_results.append({"file": str(cobol_file.relative_to(source_path)), "data": parsed_cobol})
                    source_file_record = db_population_service.add_source_file(
                        project_id, str(cobol_file.relative_to(source_path)), "COBOL", content
                    )
                    if source_file_record:
                        db_population_service.populate_cobol_data(project_id, source_file_record.id, parsed_cobol)
                except Exception as e:
                    logger.error(f"Failed to parse COBOL file {cobol_file.name}: {e}", exc_info=True)

            # Process Copybook files
            for cpy_file in cpy_files:
                if not _ensure_present():
                    return
                logger.info(f"Parsing Copybook file: {cpy_file.name}")
                try:
                    parsed_copybook = copybook_utils.copybook_file_to_detailed_json(
                        str(cpy_file), str(copybook_dir)
                    )
                    copybook_results.append({"file": str(cpy_file.relative_to(source_path)), "data": parsed_copybook})
                    with open(cpy_file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    source_file_record = db_population_service.add_source_file(
                        project_id, str(cpy_file.relative_to(source_path)), "COPYBOOK", content
                    )
                    if source_file_record:
                        db_population_service.populate_copybook_data(
                            project_id, source_file_record.id, parsed_copybook
                        )
                except Exception as e:
                    logger.error(f"Failed to parse Copybook file {cpy_file.name}: {e}", exc_info=True)

            # Process REXX files
            for rexx_file in rexx_files:
                if not _ensure_present():
                    return
                logger.info(f"Parsing REXX file: {rexx_file.name}")
                try:
                    with open(rexx_file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    parsed_rexx = rexx_utils.parse_rexx_script(content, script_name=rexx_file.name)
                    rexx_results.append({"file": str(rexx_file.relative_to(source_path)), "data": parsed_rexx})
                    source_file_record = db_population_service.add_source_file(
                        project_id, str(rexx_file.relative_to(source_path)), "REXX", content
                    )
                    if source_file_record:
                        db_population_service.populate_rexx_data(
                            project_id, source_file_record.id, parsed_rexx
                        )
                except Exception as e:
                    logger.error(f"Failed to parse REXX file {rexx_file.name}: {e}", exc_info=True)

            # Re-check presence before persisting aggregated results
            if not _ensure_present():
                return

            analysis_payload = {
                "project_id": str(project_id),
                "project_name": project.name,
                "jcl": jcl_results,
                "cobol": cobol_results,
                "copybooks": copybook_results,
                "rexx": rexx_results,
            }
            out_dir = _get_project_analysis_dir(project_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "analysis.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(analysis_payload, f, indent=2)
            logger.info(f"Wrote consolidated analysis to {out_file}")

            # Build and persist the wiki tree
            try:
                tree = wiki_tree_service.build_pages_tree(analysis_payload)
                logger.info(f"Wiki tree built for project {project_id}")
                wiki_out = wiki_tree_service.get_wiki_structure_file_path(project_id)
                with open(wiki_out, "w", encoding="utf-8") as wf:
                    json.dump(tree, wf, indent=2)
            except Exception as e:
                logger.error(f"Failed to build wiki structure for project {project_id}: {e}", exc_info=True)
                project.wiki_status = WikiStatus.FAILED
                db.add(project)
                db.commit()
                return

            # Generate wiki pages
            try:
                manifest = wiki_generation_service.generate_persist_and_log_wiki(project_id, tree)
                logger.info(f"Wiki generation manifest stored for project {project_id}")
                project.wiki_status = WikiStatus.GENERATED
            except Exception as e:
                logger.error(f"Failed to generate wiki pages for project {project_id}: {e}", exc_info=True)
                project.wiki_status = WikiStatus.FAILED

            logger.info(f"Analysis complete for project {project.name} ({project_id})")

        except Exception as e:
            logger.error(f"Critical analysis error for project {project_id}: {e}", exc_info=True)
            project = db.get(Project, project_id)
            if project:
                project.wiki_status = WikiStatus.FAILED
        finally:
            if "project" in locals() and project:
                project.analysis_end_time = datetime.now(timezone.utc)
                db.add(project)
                db.commit()