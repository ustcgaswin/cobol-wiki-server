from pathlib import Path
from uuid import UUID
from typing import Any, Dict, List
import json

from sqlmodel import Session

from app.config.db_config import engine
from app.models.project_model import Project, WikiStatus
from app.utils.logger import logger
from app.utils import cobol_utils, jcl_utils, copybook_utils, rexx_utils
from app.services import rag_service, wiki_tree_service, wiki_generation_service  # RAG first, then analysis

# Source storage remains here
PROJECT_STORAGE_PATH = Path("project_storage")

# New analysis storage base
ANALYSIS_BASE_PATH = Path("project_analysis")


def _get_project_source_path(project_id: UUID) -> Path:
    """Constructs the path to the project's source code directory."""
    return PROJECT_STORAGE_PATH / str(project_id)

def _get_project_analysis_dir(project_id: UUID) -> Path:
    """Constructs the path to where analysis artifacts are written."""
    return ANALYSIS_BASE_PATH / str(project_id) / "analysis"

def get_analysis_file_path(project_id: UUID) -> Path:
    """Path to the consolidated analysis.json."""
    return _get_project_analysis_dir(project_id) / "analysis.json"


def start_project_pipeline(project_id: UUID):
    """
    Pipeline: build RAG embeddings first, then run analysis (which logs/persists wiki tree),
    and finally generate the wiki pages.
    """
    try:
        logger.info(f"Starting project pipeline (RAG -> Analysis) for {project_id}")
        rag_service.build_embeddings_for_project(project_id)
    except Exception as e:
        # Continue to analysis even if embeddings fail
        logger.error(f"RAG embedding step failed for project {project_id}: {e}", exc_info=True)
    # Then run the existing analysis
    start_analysis_for_project(project_id)


def start_analysis_for_project(project_id: UUID):
    """
    Main background task to perform analysis on a project's source files.
    This function creates its own database session to ensure it's managed
    correctly in a background thread.
    """
    logger.info(f"Starting analysis for project_id: {project_id}")
    
    with Session(engine) as db:
        try:
            project = db.get(Project, project_id)
            if not project:
                logger.error(f"Project {project_id} not found in database for analysis.")
                return

            source_path = _get_project_source_path(project.id)
            if not source_path.exists() or not source_path.is_dir():
                logger.error(f"Source directory not found for project '{project.name}' at {source_path}")
                project.wiki_status = WikiStatus.FAILED
                db.add(project)
                db.commit()
                return

            # Define specific directories for JCL includes and Copybooks
            jcl_lib_dir = source_path
            copybook_dir = source_path

            logger.info(f"Scanning for files in: {source_path}")

            # Collect results to consolidate into analysis.json
            jcl_results: List[Dict[str, Any]] = []
            cobol_results: List[Dict[str, Any]] = []
            copybook_results: List[Dict[str, Any]] = []
            rexx_results: List[Dict[str, Any]] = []

            # Process JCL files
            for jcl_file in source_path.rglob("*.jcl"):
                logger.info(f"Parsing JCL file: {jcl_file.name}")
                try:
                    with open(jcl_file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    parsed_jcl = jcl_utils.parse_jcl_to_json(content, str(jcl_lib_dir))
                    jcl_results.append({
                        "file": str(jcl_file.relative_to(source_path)),
                        "jobs": parsed_jcl.get("jobs", []),
                        "relationships": parsed_jcl.get("relationships", {})
                    })
                    logger.info(f"JCL Analysis for {jcl_file.name} complete.")
                except Exception as e:
                    logger.error(f"Failed to parse JCL file {jcl_file.name}: {e}", exc_info=True)

            # Process COBOL files
            for cobol_file in source_path.rglob("*.cbl"):
                logger.info(f"Parsing COBOL file: {cobol_file.name}")
                try:
                    with open(cobol_file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    parsed_cobol = cobol_utils.parse_cobol_program(content)
                    cobol_results.append({
                        "file": str(cobol_file.relative_to(source_path)),
                        "data": parsed_cobol
                    })
                    logger.info(f"COBOL Analysis for {cobol_file.name} complete.")
                    logger.info(f"Cobol analysis for {cobol_file.name} : {parsed_cobol} ")
                except Exception as e:
                    logger.error(f"Failed to parse COBOL file {cobol_file.name}: {e}", exc_info=True)

            # Process Copybook files
            for cpy_file in source_path.rglob("*.cpy"):
                logger.info(f"Parsing Copybook file: {cpy_file.name}")
                try:
                    parsed_copybook = copybook_utils.copybook_file_to_detailed_json(str(cpy_file), str(copybook_dir))
                    copybook_results.append({
                        "file": str(cpy_file.relative_to(source_path)),
                        "data": parsed_copybook
                    })
                    logger.info(f"Copybook Analysis for {cpy_file.name} complete.")
                except Exception as e:
                    logger.error(f"Failed to parse Copybook file {cpy_file.name}: {e}", exc_info=True)

            # Process REXX files
            for rexx_file in source_path.rglob("*.rex*"):
                logger.info(f"Parsing REXX file: {rexx_file.name}")
                try:
                    with open(rexx_file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    parsed_rexx = rexx_utils.parse_rexx_script(content, script_name=rexx_file.name)
                    rexx_results.append({
                        "file": str(rexx_file.relative_to(source_path)),
                        "data": parsed_rexx
                    })
                    logger.info(f"REXX Analysis for {rexx_file.name} complete.")
                    logger.info(f"REXX analysis for {rexx_file.name} : {parsed_rexx} ")
                except Exception as e:
                    logger.error(f"Failed to parse REXX file {rexx_file.name}: {e}", exc_info=True)

            # Consolidate and persist analysis.json under project_analysis/<id>/analysis/
            analysis_payload = {
                "project_id": str(project.id),
                "project_name": project.name,
                "jcl": jcl_results,
                "cobol": cobol_results,
                "copybooks": copybook_results,
                "rexx": rexx_results
            }
            out_dir = _get_project_analysis_dir(project.id)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "analysis.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(analysis_payload, f, indent=2)
            logger.info(f"Wrote consolidated analysis to {out_file}")

            # Build, log, and persist the wiki tree immediately after writing analysis.json
            try:
                tree = wiki_tree_service.build_pages_tree(analysis_payload)
                logger.info(f"Wiki structure (tree) for project {project.id}:\n{json.dumps(tree, indent=2)}")
                wiki_out = wiki_tree_service.get_wiki_structure_file_path(project.id)
                with open(wiki_out, "w", encoding="utf-8") as wf:
                    json.dump(tree, wf, indent=2)
                logger.info(f"Wrote wiki structure to {wiki_out}")
            except Exception as e:
                logger.error(f"Failed to build/persist wiki structure for project {project.id}: {e}", exc_info=True)
                project.wiki_status = WikiStatus.FAILED
                db.add(project)
                db.commit()
                return  # Can't proceed to wiki page generation without a tree

            # Generate and persist wiki pages under project_wiki/<id> now that the tree is available
            try:
                manifest = wiki_generation_service.generate_persist_and_log_wiki(project.id, tree)
                logger.info(f"Wiki generation completed for project {project.id}. Summary:\n{json.dumps(manifest, indent=2)}")
                project.wiki_status = WikiStatus.GENERATED
            except Exception as e:
                logger.error(f"Failed to generate wiki pages for project {project.id}: {e}", exc_info=True)
                project.wiki_status = WikiStatus.FAILED

            logger.info(f"Successfully completed analysis for project: {project.name} ({project_id})")

        except Exception as e:
            logger.error(f"A critical error occurred during analysis for project {project_id}: {e}", exc_info=True)
            project = db.get(Project, project_id)
            if project:
                project.wiki_status = WikiStatus.FAILED
        finally:
            if 'project' in locals() and project:
                db.add(project)
                db.commit()