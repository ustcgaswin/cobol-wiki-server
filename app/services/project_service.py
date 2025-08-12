import os
import shutil
import zipfile
import asyncio
from uuid import UUID
from typing import List, Optional

from fastapi import UploadFile
from sqlmodel import Session, select

from app.models.project_model import Project
from app.schema.api_schema import ErrorDetail
from app.schema.project_schema import ProjectCreate
from app.utils.git_utils import clone_github_repo_async, rmtree_onerror
from app.utils.logger import logger

# Define the storage directory relative to the server root
SERVER_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROJECT_STORAGE_DIR = os.path.join(SERVER_ROOT, "project_storage")
ANALYSIS_STORAGE_DIR = os.path.join(SERVER_ROOT, "project_analysis")
WIKI_STORAGE_DIR = os.path.join(SERVER_ROOT, "project_wiki")


class ProjectCreationError(Exception):
    """Custom exception for project creation failures."""
    def __init__(self, message: str, error_detail: ErrorDetail):
        super().__init__(message)
        self.error_detail = error_detail

class ProjectDeletionError(Exception):
    """Custom exception for project deletion failures."""
    def __init__(self, message: str, error_detail: ErrorDetail):
        super().__init__(message)
        self.error_detail = error_detail

def save_uploaded_file(upload_file: UploadFile, dest_path: str):
    """Save a single uploaded file to the destination path."""
    with open(dest_path, "wb") as f:
        f.write(upload_file.file.read())

def extract_zip_file(upload_file: UploadFile, dest_dir: str):
    """Extract a zip file to the destination directory."""
    upload_file.file.seek(0)
    with zipfile.ZipFile(upload_file.file) as zf:
        zf.extractall(dest_dir)

def handle_uploaded_files(files: List[UploadFile], dest_dir: str):
    """Process uploaded files: extract zips, save others."""
    os.makedirs(dest_dir, exist_ok=True)
    for file in files:
        if not file or not getattr(file, "filename", None) or file.filename.strip() == "":
            continue
        filename = file.filename
        if filename.lower().endswith('.zip'):
            extract_zip_file(file, dest_dir)
        else:
            dest_path = os.path.join(dest_dir, filename)
            save_uploaded_file(file, dest_path)

def _check_duplicate_project_name(db: Session, name: str):
    """Checks for an existing project with the same name."""
    existing_project = db.exec(select(Project).where(Project.name == name)).first()
    if existing_project:
        raise ProjectCreationError(
            "A project with this name already exists.",
            ErrorDetail(code="DUPLICATE_NAME", details=f"A project named '{name}' already exists.")
        )

def create_project_from_github(db: Session, data: ProjectCreate):
    _check_duplicate_project_name(db, data.name)

    if data.github_url:
        existing_project_by_url = db.exec(select(Project).where(Project.github_url == str(data.github_url))).first()
        if existing_project_by_url:
            raise ProjectCreationError(
                "A project with this GitHub repository already exists.",
                ErrorDetail(code="DUPLICATE_GITHUB_URL", details=f"A project for repository '{data.github_url}' already exists.")
            )

    project = Project(
        name=data.name,
        description=data.description,
        github_url=str(data.github_url) if data.github_url else None,
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    project_dir = os.path.join(PROJECT_STORAGE_DIR, str(project.id))
    os.makedirs(project_dir, exist_ok=True)

    if not data.github_url:
        db.delete(project)
        db.commit()
        raise ProjectCreationError(
            "GitHub URL is required.",
            ErrorDetail(code="MISSING_URL", details="GitHub URL is required for this endpoint.")
        )

    future = clone_github_repo_async(
        str(data.github_url), data.github_token, str(project.id), project_dir
    )
    result = future.result(timeout=300)
    if not result["success"]:
        db.delete(project)
        db.commit()
        error_detail = ErrorDetail(
            code=result.get("code", "CLONE_FAILED"),
            details=result.get("details") or result.get("error")
        )
        raise ProjectCreationError(
            message=f"Failed to clone repository: {error_detail.details}",
            error_detail=error_detail
        )

    return project

def create_project_from_files(db: Session, data: ProjectCreate, files: List[UploadFile]):
    _check_duplicate_project_name(db, data.name)

    if not files:
        raise ProjectCreationError(
            "No files provided.",
            ErrorDetail(code="NO_FILES", details="At least one file is required for this endpoint.")
        )

    project = Project(
        name=data.name,
        description=data.description,
        github_url=None,
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    project_dir = os.path.join(PROJECT_STORAGE_DIR, str(project.id))
    os.makedirs(project_dir, exist_ok=True)

    try:
        handle_uploaded_files(files, project_dir)
    except Exception as e:
        db.delete(project)
        db.commit()
        shutil.rmtree(project_dir, ignore_errors=True)
        raise ProjectCreationError(
            "Failed to process uploaded files.",
            ErrorDetail(code="FILE_PROCESSING_ERROR", details=str(e))
        )

    return project

def get_project(db: Session, project_id: UUID) -> Optional[Project]:
    return db.get(Project, project_id)

def list_projects(db: Session) -> List[Project]:
    return db.exec(select(Project)).all()

async def delete_project(db: Session, project_id: UUID):
    project = db.get(Project, project_id)
    if not project:
        # The caller should handle this, e.g., by returning a 404 error.
        return False

    project_dir = os.path.join(PROJECT_STORAGE_DIR, str(project_id))
    analysis_dir = os.path.join(ANALYSIS_STORAGE_DIR, str(project_id))
    wiki_dir = os.path.join(WIKI_STORAGE_DIR, str(project_id))

    # Perform DB operation first
    db.delete(project)
    db.commit()

    # Run the blocking I/O operations in a separate thread
    errors: List[str] = []

    # Delete source files under project_storage/<id>
    try:
        if os.path.exists(project_dir):
            await asyncio.to_thread(shutil.rmtree, project_dir, onerror=rmtree_onerror)
    except FileNotFoundError:
        pass
    except Exception as e:
        # Log and accumulate the error; we will raise after attempting all deletions.
        logger.error(f"Error deleting project directory {project_dir}: {e}", exc_info=True)
        errors.append(f"Failed to remove directory {project_dir}: {e}")

    # Delete analysis artifacts under project_analysis/<id>
    try:
        if os.path.exists(analysis_dir):
            await asyncio.to_thread(shutil.rmtree, analysis_dir, onerror=rmtree_onerror)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.error(f"Error deleting analysis directory {analysis_dir}: {e}", exc_info=True)
        errors.append(f"Failed to remove directory {analysis_dir}: {e}")

    # Delete wiki pages under project_wiki/<id>
    try:
        if os.path.exists(wiki_dir):
            await asyncio.to_thread(shutil.rmtree, wiki_dir, onerror=rmtree_onerror)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.error(f"Error deleting wiki directory {wiki_dir}: {e}", exc_info=True)
        errors.append(f"Failed to remove directory {wiki_dir}: {e}")

    if errors:
        raise ProjectDeletionError(
            "Project deleted from database, but failed to remove project files.",
            ErrorDetail(code="FILE_DELETION_ERROR", details="; ".join(errors))
        )

    return True