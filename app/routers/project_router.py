from fastapi import APIRouter, UploadFile, File, Form, status, HTTPException, BackgroundTasks
from typing import List, Optional, Any, Dict
from uuid import UUID
import json

from app.schema.project_schema import ProjectCreate, ProjectRead
from app.schema.api_schema import APIResponse, ErrorDetail
from app.schema.analysis_schema import AnalysisStatus
from app.services.project_service import (
    create_project_from_github,
    create_project_from_files,
    get_project,
    list_projects,
    delete_project,
    ProjectCreationError,
    ProjectDeletionError
 )
from app.config.db_config import SessionDep
from app.models.project_model import WikiStatus
from app.services import analysis_service, rag_service
from app.utils.logger import logger


project_router = APIRouter(
    prefix="/projects",
    tags=["projects"],
    responses={404: {"description": "Not found"}},
)

@project_router.post(
    "/upload_github",
    response_model=APIResponse[ProjectRead],
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": APIResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": APIResponse},
    }
)
async def create_project_github_endpoint(
    db: SessionDep,
    name: str = Form(...),
    description: Optional[str] = Form(None),
    github_url: str = Form(...),
    github_token: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
):
    data = ProjectCreate(
        name=name,
        description=description,
        github_url=github_url,
        github_token=github_token,
    )
    try:
        project = create_project_from_github(db, data)

        # Auto-start pipeline (RAG -> Analysis)
        project.wiki_status = WikiStatus.ANALYZING
        db.add(project)
        db.commit()
        db.refresh(project)
        if background_tasks:
            logger.info(f"Adding pipeline task (RAG -> Analysis) for project {project.name} ({project.id}) to background queue.")
            background_tasks.add_task(analysis_service.start_project_pipeline, project.id)

        return APIResponse(
            success=True,
            message="Project created from GitHub successfully. Pipeline started in the background.",
            data=project,
            count=1,
        )
    except ProjectCreationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=APIResponse(
                success=False,
                message=str(e),
                error=e.error_detail
            ).model_dump()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=APIResponse(
                success=False,
                message="An unexpected server error occurred.",
                error=ErrorDetail(code="UNEXPECTED_ERROR", details=str(e))
            ).model_dump()
        )

@project_router.post(
    "/upload_files",
    response_model=APIResponse[ProjectRead],
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": APIResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": APIResponse},
    }
)
async def create_project_upload_endpoint(
    db: SessionDep,
    name: str = Form(...),
    description: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
):
    valid_files = [f for f in files if f and getattr(f, "filename", None) and f.filename.strip() != ""]
    data = ProjectCreate(name=name, description=description)

    try:
        project = create_project_from_files(db, data, files=valid_files)

        # Auto-start pipeline (RAG -> Analysis)
        project.wiki_status = WikiStatus.ANALYZING
        db.add(project)
        db.commit()
        db.refresh(project)
        if background_tasks:
            logger.info(f"Adding pipeline task (RAG -> Analysis) for project {project.name} ({project.id}) to background queue.")
            background_tasks.add_task(analysis_service.start_project_pipeline, project.id)

        return APIResponse(
            success=True,
            message="Project created from files successfully. Pipeline started in the background.",
            data=project,
            count=1,
        )
    except ProjectCreationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=APIResponse(
                success=False,
                message=str(e),
                error=e.error_detail
            ).model_dump()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=APIResponse(
                success=False,
                message="An unexpected server error occurred.",
                error=ErrorDetail(code="UNEXPECTED_ERROR", details=str(e))
            ).model_dump()
        )

@project_router.get("/", response_model=APIResponse[List[ProjectRead]])
async def list_projects_endpoint(db: SessionDep):
    projects = list_projects(db)
    return APIResponse(
        success=True,
        message="Projects fetched successfully",
        data=projects,
        count=len(projects),
    )

@project_router.get("/{project_id}", response_model=APIResponse[ProjectRead])
async def get_project_endpoint(project_id: UUID, db: SessionDep):
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=APIResponse(
                success=False,
                message="Project not found",
                error=ErrorDetail(code="NOT_FOUND", details="No project with the given ID")
            ).model_dump()
        )
    return APIResponse(
        success=True,
        message="Project fetched successfully",
        data=project,
        count=1,
    )

@project_router.delete("/{project_id}", response_model=APIResponse[None], status_code=status.HTTP_200_OK)
async def delete_project_endpoint(project_id: UUID, db: SessionDep):
    try:
        ok = await delete_project(db, project_id)
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=APIResponse(
                    success=False,
                    message="Project not found",
                    error=ErrorDetail(code="NOT_FOUND", details="No project with the given ID")
                ).model_dump()
            )
        return APIResponse(
            success=True,
            message="Project deleted successfully",
        )
    except ProjectDeletionError as e:
        # This handles cases where the DB entry was deleted but files were not.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=APIResponse(
                success=False,
                message=str(e),
                error=e.error_detail
            ).model_dump()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=APIResponse(
                success=False,
                message="An unexpected server error occurred during deletion.",
                error=ErrorDetail(code="UNEXPECTED_DELETION_ERROR", details=str(e))
            ).model_dump()
        )

# Analysis status under projects
@project_router.get(
    "/{project_id}/analysis/status",
    response_model=APIResponse[AnalysisStatus],
    status_code=status.HTTP_200_OK,
)
async def get_project_analysis_status(project_id: UUID, db: SessionDep):
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=APIResponse(
                success=False,
                message="Project not found",
                error=ErrorDetail(code="NOT_FOUND", details="No project with the given ID")
            ).model_dump()
        )
    return APIResponse(
        success=True,
        message="Status retrieved.",
        data=AnalysisStatus.model_validate(project),
    )

# Analysis result under projects
@project_router.get(
    "/{project_id}/analysis/result",
    response_model=APIResponse[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
)
async def get_project_analysis_result(project_id: UUID, db: SessionDep):
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=APIResponse(
                success=False,
                message="Project not found",
                error=ErrorDetail(code="NOT_FOUND", details="No project with the given ID")
            ).model_dump()
        )

    analysis_file = analysis_service.get_analysis_file_path(project_id)
    if not analysis_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=APIResponse(
                success=False,
                message="Analysis result not found. It may still be running or failed.",
                error=ErrorDetail(code="RESULT_NOT_READY", details="analysis.json not found")
            ).model_dump()
        )

    with open(analysis_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return APIResponse(
        success=True,
        message="Analysis result fetched.",
        data=payload,
        count=1,
    )

# RAG status under projects
@project_router.get(
    "/{project_id}/rag/status",
    response_model=APIResponse[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
)
async def get_project_rag_status(project_id: UUID, db: SessionDep):
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=APIResponse(
                success=False,
                message="Project not found",
                error=ErrorDetail(code="NOT_FOUND", details="No project with the given ID")
            ).model_dump()
        )
    status_payload = rag_service.get_embedding_status(project_id)
    return APIResponse(
        success=True,
        message="RAG status retrieved.",
        data=status_payload,
        count=1,
    )