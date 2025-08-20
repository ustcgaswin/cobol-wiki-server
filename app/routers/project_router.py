from fastapi import APIRouter, UploadFile, File, Form, status, HTTPException
from typing import List, Optional, Dict
from uuid import UUID
import json
from pathlib import Path

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
    ProjectDeletionError,
)
from app.config.db_config import SessionDep
from app.models.project_model import WikiStatus
from app.services import analysis_service
from app.utils.logger import logger
from app.services.wiki_tree_service import get_wiki_structure_for_project

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
    },
)
async def create_project_github_endpoint(
    db: SessionDep,
    name: str = Form(...),
    description: Optional[str] = Form(None),
    github_url: str = Form(...),
    github_token: Optional[str] = Form(None),
):
    data = ProjectCreate(
        name=name,
        description=description,
        github_url=github_url,
        github_token=github_token,
    )
    try:
        project = create_project_from_github(db, data)

        project.wiki_status = WikiStatus.ANALYZING
        db.add(project)
        db.commit()
        db.refresh(project)

        analysis_service.start_project_pipeline_background(project.id)
        logger.info(f"Started pipeline for project {project.name} ({project.id})")

        return APIResponse(
            success=True,
            message="Project created from GitHub successfully.",
            data=project,
            count=1,
        )
    except ProjectCreationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=APIResponse(
                success=False,
                message=str(e),
                error=e.error_detail,
            ).model_dump(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=APIResponse(
                success=False,
                message="An unexpected server error occurred.",
                error=ErrorDetail(code="UNEXPECTED_ERROR", details=str(e)),
            ).model_dump(),
        )


@project_router.post(
    "/upload_files",
    response_model=APIResponse[ProjectRead],
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": APIResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": APIResponse},
    },
)
async def create_project_upload_endpoint(
    db: SessionDep,
    name: str = Form(...),
    description: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
):
    valid_files = [f for f in files if f and getattr(f, "filename", None) and f.filename.strip() != ""]
    data = ProjectCreate(name=name, description=description)

    try:
        project = create_project_from_files(db, data, files=valid_files)

        project.wiki_status = WikiStatus.ANALYZING
        db.add(project)
        db.commit()
        db.refresh(project)

        analysis_service.start_project_pipeline_background(project.id)
        logger.info(f"Started pipeline for project {project.name} ({project.id})")

        return APIResponse(
            success=True,
            message="Project created from files successfully.",
            data=project,
            count=1,
        )
    except ProjectCreationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=APIResponse(
                success=False,
                message=str(e),
                error=e.error_detail,
            ).model_dump(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=APIResponse(
                success=False,
                message="An unexpected server error occurred.",
                error=ErrorDetail(code="UNEXPECTED_ERROR", details=str(e)),
            ).model_dump(),
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
                error=ErrorDetail(code="NOT_FOUND", details="No project with the given ID"),
            ).model_dump(),
        )
    return APIResponse(
        success=True,
        message="Project fetched successfully",
        data=project,
        count=1,
    )


@project_router.delete(
    "/{project_id}",
    response_model=APIResponse[None],
    status_code=status.HTTP_200_OK,
)
async def delete_project_endpoint(project_id: UUID, db: SessionDep):
    try:
        ok = await delete_project(db, project_id)
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=APIResponse(
                    success=False,
                    message="Project not found",
                    error=ErrorDetail(code="NOT_FOUND", details="No project with the given ID"),
                ).model_dump(),
            )
        return APIResponse(
            success=True,
            message="Project deleted successfully",
        )
    except ProjectDeletionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=APIResponse(
                success=False,
                message=str(e),
                error=e.error_detail,
            ).model_dump(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=APIResponse(
                success=False,
                message="An unexpected server error occurred during deletion.",
                error=ErrorDetail(code="UNEXPECTED_DELETION_ERROR", details=str(e)),
            ).model_dump(),
        )


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
                error=ErrorDetail(code="NOT_FOUND", details="No project with the given ID"),
            ).model_dump(),
        )
    return APIResponse(
        success=True,
        message="Status retrieved.",
        data=AnalysisStatus.model_validate(project),
    )


@project_router.get(
    "/{project_id}/analysis/result",
    response_model=APIResponse[Dict[str, object]],
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
                error=ErrorDetail(code="NOT_FOUND", details="No project with the given ID"),
            ).model_dump(),
        )

    analysis_file = analysis_service.get_analysis_file_path(project_id)
    if not analysis_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=APIResponse(
                success=False,
                message="Analysis result not found. It may still be running or failed.",
                error=ErrorDetail(code="RESULT_NOT_READY", details="analysis.json not found"),
            ).model_dump(),
        )

    with open(analysis_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return APIResponse(
        success=True,
        message="Analysis result fetched.",
        data=payload,
        count=1,
    )


@project_router.get(
    "/{project_id}/wiki/content",
    response_model=APIResponse[dict],
    status_code=status.HTTP_200_OK,
)
async def get_project_wiki_content(project_id: UUID, db: SessionDep):
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=APIResponse(
                success=False,
                message="Project not found",
                error=ErrorDetail(code="NOT_FOUND", details="No project with the given ID"),
            ).model_dump(),
        )

    if project.wiki_status != WikiStatus.GENERATED:
        return APIResponse(
            success=False,
            message="Wiki not generated yet. Please check back later.",
            data={},
            count=0,
        )

    wiki_dir = Path("project_wiki") / str(project_id)
    if not wiki_dir.exists():
        return APIResponse(
            success=False,
            message="Wiki directory not found, even though status is GENERATED.",
            data={},
            count=0,
        )

    try:
        wiki_tree = get_wiki_structure_for_project(project_id)
    except Exception as e:
        return APIResponse(
            success=False,
            message=f"Failed to load wiki tree: {e}",
            data={},
            count=0,
        )

    def fill_tree_with_content(tree, prefix=""):
        result = {}
        for key, value in tree.items():
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict) and value:
                result[key] = fill_tree_with_content(value, path)
            else:
                md_path = wiki_dir / f"{path}.md"
                if md_path.exists():
                    with open(md_path, "r", encoding="utf-8") as f:
                        result[key] = f.read()
                else:
                    result[key] = None
        return result

    wiki_content = fill_tree_with_content(wiki_tree)

    return APIResponse(
        success=True,
        message="Wiki content fetched in tree structure.",
        data=wiki_content,
        count=len(wiki_content),
    )