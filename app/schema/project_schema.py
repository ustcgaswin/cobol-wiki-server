from typing import List, Optional
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, ConfigDict, HttpUrl


class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectCreate(ProjectBase):
    github_url: Optional[HttpUrl] = None
    github_token: Optional[str] = None
    


class ProjectRead(ProjectBase):
    id: UUID
    github_url: Optional[HttpUrl] = None
    wiki_status: str
    created_at: datetime

    # Pydantic v2: replace orm_mode with from_attributes
    model_config = ConfigDict(from_attributes=True)


class ProjectList(BaseModel):
    projects: List[ProjectRead]