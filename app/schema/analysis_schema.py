from uuid import UUID
from pydantic import BaseModel, ConfigDict
from app.models.project_model import WikiStatus

class AnalysisStatus(BaseModel):
    """
    Represents the data returned when analysis is triggered.
    It's designed to be nested within the generic APIResponse.
    """
    id: UUID
    wiki_status: WikiStatus

    model_config = ConfigDict(from_attributes=True)