from typing import Generic, TypeVar, Optional
from pydantic import BaseModel
from pydantic.generics import GenericModel

T = TypeVar("T")

class ErrorDetail(BaseModel):
    code: Optional[str] = None
    details: Optional[str] = None

class APIResponse(GenericModel, Generic[T]):
    success: bool
    message: str
    data: Optional[T] = None
    count: Optional[int] = None
    error: Optional[ErrorDetail] = None