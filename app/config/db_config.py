from typing import Annotated
from sqlmodel import Session, SQLModel, create_engine
from fastapi import Depends
from .app_config import settings

engine = create_engine(settings.DATABASE_URL)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
