import uuid
import dspy
import mlflow
from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.utils.logger import setup_logging, logger
from app.config.llm_config import llm
from app.config.db_config import create_db_and_tables
from app.routers.project_router import project_router



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure libraries that might interfere with logging FIRST.
    # MLflow is a known library to do this.
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("DSPy")
    mlflow.config.enable_async_logging(True)
    mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)
    
    # NOW, set up our custom logging. This will override any changes made by other libraries.
    setup_logging()

    logger.info("Creating database and tables…")
    create_db_and_tables()
    
    logger.info("Configuring DSPy LLM...")
    dspy.settings.configure(lm=llm,track_usage=True)
    logger.info("DSPy LLM configured.")
    yield
    logger.info("Application shutdown")

app = FastAPI(
    title="Cobol wiki generator",
    description="Tool to generate wiki for cobol codebases",
    lifespan=lifespan,
)

# Middleware to inject a request ID
# This should be one of the first middleware to be added
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    # Attach it to Loguru’s context
    logger.bind(request_id=rid)
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response

# CORS
# It's better to be explicit with origins than using a wildcard
origins = [
    "http://localhost:5173", # Your React frontend
    "http://localhost:5174"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Use the origins list
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def get_health_status():
    logger.info("Health check endpoint hit")
    return {"status": "ok"}

# API v1
api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(project_router)
app.include_router(api_v1_router)