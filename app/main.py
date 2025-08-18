import uuid
import dspy
import mlflow
import datetime
from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.utils.logger import setup_logging, logger
from app.config.llm_config import llm
from app.config.db_config import create_db_and_tables
from app.routers.project_router import project_router
from app.config.app_config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Prepare a readable timestamp to make sessions easy to find in the MLflow UI.
    now = datetime.datetime.now()
    readable_dt = now.strftime("%Y-%m-%d %H:%M:%S")
    slug_dt = now.strftime("%Y%m%d-%H%M%S")
    unique_experiment_name = f"DSPy-session-{slug_dt}-{uuid.uuid4().hex[:8]}"

    if settings.USE_MLFLOW_TRACKING:
        logger.info("MLflow tracking enabled.")
        try:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment(unique_experiment_name)
            # Tag the experiment.
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                exp = client.get_experiment_by_name(unique_experiment_name)
                if exp:
                    client.set_experiment_tag(exp.experiment_id, "dspy-session-date-time", readable_dt)
                    client.set_experiment_tag(exp.experiment_id, "dspy-session-slug", slug_dt)
            except Exception as e:
                logger.warning(f"Could not set experiment tags: {e}")
            try:
                mlflow.config.enable_async_logging(True)
            except Exception as e:
                logger.warning(f"Could not enable async logging: {e}")
            try:
                mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)
            except Exception as e:
                logger.warning(f"Failed to enable dspy autologging: {e}")
        except Exception as e:
            logger.error(f"MLflow initialization failed: {e}")
    else:
        logger.info("MLflow tracking disabled (use_mlflow_tracking = false).")

    # Set up logging after optional MLflow.
    setup_logging()

    logger.info("Creating database and tables…")
    create_db_and_tables()

    logger.info("Configuring DSPy LLM...")
    dspy.settings.configure(lm=llm, track_usage=True)
    logger.info("DSPy LLM configured.")
    yield
    logger.info("Application shutdown")


app = FastAPI(
    title="Cobol wiki generator",
    description="Tool to generate wiki for cobol codebases",
    lifespan=lifespan,
)

# Middleware to inject a request ID
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    logger.bind(request_id=rid)
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response

# CORS
origins = [
    "http://localhost:5173",
    "http://localhost:5174"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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