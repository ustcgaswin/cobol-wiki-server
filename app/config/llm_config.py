import dspy
from .app_config import settings

# LLM setup
deployment_url = settings.AZURE_OPENAI_ENDPOINT
api_key = settings.AZURE_OPENAI_API_KEY
deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME
model = settings.AZURE_OPENAI_MODEL
version = settings.AZURE_OPENAI_API_VERSION

llm = dspy.LM(
    f'azure/{deployment_name}',
    api_key=api_key,
    api_base=deployment_url,
    api_version=version,
    max_tokens=16384
    
)

# Embedder setup
embed_api_base = settings.AZURE_OPENAI_EMBED_API_ENDPOINT
embed_api_key = settings.AZURE_OPENAI_EMBED_API_KEY
embed_deployment_name = settings.AZURE_OPENAI_EMBED_DEPLOYMENT_NAME
embed_model = settings.AZURE_OPENAI_EMBED_MODEL
embed_version = settings.AZURE_OPENAI_EMBED_VERSION

embedder = dspy.Embedder(
    model=f'azure/{embed_deployment_name}',
    api_key=embed_api_key,
    api_base=embed_api_base,
    api_version=embed_version
)