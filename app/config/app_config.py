from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database settings
    DATABASE_URL: str

    # Azure OpenAI settings
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_DEPLOYMENT_NAME: str
    AZURE_OPENAI_MODEL: str
    AZURE_OPENAI_API_VERSION: str

    AZURE_OPENAI_EMBED_API_ENDPOINT:str
    AZURE_OPENAI_EMBED_API_KEY:str
    AZURE_OPENAI_EMBED_MODEL:str
    AZURE_OPENAI_EMBED_VERSION:str
    AZURE_OPENAI_EMBED_DEPLOYMENT_NAME:str

    LOG_LEVEL: str = "INFO"

    # This tells Pydantic to load variables from a .env file if it exists
    model_config = SettingsConfigDict(env_file=".env",extra="ignore")

# Create a single, importable instance of the settings
settings = Settings()