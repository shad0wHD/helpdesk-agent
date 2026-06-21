from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    groq_api_key: str
    # Only needed for production RAG embeddings (voyage-3); not required for demo mode
    anthropic_api_key: str = ""
    slack_bot_token: str
    slack_signing_secret: str
    slack_app_token: str

    jira_base_url: str
    jira_email: str
    jira_api_token: str
    jira_project_key: str = "SCRUM"

    database_url: str = "postgresql+psycopg://agent:agent@localhost:5432/servicedesk"

    log_level: str = "INFO"
    environment: str = "development"

    # LLM model used throughout the agent graph
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"


settings = Settings()  # type: ignore[call-arg]
