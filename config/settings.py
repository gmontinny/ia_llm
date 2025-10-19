"""
Módulo de configuração centralizado para gerenciar variáveis de ambiente e configurações da aplicação.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Carrega variáveis de ambiente
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / 'env'

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()  # Tenta carregar .env padrão


class Settings:
    """Classe para gerenciar todas as configurações da aplicação."""

    # API Keys
    HUGGINGFACE_API_KEY: Optional[str] = os.getenv('HUGGINGFACE_API_KEY')
    HUGGINGFACEHUB_API_TOKEN: Optional[str] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    TAVILY_API_KEY: Optional[str] = os.getenv('TAVILY_API_KEY')
    SERPAPI_API_KEY: Optional[str] = os.getenv('SERPAPI_API_KEY')
    LANGCHAIN_API_KEY: Optional[str] = os.getenv('LANGCHAIN_API_KEY')
    GROQ_API_KEY: Optional[str] = os.getenv('GROQ_API_KEY')

    # Modelos padrão
    DEFAULT_OPENAI_MODEL: str = "gpt-4o-mini"
    DEFAULT_HF_MODEL: str = "google/flan-t5-base"
    # Backend padrão para Hugging Face (endpoint | pipeline)
    DEFAULT_HF_BACKEND: str = "pipeline"
    # Modelo de fallback: usar o mesmo para evitar diferenças entre endpoint e pipeline
    DEFAULT_HF_FALLBACK_MODEL: str = "google/flan-t5-base"
    # Template padrão de chat para Hugging Face (phi3, llama3, mistral, plain)
    # Usamos 'plain' por padrão para compatibilidade com modelos como DialoGPT.
    DEFAULT_HF_CHAT_TEMPLATE: str = "plain"
    # Alias de compatibilidade para exemplos e chamadas que usam HUGGINGFACE no nome
    DEFAULT_HUGGINGFACE_MODEL: str = DEFAULT_HF_MODEL
    DEFAULT_OLLAMA_MODEL: str = "phi3"
    DEFAULT_GROQ_MODEL: str = "llama3-70b-8192"
    DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-m3"

    # Parâmetros de geração
    DEFAULT_TEMPERATURE: float = 0.1
    DEFAULT_MAX_TOKENS: int = 1024

    # Configurações de RAG
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVER_K: int = 3
    RETRIEVER_FETCH_K: int = 4

    # Diretórios
    VECTORSTORE_DIR: Path = BASE_DIR / 'vectorstore'
    TEMP_DIR: Path = BASE_DIR / 'temp'
    LOGS_DIR: Path = BASE_DIR / 'logs'

    @classmethod
    def validate_api_keys(cls) -> dict:
        """Valida quais API keys estão disponíveis."""
        return {
            'openai': bool(cls.OPENAI_API_KEY),
            'huggingface': bool(cls.HUGGINGFACE_API_KEY),
            'tavily': bool(cls.TAVILY_API_KEY),
            'serpapi': bool(cls.SERPAPI_API_KEY),
            'langchain': bool(cls.LANGCHAIN_API_KEY),
            'groq': bool(cls.GROQ_API_KEY)
        }

    @classmethod
    def create_directories(cls):
        """Cria diretórios necessários se não existirem."""
        cls.VECTORSTORE_DIR.mkdir(exist_ok=True)
        cls.TEMP_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)


# Instância global de configurações
settings = Settings()
settings.create_directories()