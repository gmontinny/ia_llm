"""Factory para criar modelos LLM de forma dinâmica."""
from typing import Optional, Type
from .base_model import BaseLLMModel, ModelConfig
from .openai_model import OpenAIModel
from .huggingface_model import HuggingFaceModel
from .ollama_model import OllamaModel
from config.settings import settings
from src.utils.logger import default_logger


class ModelFactory:
    """Factory para criação de modelos LLM."""

    _models = {
        'openai': OpenAIModel,
        'huggingface': HuggingFaceModel,
        'hf': HuggingFaceModel,
        'ollama': OllamaModel
    }

    @classmethod
    def create_model(
            cls,
            provider: str,
            config: Optional[ModelConfig] = None
    ) -> BaseLLMModel:
        """
        Cria um modelo baseado no provedor especificado.

        Args:
            provider: Nome do provedor (openai, huggingface, ollama)
            config: Configuração opcional do modelo

        Returns:
            Instância do modelo

        Raises:
            ValueError: Se o provedor não for suportado
        """
        provider_lower = provider.lower()

        if provider_lower not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(
                f"Provedor '{provider}' não suportado. "
                f"Disponíveis: {available}"
            )

        model_class = cls._models[provider_lower]

        try:
            model = model_class(config)
            default_logger.info(f"Modelo criado: {provider} - {model.config.model_name}")
            return model
        except Exception as e:
            default_logger.error(f"Erro ao criar modelo {provider}: {e}")
            raise

    @classmethod
    def get_available_providers(cls) -> list:
        """Retorna lista de provedores disponíveis."""
        api_status = settings.validate_api_keys()
        available = []

        for provider in cls._models.keys():
            if provider in api_status:
                if api_status[provider]:
                    available.append(provider)
            elif provider == 'ollama':
                # Ollama não precisa de API key
                available.append(provider)

        return list(set(available))  # Remove duplicatas

    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseLLMModel]):
        """
        Registra um novo modelo customizado.

        Args:
            name: Nome do provedor
            model_class: Classe do modelo
        """
        cls._models[name.lower()] = model_class
        default_logger.info(f"Modelo customizado registrado: {name}")