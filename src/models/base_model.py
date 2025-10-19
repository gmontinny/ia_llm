"""Classe base abstrata para todos os modelos LLM."""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass
from config.settings import settings
from ..utils.logger import Logger


@dataclass
class ModelConfig:
    """Configuração para modelos LLM."""
    model_name: str
    temperature: float = settings.DEFAULT_TEMPERATURE
    max_tokens: int = settings.DEFAULT_MAX_TOKENS
    streaming: bool = False
    additional_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


class BaseLLMModel(ABC):
    """Classe base abstrata para modelos LLM."""

    def __init__(self, config: ModelConfig):
        """
        Inicializa o modelo base.

        Args:
            config: Configuração do modelo
        """
        self.config = config
        self.logger = Logger.get_logger(self.__class__.__name__)
        self._client = None

    @abstractmethod
    def _initialize_client(self):
        """Inicializa o cliente do provedor específico."""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Gera uma resposta para o prompt fornecido.

        Args:
            prompt: Texto de entrada
            **kwargs: Parâmetros adicionais

        Returns:
            Resposta gerada
        """
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Gera uma resposta em streaming.

        Args:
            prompt: Texto de entrada
            **kwargs: Parâmetros adicionais

        Yields:
            Chunks da resposta
        """
        pass

    @abstractmethod
    def chat(
            self,
            messages: List[Dict[str, str]],
            **kwargs
    ) -> str:
        """
        Interface de chat com histórico de mensagens.

        Args:
            messages: Lista de mensagens no formato [{"role": "user", "content": "..."}]
            **kwargs: Parâmetros adicionais

        Returns:
            Resposta do modelo
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo."""
        return {
            'provider': self.__class__.__name__,
            'model_name': self.config.model_name,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
            'streaming': self.config.streaming
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model_name})"