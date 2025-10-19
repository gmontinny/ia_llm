"""Implementação do modelo Ollama (local)."""
from typing import List, Dict, Any, Generator, Optional
from functools import wraps
try:
    from langchain_ollama import ChatOllama
except ImportError:  # fallback to community integration
    from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config.settings import settings
from .base_model import BaseLLMModel, ModelConfig


def ollama_error_handler(func):
    """Um decorador para tratar erros comuns da API Ollama."""
    @wraps(func)
    def wrapper(self: 'OllamaModel', *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            msg = str(e)
            if "status code 404" in msg or " 404" in msg:
                hint = (
                    f"O modelo '{self.config.model_name}' não foi encontrado localmente no Ollama. "
                    f"Para corrigir, execute o comando `ollama pull {self.config.model_name}` no seu terminal."
                )
                self.logger.error(f"Erro em '{func.__name__}': {msg} | Dica: {hint}")
                raise ValueError(hint) from e
            
            self.logger.error(f"Erro em '{func.__name__}': {e}")
            raise
    return wrapper


class OllamaModel(BaseLLMModel):
    """Implementação para modelos Ollama (execução local)."""

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Inicializa o modelo Ollama.

        Args:
            config: Configuração do modelo (usa padrão se não fornecida)
        """
        if config is None:
            config = ModelConfig(
                model_name=settings.DEFAULT_OLLAMA_MODEL,
                temperature=settings.DEFAULT_TEMPERATURE,
                max_tokens=settings.DEFAULT_MAX_TOKENS
            )

        super().__init__(config)
        self._initialize_client()

    def _initialize_client(self):
        """Inicializa o cliente Ollama."""
        try:
            self._client = ChatOllama(
                model=self.config.model_name,
                temperature=self.config.temperature,
                **self.config.additional_params
            )
            self.logger.info(f"Cliente Ollama inicializado: {self.config.model_name}")
        except Exception as e:
            self.logger.error(f"Erro ao inicializar Ollama: {e}")
            self.logger.info("Certifique-se de que o Ollama está instalado e rodando")
            raise

    @ollama_error_handler
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Gera uma resposta para o prompt.

        Args:
            prompt: Texto de entrada
            **kwargs: Parâmetros adicionais

        Returns:
            Resposta gerada
        """
        messages = [HumanMessage(content=prompt)]
        response = self._client.invoke(messages)
        return response.content

    @ollama_error_handler
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Gera uma resposta em streaming.

        Args:
            prompt: Texto de entrada
            **kwargs: Parâmetros adicionais

        Yields:
            Chunks da resposta
        """
        messages = [HumanMessage(content=prompt)]
        for chunk in self._client.stream(messages):
            if hasattr(chunk, 'content'):
                yield chunk.content

    @ollama_error_handler
    def chat(
            self,
            messages: List[Dict[str, str]],
            system_message: Optional[str] = None,
            **kwargs
    ) -> str:
        """
        Interface de chat com histórico.

        Args:
            messages: Lista de mensagens
            system_message: Mensagem de sistema opcional
            **kwargs: Parâmetros adicionais

        Returns:
            Resposta do assistente
        """
        formatted_messages = []

        if system_message:
            formatted_messages.append(SystemMessage(content=system_message))

        for msg in messages:
            if msg['role'] == 'user':
                formatted_messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                formatted_messages.append(AIMessage(content=msg['content']))

        response = self._client.invoke(formatted_messages)
        return response.content