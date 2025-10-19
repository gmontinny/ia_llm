"""Implementação do modelo OpenAI."""
from typing import List, Dict, Any, Generator, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config.settings import settings
from src.utils.validators import validate_api_key
from .base_model import BaseLLMModel, ModelConfig


class OpenAIModel(BaseLLMModel):
    """Implementação para modelos OpenAI."""

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Inicializa o modelo OpenAI.

        Args:
            config: Configuração do modelo (usa padrão se não fornecida)
        """
        if config is None:
            config = ModelConfig(
                model_name=settings.DEFAULT_OPENAI_MODEL,
                temperature=settings.DEFAULT_TEMPERATURE,
                max_tokens=settings.DEFAULT_MAX_TOKENS
            )

        super().__init__(config)

        if not validate_api_key(settings.OPENAI_API_KEY):
            raise ValueError("OpenAI API key não encontrada ou inválida")

        self._initialize_client()

    def _initialize_client(self):
        """Inicializa o cliente OpenAI."""
        self._client = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            streaming=self.config.streaming,
            api_key=settings.OPENAI_API_KEY,
            **self.config.additional_params
        )
        self.logger.info(f"Cliente OpenAI inicializado: {self.config.model_name}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Gera uma resposta para o prompt.

        Args:
            prompt: Texto de entrada
            **kwargs: Parâmetros adicionais

        Returns:
            Resposta gerada
        """
        try:
            messages = [HumanMessage(content=prompt)]
            response = self._client.invoke(messages)
            return response.content
        except Exception as e:
            self.logger.error(f"Erro ao gerar resposta: {e}")
            raise

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Gera uma resposta em streaming.

        Args:
            prompt: Texto de entrada
            **kwargs: Parâmetros adicionais

        Yields:
            Chunks da resposta
        """
        try:
            messages = [HumanMessage(content=prompt)]
            for chunk in self._client.stream(messages):
                if hasattr(chunk, 'content'):
                    yield chunk.content
        except Exception as e:
            self.logger.error(f"Erro no streaming: {e}")
            raise

    def chat(
            self,
            messages: List[Dict[str, str]],
            system_message: Optional[str] = None,
            **kwargs
    ) -> str:
        """
        Interface de chat com histórico.

        Args:
            messages: Lista de mensagens [{"role": "user/assistant", "content": "..."}]
            system_message: Mensagem de sistema opcional
            **kwargs: Parâmetros adicionais

        Returns:
            Resposta do assistente
        """
        try:
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
        except Exception as e:
            self.logger.error(f"Erro no chat: {e}")
            raise