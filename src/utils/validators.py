"""Funções de validação para a aplicação."""
from typing import Any, List, Optional
import re


def validate_temperature(temperature: float) -> bool:
    """Valida se a temperatura está no intervalo válido [0, 2]."""
    return 0 <= temperature <= 2


def validate_max_tokens(max_tokens: int) -> bool:
    """Valida se max_tokens é um valor positivo."""
    return max_tokens > 0


def validate_model_name(model_name: str, provider: str) -> bool:
    """
    Valida se o nome do modelo é válido para o provedor especificado.

    Args:
        model_name: Nome do modelo
        provider: Provedor (openai, huggingface, ollama, groq)

    Returns:
        True se válido, False caso contrário
    """
    if not model_name or not isinstance(model_name, str):
        return False

    patterns = {
        'openai': r'^(gpt-|text-|davinci)',
        'huggingface': r'^[\w\-]+/[\w\-\.]+$',
        'ollama': r'^[\w\-]+$',
        'groq': r'^[\w\-]+$'
    }

    pattern = patterns.get(provider.lower())
    if pattern:
        return bool(re.match(pattern, model_name))

    return True


def validate_api_key(api_key: Optional[str]) -> bool:
    """Valida se uma API key está presente e não vazia."""
    return bool(api_key and api_key.strip())