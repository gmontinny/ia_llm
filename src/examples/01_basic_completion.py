"""
Exemplo 1: Completação Básica de Texto
Demonstra como usar diferentes provedores para gerar texto simples.
"""
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.model_factory import ModelFactory
from src.models.base_model import ModelConfig
from config.settings import settings


def example_basic_completion():
    """Demonstra completação básica com diferentes modelos."""

    print("=" * 70)
    print("EXEMPLO 1: COMPLETAÇÃO BÁSICA DE TEXTO")
    print("=" * 70)

    # Prompt de exemplo
    prompt = "Explique em 2 parágrafos o que são Large Language Models (LLMs)."

    print(f"\n📝 Prompt: {prompt}\n")

    # Testa cada provedor disponível
    providers = ['openai', 'huggingface', 'ollama']

    for provider in providers:
        print(f"\n{'=' * 70}")
        print(f"🤖 Testando: {provider.upper()}")
        print('=' * 70)

        try:
            # Cria modelo com configuração personalizada
            config = ModelConfig(
                model_name=getattr(settings, f'DEFAULT_{provider.upper()}_MODEL'),
                temperature=0.7,
                max_tokens=200
            )

            model = ModelFactory.create_model(provider, config)

            # Gera resposta
            print(f"\n⏳ Gerando resposta...\n")
            response = model.generate(prompt)

            print(f"✅ Resposta:\n{response}\n")

        except ValueError as e:
            print(f"⚠️  {provider} não disponível: {e}\n")
        except Exception as e:
            print(f"❌ Erro ao usar {provider}: {e}\n")


if __name__ == "__main__":
    example_basic_completion()