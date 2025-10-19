"""
Exemplo 2: Chat com Histórico
Demonstra como manter contexto entre múltiplas mensagens.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.model_factory import ModelFactory
from src.models.base_model import ModelConfig


def example_chat_with_history():
    """Demonstra conversação com histórico de mensagens."""

    print("=" * 70)
    print("EXEMPLO 2: CHAT COM HISTÓRICO")
    print("=" * 70)

    # Escolhe o provedor
    provider = 'openai'  # Altere para 'huggingface' ou 'ollama'

    try:
        # Cria o modelo
        print(f"\n🤖 Inicializando {provider}...\n")
        model = ModelFactory.create_model(provider)

        # Define o sistema
        system_message = "Você é um professor especialista em Inteligência Artificial."

        # Histórico de conversa
        messages = []

        # Primeira pergunta
        print("👤 Usuário: O que é Machine Learning?")
        messages.append({"role": "user", "content": "O que é Machine Learning?"})

        response = model.chat(messages, system_message=system_message)
        messages.append({"role": "assistant", "content": response})

        print(f"\n🤖 Assistente: {response}\n")
        print("-" * 70)

        # Segunda pergunta (com contexto)
        print("\n👤 Usuário: E qual a diferença para Deep Learning?")
        messages.append({
            "role": "user",
            "content": "E qual a diferença para Deep Learning?"
        })

        response = model.chat(messages, system_message=system_message)
        messages.append({"role": "assistant", "content": response})

        print(f"\n🤖 Assistente: {response}\n")
        print("-" * 70)

        # Terceira pergunta (contextual)
        print("\n👤 Usuário: Dê um exemplo prático disso")
        messages.append({
            "role": "user",
            "content": "Dê um exemplo prático disso"
        })

        response = model.chat(messages, system_message=system_message)

        print(f"\n🤖 Assistente: {response}\n")

    except Exception as e:
        print(f"❌ Erro: {e}")


if __name__ == "__main__":
    example_chat_with_history()