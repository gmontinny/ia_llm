"""
Exemplo 3: Respostas em Streaming
Demonstra como receber respostas token por token.
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.model_factory import ModelFactory
from src.models.base_model import ModelConfig


def example_streaming():
    """Demonstra streaming de respostas."""

    print("=" * 70)
    print("EXEMPLO 3: STREAMING DE RESPOSTAS")
    print("=" * 70)

    provider = 'openai'  # ou 'huggingface', 'ollama'

    try:
        print(f"\n🤖 Inicializando {provider} com streaming...\n")

        config = ModelConfig(
            model_name="gpt-4o-mini",
            temperature=0.7,
            streaming=True
        )

        model = ModelFactory.create_model(provider, config)

        prompt = "Conte uma história curta sobre um robô que aprende a sentir emoções."

        print(f"📝 Prompt: {prompt}\n")
        print("🤖 Resposta (streaming):\n")
        print("-" * 70)

        # Recebe resposta em streaming
        full_response = ""
        for chunk in model.generate_stream(prompt):
            print(chunk, end='', flush=True)
            full_response += chunk
            time.sleep(0.01)  # Simula digitação

        print("\n" + "-" * 70)
        print(f"\n✅ Resposta completa recebida ({len(full_response)} caracteres)\n")

    except Exception as e:
        print(f"❌ Erro: {e}")


if __name__ == "__main__":
    example_streaming()