"""
Exemplo 5: Comparação entre Modelos
Compara respostas de diferentes provedores para o mesmo prompt.
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.model_factory import ModelFactory
from src.models.base_model import ModelConfig


def example_model_comparison():
    """Compara respostas de diferentes modelos."""

    print("=" * 70)
    print("EXEMPLO 5: COMPARAÇÃO ENTRE MODELOS")
    print("=" * 70)

    # Prompt comum
    prompt = "Explique o conceito de 'transfer learning' em 3 frases simples."

    print(f"\n📝 Prompt comum: {prompt}\n")

    # Configuração comum
    common_config = {
        'temperature': 0.3,
        'max_tokens': 150
    }

    # Provedores a testar
    providers = ['openai', 'huggingface', 'ollama']

    results = []

    for provider in providers:
        print(f"\n{'=' * 70}")
        print(f"🤖 Testando: {provider.upper()}")
        print('=' * 70)

        try:
            # Cria modelo
            model = ModelFactory.create_model(provider)

            # Mede tempo de resposta
            start_time = time.time()
            response = model.generate(prompt)
            elapsed_time = time.time() - start_time

            # Armazena resultado
            results.append({
                'provider': provider,
                'response': response,
                'time': elapsed_time,
                'tokens': len(response.split()),
                'chars': len(response)
            })

            print(f"\n⏱️  Tempo: {elapsed_time:.2f}s")
            print(f"📊 Tokens: ~{len(response.split())} palavras")
            print(f"\n✅ Resposta:\n{response}\n")

        except Exception as e:
            print(f"❌ Erro: {e}\n")

    # Resumo comparativo
    if results:
        print("\n" + "=" * 70)
        print("📊 RESUMO COMPARATIVO")
        print("=" * 70)

        print(f"\n{'Provider':<15} {'Tempo (s)':<12} {'Palavras':<12} {'Caracteres'}")
        print("-" * 70)

        for result in results:
            print(f"{result['provider']:<15} {result['time']:<12.2f} {result['tokens']:<12} {result['chars']}")

        # Modelo mais rápido
        fastest = min(results, key=lambda x: x['time'])
        print(f"\n⚡ Mais rápido: {fastest['provider']} ({fastest['time']:.2f}s)")

        # Resposta mais longa
        longest = max(results, key=lambda x: x['chars'])
        print(f"📝 Resposta mais longa: {longest['provider']} ({longest['chars']} caracteres)")


if __name__ == "__main__":
    example_model_comparison()