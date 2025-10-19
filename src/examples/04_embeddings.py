"""
Exemplo 4: Embeddings e Similaridade
Demonstra como criar embeddings e calcular similaridade entre textos.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings


def cosine_similarity(vec1, vec2):
    """Calcula similaridade de cosseno entre dois vetores."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def example_embeddings():
    """Demonstra criação de embeddings e cálculo de similaridade."""

    print("=" * 70)
    print("EXEMPLO 4: EMBEDDINGS E SIMILARIDADE SEMÂNTICA")
    print("=" * 70)

    try:
        print(f"\n🔧 Carregando modelo de embeddings: {settings.DEFAULT_EMBEDDING_MODEL}\n")

        embeddings = HuggingFaceEmbeddings(
            model_name=settings.DEFAULT_EMBEDDING_MODEL
        )

        # Textos de exemplo
        texts = [
            "Machine Learning é um subcampo da Inteligência Artificial",
            "IA permite que computadores aprendam sem programação explícita",
            "Eu gosto de comer pizza no fim de semana",
            "Deep Learning usa redes neurais profundas para aprendizado",
        ]

        print("📝 Textos para análise:")
        for i, text in enumerate(texts, 1):
            print(f"   {i}. {text}")

        print("\n⏳ Gerando embeddings...\n")

        # Gera embeddings
        vectors = [embeddings.embed_query(text) for text in texts]

        print(f"✅ Embeddings gerados! Dimensão: {len(vectors[0])}\n")

        # Calcula matriz de similaridade
        print("📊 Matriz de Similaridade (cosseno):\n")
        print("     ", end="")
        for i in range(len(texts)):
            print(f"  T{i + 1}  ", end="")
        print()

        for i, vec1 in enumerate(vectors):
            print(f"T{i + 1}  ", end="")
            for j, vec2 in enumerate(vectors):
                similarity = cosine_similarity(vec1, vec2)
                print(f"{similarity:.3f}  ", end="")
            print()

        # Análise de similaridade
        print("\n" + "=" * 70)
        print("🔍 ANÁLISE DE SIMILARIDADE")
        print("=" * 70)

        # Compara texto 1 com todos os outros
        print("\nComparando 'Machine Learning...' com:")
        for i in range(1, len(texts)):
            sim = cosine_similarity(vectors[0], vectors[i])
            print(f"  • Texto {i + 1}: {sim:.3f} - {texts[i][:50]}...")

        print("\n💡 Interpretação:")
        print("  • Valores próximos a 1.0 = alta similaridade semântica")
        print("  • Valores próximos a 0.0 = baixa similaridade semântica")
        print("  • Textos sobre IA/ML têm maior similaridade entre si")

    except Exception as e:
        print(f"❌ Erro: {e}")


if __name__ == "__main__":
    example_embeddings()