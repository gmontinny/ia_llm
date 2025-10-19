"""
Aplicação 2: LLM Playground
Interface para testar e comparar diferentes modelos side-by-side.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import time
from src.models.model_factory import ModelFactory
from src.models.base_model import ModelConfig
from config.settings import settings

# Configuração da página
st.set_page_config(
    page_title="LLM Playground 🎮",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 LLM Playground")
st.markdown("Compare diferentes modelos de linguagem lado a lado!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações Globais")

    temperature = st.slider("Temperatura", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Máximo de Tokens", 100, 1000, 300, 50)

    st.markdown("---")
    st.markdown("### 💡 Dicas")
    st.info("""
    - **Temperatura baixa (0.1-0.3)**: Respostas mais focadas e determinísticas
    - **Temperatura média (0.5-0.8)**: Equilíbrio entre criatividade e coerência
    - **Temperatura alta (0.9-2.0)**: Respostas mais criativas e variadas
    """)

# Área principal dividida em colunas
col1, col2 = st.columns(2)

# Provedores disponíveis
available_providers = ModelFactory.get_available_providers()

if len(available_providers) < 2:
    st.warning("⚠️ Pelo menos 2 provedores são necessários para comparação.")
    st.info(f"Provedores disponíveis: {', '.join(available_providers)}")

# Seleção de modelos
with col1:
    st.subheader("🤖 Modelo 1")
    provider1 = st.selectbox(
        "Provedor 1",
        options=available_providers,
        key="provider1"
    )

with col2:
    st.subheader("🤖 Modelo 2")
    provider2 = st.selectbox(
        "Provedor 2",
        options=available_providers,
        key="provider2",
        index=min(1, len(available_providers) - 1)
    )

# Input de prompt
st.markdown("---")
prompt = st.text_area(
    "📝 Digite seu prompt:",
    height=100,
    placeholder="Ex: Explique o que é inteligência artificial em termos simples"
)

# Botões de ação
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])

with col_btn1:
    generate_btn = st.button("🚀 Gerar Respostas", use_container_width=True, type="primary")

with col_btn2:
    example_btn = st.button("💡 Exemplo", use_container_width=True)

if example_btn:
    st.session_state[
        'example_prompt'] = "Explique o conceito de transfer learning em machine learning de forma simples."
    st.rerun()

if 'example_prompt' in st.session_state:
    prompt = st.session_state['example_prompt']
    del st.session_state['example_prompt']

# Geração de respostas
if generate_btn and prompt:
    st.markdown("---")

    col_res1, col_res2 = st.columns(2)

    # Modelo 1
    with col_res1:
        st.subheader(f"📤 {provider1.upper()}")

        with st.spinner("Gerando..."):
            try:
                config1 = ModelConfig(
                    model_name=getattr(settings, f'DEFAULT_{provider1.upper()}_MODEL'),
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                model1 = ModelFactory.create_model(provider1, config1)

                start_time = time.time()
                response1 = model1.generate(prompt)
                elapsed1 = time.time() - start_time

                st.success(f"✅ Gerado em {elapsed1:.2f}s")
                st.markdown(response1)

                # Métricas
                with st.expander("📊 Métricas"):
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("Tempo", f"{elapsed1:.2f}s")
                    col_m2.metric("Palavras", len(response1.split()))

            except Exception as e:
                st.error(f"❌ Erro: {e}")

    # Modelo 2
    with col_res2:
        st.subheader(f"📤 {provider2.upper()}")

        with st.spinner("Gerando..."):
            try:
                config2 = ModelConfig(
                    model_name=getattr(settings, f'DEFAULT_{provider2.upper()}_MODEL'),
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                model2 = ModelFactory.create_model(provider2, config2)

                start_time = time.time()
                response2 = model2.generate(prompt)
                elapsed2 = time.time() - start_time

                st.success(f"✅ Gerado em {elapsed2:.2f}s")
                st.markdown(response2)

                # Métricas
                with st.expander("📊 Métricas"):
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("Tempo", f"{elapsed2:.2f}s")
                    col_m2.metric("Palavras", len(response2.split()))

            except Exception as e:
                st.error(f"❌ Erro: {e}")

elif generate_btn:
    st.warning("⚠️ Por favor, digite um prompt antes de gerar.")