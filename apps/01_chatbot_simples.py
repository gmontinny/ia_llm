"""
Aplicação 1: Chatbot Simples
Chatbot com interface amigável e suporte a múltiplos provedores.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.models.model_factory import ModelFactory
from src.models.base_model import ModelConfig
from config.settings import settings

# Configuração da página
st.set_page_config(
    page_title="Chatbot Simples 🤖",
    page_icon="🤖",
    layout="wide"
)

# Título
st.title("🤖 Chatbot Simples com LLMs")
st.markdown("Converse com diferentes modelos de linguagem!")

# Inicializa o histórico de chat no início da execução
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá! Sou seu assistente virtual. Como posso ajudar você?")
    ]

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações")

    # Seleção de provedor
    available_providers = ModelFactory.get_available_providers()

    if not available_providers:
        st.error("❌ Nenhum provedor disponível! Verifique suas API keys.")
        st.stop()

    provider = st.selectbox(
        "Provedor",
        options=available_providers,
        help="Selecione o provedor de LLM"
    )

    # Configurações do modelo
    temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Controla a criatividade das respostas"
    )

    max_tokens = st.slider(
        "Máximo de Tokens",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
        help="Limite de tokens na resposta"
    )

    # Mensagem de sistema
    system_message = st.text_area(
        "Mensagem do Sistema",
        value="Você é um assistente prestativo e está respondendo perguntas em português.",
        height=100,
        help="Define o comportamento do assistente"
    )

    # Botão para limpar histórico
    if st.button("🗑️ Limpar Conversa", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # Informações
    st.markdown("---")
    st.markdown("### 📊 Estatísticas")
    msg_count = len([m for m in st.session_state.chat_history if isinstance(m, HumanMessage)])
    st.metric("Mensagens enviadas", msg_count)

# Exibe histórico de mensagens
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.write(message.content)

# Input do usuário
user_input = st.chat_input("Digite sua mensagem aqui...")

if user_input:
    # Adiciona mensagem do usuário
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("user", avatar="👤"):
        st.write(user_input)

    # Gera resposta
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Pensando..."):
            try:
                # Cria modelo com configurações
                config = ModelConfig(
                    model_name=getattr(settings, f'DEFAULT_{provider.upper()}_MODEL'),
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                model = ModelFactory.create_model(provider, config)

                # Prepara mensagens para o chat
                messages = []
                for msg in st.session_state.chat_history:
                    if isinstance(msg, HumanMessage):
                        messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        messages.append({"role": "assistant", "content": msg.content})

                # Gera resposta
                response = model.chat(messages, system_message=system_message)

                # Exibe resposta
                st.write(response)

                # Adiciona ao histórico
                st.session_state.chat_history.append(AIMessage(content=response))

            except Exception as e:
                st.error(f"❌ Erro ao gerar resposta: {e}")