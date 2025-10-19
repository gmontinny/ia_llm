# 🤖 Introdução a Large Language Models (LLMs)

Aplicação educacional para aprender, experimentar e construir com LLMs usando provedores como OpenAI, Hugging Face e Ollama. Inclui exemplos práticos e apps em Streamlit.


## 📋 Índice

- [Características](#características)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Como Executar](#como-executar)
- [Exemplos de Código](#exemplos-de-código)
- [Apps (Streamlit)](#apps-streamlit)
- [Dicas e Solução de Problemas](#dicas-e-solução-de-problemas)


## ✨ Características

- Múltiplos provedores: OpenAI, Hugging Face (Inference API e pipeline local), Ollama
- Arquitetura modular e extensível (ModelFactory, BaseModel)
- Exemplos educativos cobrindo textos, chat com histórico, streaming e embeddings
- Apps em Streamlit prontos para uso
- Logging configurado e mensagens de erro úteis


## 📁 Estrutura do Projeto

```text
ia_llm/
├─ apps/
│  └─ 01_chatbot_simples.py         # App Streamlit: chatbot básico
├─ src/
│  ├─ examples/
│  │  ├─ 01_basic_completion.py     # Exemplo: conclusão básica
│  │  ├─ 02_chat_with_history.py    # Exemplo: chat com histórico
│  │  ├─ 03_streaming_responses.py  # Exemplo: streaming de respostas
│  │  ├─ 04_embeddings.py           # Exemplo: embeddings e buscas
│  │  └─ 05_comparing_models.py     # Exemplo: comparando modelos
│  ├─ models/
│  │  ├─ huggingface_model.py       # Integração com Hugging Face
│  │  ├─ ollama_model.py            # Integração com Ollama (local)
│  │  └─ model_factory.py           # Fábrica de modelos
│  └─ utils/                        # Utilidades e validadores
├─ config/
│  └─ settings.py                   # Configurações e variáveis de ambiente
├─ tests/                           # Testes automatizados (pytest)
├─ requirements.txt                 # Dependências do projeto
├─ README.md                        # Documentação do projeto
└─ notebooks (.ipynb)               # Materiais do curso e projetos
```


## 🧰 Requisitos

- Python 3.10+
- Pip atualizados: `python -m pip install --upgrade pip`
- Para usar pipeline local da Hugging Face (transformers): é necessário PyTorch instalado
  - Windows (CPU): `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  - Veja também: https://pytorch.org/get-started/locally/
- Para Ollama: instalar e ter um modelo baixado (ex.: `ollama run phi3`)


## 🚀 Instalação

1) Clone o repositório e acesse a pasta

```bash
git clone <URL_DO_REPO>
cd ia_llm
```

2) Crie e ative um ambiente virtual

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3) Instale as dependências

```bash
pip install -r requirements.txt
```


## 🔧 Configuração

Defina as variáveis de ambiente conforme os provedores que você usará. Você pode criar um arquivo `.env` ou exportar no shell.

Variáveis comuns suportadas:
- OPENAI_API_KEY
- HUGGINGFACEHUB_API_TOKEN (preferencial) ou HUGGINGFACE_API_KEY
- OLLAMA_HOST (ex.: http://localhost:11434)

Algumas configurações padrão podem ser alteradas no arquivo `config/settings.py`, por exemplo:
- DEFAULT_OPENAI_MODEL
- DEFAULT_HF_MODEL (padrão sugerido: google/flan-t5-base)
- DEFAULT_OLLAMA_MODEL
- DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
- DEFAULT_HF_BACKEND (endpoint ou pipeline)

Exemplo rápido (Windows PowerShell):
```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:HUGGINGFACEHUB_API_TOKEN = "hf_..."
$env:OLLAMA_HOST = "http://localhost:11434"
```


## ▶️ Como Executar

- Rodar um exemplo simples de completação:
```powershell
python src\examples\01_basic_completion.py
```

- Rodar o app Streamlit (chatbot simples):
```powershell
streamlit run apps\01_chatbot_simples.py
```

- Rodar outros exemplos:
```powershell
python src\examples\02_chat_with_history.py
python src\examples\03_streaming_responses.py
python src\examples\04_embeddings.py
python src\examples\05_comparing_models.py
```

- Executar teste rápido do pipeline HF (T5):
```powershell
python tests\teste_rapido.py
```


## 💡 Exemplos de Código

A maneira recomendada é usar o ModelFactory e ModelConfig.

- Completação básica (generate):
```python
from src.models.model_factory import ModelFactory
from src.models.base_model import ModelConfig
from config.settings import settings

config = ModelConfig(
    model_name=settings.DEFAULT_HF_MODEL,  # ou DEFAULT_OPENAI_MODEL, DEFAULT_OLLAMA_MODEL
    temperature=0.7,
    max_tokens=256,
    additional_params={
        # Para HF: escolher backend explicitamente (opcional)
        # 'backend': 'pipeline'  # ou 'endpoint'
    }
)

llm = ModelFactory.create_model('huggingface', config)  # 'openai' | 'ollama'
texto = llm.generate("Explique brevemente o que é um LLM.")
print(texto)
```

- Chat com histórico:
```python
from src.models.model_factory import ModelFactory
from src.models.base_model import ModelConfig
from config.settings import settings

config = ModelConfig(
    model_name=settings.DEFAULT_HF_MODEL,
    temperature=0.7,
    max_tokens=300
)
llm = ModelFactory.create_model('huggingface', config)

messages = [
    {"role": "user", "content": "Olá!"},
    {"role": "assistant", "content": "Olá! Como posso ajudar?"},
    {"role": "user", "content": "Resuma o conceito de embeddings em 2 frases."}
]
resposta = llm.chat(messages, system_message="Você é um assistente útil e responde em português.")
print(resposta)
```

- Streaming de respostas (quando suportado pelo backend/provedor):
```python
from src.models.model_factory import ModelFactory
from src.models.base_model import ModelConfig
from config.settings import settings

llm = ModelFactory.create_model(
    'huggingface',
    ModelConfig(model_name=settings.DEFAULT_HF_MODEL)
)

for chunk in llm.generate_stream("Liste 5 aplicações de LLMs."):
    print(chunk, end="", flush=True)
```


## 🧩 Apps (Streamlit)

Esta pasta contém aplicações interativas construídas com Streamlit para você experimentar os provedores de LLM do projeto.

- 01_chatbot_simples.py
  - O que é: um chatbot básico com histórico de conversa, seleção de provedor (OpenAI, Hugging Face ou Ollama), controle de temperatura e limite de tokens.
  - Recursos:
    - Interface de chat usando st.chat_message
    - Histórico persistente na sessão
    - Mensagem de sistema configurável (definição de comportamento do assistente)
    - Métrica simples de mensagens enviadas
  - Como executar:
    - Windows PowerShell: `streamlit run apps\01_chatbot_simples.py`
    - Linux/Mac: `streamlit run apps/01_chatbot_simples.py`
  - Pré‑requisitos/configuração:
    - Defina as variáveis de ambiente dos provedores que deseja usar (ver seção Configuração)
    - Ajuste os modelos padrão em config/settings.py se necessário
    - Para usar Hugging Face via pipeline local, instale PyTorch (ver Requisitos)
  - Personalização rápida:
    - Altere a mensagem de sistema na sidebar para guiar o tom/respostas do assistente
    - Troque o provider na sidebar para comparar qualidade/latência
    - Ajuste temperature e max_tokens conforme seu caso

Dica: Para criar novos apps, copie este arquivo como base, altere o título, os controles da sidebar e troque a lógica de geração conforme o objetivo do app (por exemplo, usar embeddings, RAG, etc.).

## 🆘 Dicas e Solução de Problemas

- Hugging Face Inference API sem token (401/403/404): o código tenta automaticamente o backend `pipeline` para o mesmo modelo e, se configurado, um modelo de fallback. Configure HUGGINGFACEHUB_API_TOKEN para melhor experiência.
- Pipeline local (transformers) requer PyTorch. No Windows (CPU): `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- Se usar Ollama, verifique se o serviço está ativo e se o modelo foi baixado previamente (`ollama run phi3`).
- Ajuste DEFAULT_HF_BACKEND para `pipeline` se quiser forçar execução local.
- Mensagens de erro incluem links úteis para o repositório do modelo na Hugging Face.


---
Pronto! Explore os exemplos, rode o app e adapte para seus projetos.