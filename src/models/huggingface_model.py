"""Implementação do modelo HuggingFace."""
from typing import List, Dict, Any, Generator, Optional
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from huggingface_hub import HfApi
# HfHubHTTPError is not exported at top-level in some huggingface_hub versions (e.g., 0.23.x)
# Try common locations, fallback to generic Exception so our except blocks still work.
try:
    from huggingface_hub.errors import HfHubHTTPError  # available in many versions incl. 0.23.x
except Exception:
    try:
        from huggingface_hub.utils import HfHubHTTPError  # older versions
    except Exception:  # final fallback
        HfHubHTTPError = Exception  # type: ignore
from config.settings import settings
from src.utils.validators import validate_api_key  # mantido para validação opcional
from .base_model import BaseLLMModel, ModelConfig


class HuggingFaceModel(BaseLLMModel):
    """Implementação para modelos HuggingFace."""

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Inicializa o modelo HuggingFace.

        Args:
            config: Configuração do modelo (usa padrão se não fornecida)
        """
        if config is None:
            config = ModelConfig(
                model_name=settings.DEFAULT_HF_MODEL,
                temperature=settings.DEFAULT_TEMPERATURE,
                max_tokens=settings.DEFAULT_MAX_TOKENS
            )

        super().__init__(config)

        # Controla o repositório ativo, backend e fallback automático (inicialize antes de qualquer uso)
        self._active_repo_id = self.config.model_name
        self._active_backend = (self.config.additional_params or {}).get('backend', getattr(settings, 'DEFAULT_HF_BACKEND', 'endpoint'))
        self._used_fallback = False

        # Token não é estritamente obrigatório para modelos públicos e para backend 'pipeline'.
        # Se nenhum token estiver configurado, apenas avisamos quando o backend for 'endpoint'.
        _hf_token = settings.HUGGINGFACEHUB_API_TOKEN or settings.HUGGINGFACE_API_KEY
        if (self._active_backend or 'endpoint').lower() == 'endpoint':
            if not (_hf_token and validate_api_key(_hf_token)):
                self.logger.warning(
                    "Nenhum token válido encontrado em HUGGINGFACEHUB_API_TOKEN/HUGGINGFACE_API_KEY. "
                    "Modelos públicos podem funcionar, mas é recomendável configurar um token."
                )

        # Define o template de chat (permite override via additional_params)
        self._chat_template = (
            (self.config.additional_params or {}).get('chat_template')
            or getattr(settings, 'DEFAULT_HF_CHAT_TEMPLATE', 'phi3')
        )

        self._initialize_client()

    def _initialize_client(self):
        """Inicializa o cliente HuggingFace com suporte a dois backends:
        - endpoint: Hosted Inference API (HuggingFaceEndpoint)
        - pipeline: transformers.pipeline local (HuggingFacePipeline)
        Tenta automaticamente 'pipeline' para o mesmo modelo caso a Inference API retorne 404/401/403.
        """
        hf_token = settings.HUGGINGFACEHUB_API_TOKEN or settings.HUGGINGFACE_API_KEY
        target_repo = self.config.model_name
        preferred_backend = (self._active_backend or 'endpoint').lower()

        def _finalize(client, repo, backend):
            self._client = client
            self._active_repo_id = repo
            self._active_backend = backend
            self.logger.info(f"Cliente HuggingFace inicializado: {repo} (backend={backend})")

        # Se o usuário já pediu explicitamente pipeline, tente direto
        if preferred_backend == 'pipeline':
            try:
                client = self._build_pipeline_client(target_repo)
                _finalize(client, target_repo, 'pipeline')
                return
            except Exception as e:
                self.logger.warning(f"Falha ao inicializar pipeline para '{target_repo}': {e}. Tentando endpoint.")
                preferred_backend = 'endpoint'

        # Caminho padrão: endpoint com pré-checagem
        use_fallback = False
        try:
            api = HfApi(token=hf_token)
            api.model_info(target_repo, token=hf_token)
        except HfHubHTTPError as e:
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status in (401, 403, 404):
                # Tenta pipeline para o MESMO modelo antes de mudar para fallback
                try:
                    client = self._build_pipeline_client(target_repo)
                    self.logger.warning(
                        f"Modelo '{target_repo}' indisponível na Inference API ({status}). Usando backend pipeline local." 
                    )
                    _finalize(client, target_repo, 'pipeline')
                    return
                except Exception as e2:
                    self.logger.warning(f"Falha no pipeline para '{target_repo}': {e2}. Avaliando fallback...")
                # Se chegou aqui, tenta fallback (endpoint → pipeline)
                fb = getattr(settings, 'DEFAULT_HF_FALLBACK_MODEL', None)
                if fb and fb != target_repo:
                    # Primeiro endpoint no fallback
                    try:
                        client = self._build_endpoint_client(fb, hf_token)
                        self.logger.warning(
                            f"Modelo '{target_repo}' indisponível ({status}). Usando fallback endpoint '{fb}'."
                        )
                        use_fallback = True
                        _finalize(client, fb, 'endpoint')
                        return
                    except Exception as e_fb_ep:
                        self.logger.warning(f"Falha no fallback endpoint '{fb}': {e_fb_ep}. Tentando pipeline...")
                    # Depois pipeline no fallback
                    try:
                        client = self._build_pipeline_client(fb)
                        self.logger.warning(
                            f"Modelo '{target_repo}' indisponível ({status}). Usando fallback pipeline '{fb}'."
                        )
                        use_fallback = True
                        _finalize(client, fb, 'pipeline')
                        return
                    except Exception as e_fb_pi:
                        self.logger.error(f"Falha também no fallback pipeline '{fb}': {e_fb_pi}")
                # Sem fallback viável
                base_hint = f"Verifique se o repositório existe e está acessível: https://huggingface.co/{target_repo}"
                hint = (
                    "Falha ao inicializar modelo via Inference API e pipeline.\n"
                    f"- Modelo: {target_repo}\n"
                    f"- {base_hint}\n"
                    f"- Pesquise: {self._build_hf_search_url(target_repo)}\n"
                    "- Em modelos públicos, a Inference API pode exigir autenticação.\n"
                )
                raise ValueError(hint) from e
            else:
                self.logger.error(f"Erro ao consultar modelo no HF Hub: {e}")
                raise
        except Exception as e:
            self.logger.warning(f"Pré-checagem falhou: {e}. Tentando inicializar endpoint diretamente.")
        # Se chegou aqui, tenta endpoint direto
        try:
            client = self._build_endpoint_client(target_repo, hf_token)
            _finalize(client, target_repo, 'endpoint')
            self._used_fallback = use_fallback
            return
        except Exception as e:
            # Como último recurso, tente pipeline direto
            self.logger.warning(f"Falha no endpoint '{target_repo}': {e}. Tentando pipeline como último recurso.")
            client = self._build_pipeline_client(target_repo)
            _finalize(client, target_repo, 'pipeline')
            self._used_fallback = use_fallback
            return

    def _build_endpoint_client(self, repo_id: str, hf_token: Optional[str]):
        """Cria um cliente baseado na Hosted Inference API (HuggingFaceEndpoint)."""
        client_kwargs = dict(self.config.additional_params or {})
        if (self._chat_template or '').lower() == 'phi3' and 'stop_sequences' not in client_kwargs:
            client_kwargs['stop_sequences'] = ["<|end|>"]
        # Seleciona a task adequada para o modelo (ex.: T5 → text2text-generation)
        task = self._detect_pipeline_task(repo_id)
        return HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_tokens,
            return_full_text=False,
            task=task,
            huggingfacehub_api_token=hf_token,
            **client_kwargs
        )

    def _detect_pipeline_task(self, repo_id: str) -> str:
        """Heurística simples para escolher a task do transformers.pipeline.
        Pode ser sobrescrita por additional_params['task'].
        """
        task_override = (self.config.additional_params or {}).get('task')
        if isinstance(task_override, str) and task_override:
            return task_override
        rid = (repo_id or '').lower()
        if 't5' in rid:
            return 'text2text-generation'
        # fallback genérico
        return 'text-generation'

    def _build_pipeline_client(self, repo_id: str):
        """Cria um cliente local usando transformers.pipeline + HuggingFacePipeline.
        Faz import lazy de transformers para evitar carregar torch quando não necessário.
        """
        try:
            from transformers import pipeline as _hf_pipeline  # lazy import
        except Exception as imp_err:
            # Orientação clara para instalação do PyTorch (CPU) no Windows
            tip = (
                "Para usar o backend 'pipeline' é necessário ter o PyTorch instalado.\n"
                "No Windows (CPU), tente instalar com:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "Ou consulte: https://pytorch.org/get-started/locally/"
            )
            self.logger.error(f"Falha ao importar transformers: {imp_err}. {tip}")
            raise RuntimeError(tip) from imp_err

        task = self._detect_pipeline_task(repo_id)
        try:
            rid_lower = (repo_id or '').lower()
            if 't5' in rid_lower:
                pipe = _hf_pipeline(
                    task,
                    model=repo_id,
                    tokenizer=repo_id,
                    max_length=self.config.max_tokens,
                    torch_dtype='auto'
                )
            else:
                pipe = _hf_pipeline(
                    task,
                    model=repo_id,
                    tokenizer=repo_id,
                    max_length=self.config.max_tokens
                )
        except OSError as os_err:
            # Erros comuns de DLL do torch em Windows
            tip = (
                "Erro ao carregar modelo. Verifique se o modelo existe e sua conexão com a internet.\n"
                "No Windows, se usar pipeline local, instale PyTorch (CPU) com:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "Se já instalado, verifique conflitos de versões e as DLLs do Visual C++ Redistributable."
            )
            self.logger.error(f"Erro ao criar pipeline: {os_err}. {tip}")
            raise RuntimeError(tip) from os_err

        return HuggingFacePipeline(pipeline=pipe)

    def _build_client(self, repo_id: str, hf_token: Optional[str], backend: str = 'endpoint'):
        """Cria um cliente para o repo especificado usando o backend indicado."""
        if (backend or 'endpoint').lower() == 'pipeline':
            return self._build_pipeline_client(repo_id)
        return self._build_endpoint_client(repo_id, hf_token)

    def _sanitize_hf_error_url(self, msg: str) -> str:
        """Normaliza URLs comuns nos erros do HF para evitar caminhos inexistentes.
        - Substitui 'https://huggingface.co/models/' por 'https://huggingface.co/'
        Mantém outros endpoints (ex.: api-inference) inalterados.
        """
        try:
            return msg.replace("https://huggingface.co/models/", "https://huggingface.co/")
        except Exception:
            return msg

    def _build_hf_search_url(self, repo_id: str) -> str:
        """Gera URL de busca na Hugging Face para o modelo informado.
        Ex.: https://huggingface.co/models?sort=trending&search=Phi-3-mini-4k-instruct
        """
        try:
            term = (repo_id or '').split('/')[-1] or repo_id
            return f"https://huggingface.co/models?sort=trending&search={term}"
        except Exception:
            return "https://huggingface.co/models?sort=trending"

    def _format_prompt(self, prompt: str, chat_template: Optional[str] = None) -> str:
        """
        Formata o prompt de acordo com o template do modelo.

        Args:
            prompt: Prompt original
            chat_template: Tipo de template (phi3, llama3, etc). Se None, usa o template configurado.

        Returns:
            Prompt formatado
        """
        tpl = (chat_template or self._chat_template or "phi3").lower()
        templates = {
            'phi3': f"<|system|>You are a helpful AI assistant.<|end|><|user|>{prompt}<|end|><|assistant|>",
            'llama3': f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            'mistral': f"[INST] {prompt} [/INST]"
        }
        return templates.get(tpl, prompt)

    def generate(self, prompt: str, use_template: bool = True, **kwargs) -> str:
        """
        Gera uma resposta para o prompt.

        Args:
            prompt: Texto de entrada
            use_template: Se deve usar template de chat
            **kwargs: Parâmetros adicionais

        Returns:
            Resposta gerada
        """
        try:
            if use_template:
                prompt = self._format_prompt(prompt)

            response = self._client.invoke(prompt)
            self.logger.debug(f"Tipo de resposta (generate): {type(response)} | backend={self._active_backend} repo={self._active_repo_id}")
            # Alguns clientes retornam string direta; outros podem retornar objeto com `.content`
            if isinstance(response, str):
                return response
            if hasattr(response, 'content'):
                return response.content
            # Fallback para dicionários comuns
            if isinstance(response, dict):
                return response.get('generated_text') or response.get('text') or str(response)
            # Alguns wrappers do LangChain retornam objetos com atributo .text
            if hasattr(response, 'text'):
                try:
                    return getattr(response, 'text')
                except Exception:
                    pass
            return str(response)
        except Exception as e:
            msg = str(e)
            # Tenta fallback automático uma vez em 401/403/404
            if any(code in msg for code in [" 404", "404", " 401", "401", " 403", "403"]):
                fb = getattr(settings, 'DEFAULT_HF_FALLBACK_MODEL', None)
                if fb and not self._used_fallback and fb != self._active_repo_id:
                    safe_msg = self._sanitize_hf_error_url(msg)
                    self.logger.warning(
                        f"Falha no modelo '{self._active_repo_id}' ({safe_msg}). Tentando fallback '{fb}'."
                    )
                    hf_token = settings.HUGGINGFACEHUB_API_TOKEN or settings.HUGGINGFACE_API_KEY
                    try:
                        self._client = self._build_client(fb, hf_token)
                        self._active_repo_id = fb
                        self._used_fallback = True
                        # Re-tenta a chamada
                        response = self._client.invoke(prompt)
                        if isinstance(response, str):
                            return response
                        if hasattr(response, 'content'):
                            return response.content
                        if isinstance(response, dict):
                            return response.get('generated_text') or response.get('text') or str(response)
                        return str(response)
                    except Exception as e2:
                        self.logger.error(f"Falha também no fallback '{fb}': {e2}")
                        # Continua para formar dica abaixo
                hint = (
                    f"Falha ao acessar o modelo '{self.config.model_name}' na Hugging Face.\n"
                    f"- Repo: https://huggingface.co/{self.config.model_name}\n"
                    f"- Buscar: {self._build_hf_search_url(self.config.model_name)}\n"
                    "Dica: verifique o token (HUGGINGFACEHUB_API_TOKEN) e tente novamente."
                )
                safe_msg = self._sanitize_hf_error_url(msg)
                self.logger.error(f"Erro ao gerar resposta: {safe_msg} | {hint}")
                raise ValueError(hint) from e
            self.logger.error(f"Erro ao gerar resposta: {e}")
            raise

    def generate_stream(self, prompt: str, use_template: bool = True, **kwargs) -> Generator[str, None, None]:
        """
        Gera uma resposta em streaming.

        Args:
            prompt: Texto de entrada
            use_template: Se deve usar template de chat
            **kwargs: Parâmetros adicionais

        Yields:
            Chunks da resposta
        """
        try:
            if use_template:
                prompt = self._format_prompt(prompt)

            if hasattr(self._client, 'stream'):
                for chunk in self._client.stream(prompt):
                    if isinstance(chunk, str):
                        yield chunk
                    elif hasattr(chunk, 'content'):
                        yield chunk.content
                    else:
                        yield str(chunk)
            else:
                # Backend sem suporte a streaming (ex.: HuggingFacePipeline)
                full = self.generate(prompt, use_template=False)
                yield full
        except Exception as e:
            msg = str(e)
            if any(code in msg for code in [" 404", "404", " 401", "401", " 403", "403"]):
                fb = getattr(settings, 'DEFAULT_HF_FALLBACK_MODEL', None)
                if fb and not self._used_fallback and fb != self._active_repo_id:
                    safe_msg = self._sanitize_hf_error_url(msg)
                    self.logger.warning(
                        f"Falha no modelo '{self._active_repo_id}' durante streaming ({safe_msg}). Tentando fallback '{fb}'."
                    )
                    hf_token = settings.HUGGINGFACEHUB_API_TOKEN or settings.HUGGINGFACE_API_KEY
                    try:
                        self._client = self._build_client(fb, hf_token)
                        self._active_repo_id = fb
                        self._used_fallback = True
                        # Re-tenta o streaming com o fallback
                        for chunk in self._client.stream(prompt):
                            if isinstance(chunk, str):
                                yield chunk
                            elif hasattr(chunk, 'content'):
                                yield chunk.content
                            else:
                                yield str(chunk)
                        return
                    except Exception as e2:
                        self.logger.error(f"Falha também no fallback '{fb}' (streaming): {e2}")
                        # Continua para formar dica abaixo
                hint = (
                    f"Falha ao acessar o modelo '{self.config.model_name}' (stream).\n"
                    f"- Repo: https://huggingface.co/{self.config.model_name}\n"
                    f"- Buscar: {self._build_hf_search_url(self.config.model_name)}\n"
                    "Dica: verifique o token (HUGGINGFACEHUB_API_TOKEN)."
                )
                safe_msg = self._sanitize_hf_error_url(msg)
                self.logger.error(f"Erro no streaming: {safe_msg} | {hint}")
                raise ValueError(hint) from e
            self.logger.error(f"Erro no streaming: {e}")
            raise

    def _format_chat_prompt(self, messages: List[Dict[str, str]], system_message: Optional[str]) -> str:
        """Formata o histórico de chat conforme o template configurado."""
        tpl = (self._chat_template or "plain").lower()
        if tpl == 'phi3':
            chat_prompt = f"<|system|>{system_message or 'You are a helpful AI assistant.'}<|end|>"
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'user':
                    chat_prompt += f"<|user|>{content}<|end|>"
                elif role == 'assistant':
                    chat_prompt += f"<|assistant|>{content}<|end|>"
            chat_prompt += "<|assistant|>"
            return chat_prompt
        else:
            # Formatação simples (compatível com modelos como DialoGPT)
            parts = []
            if system_message:
                parts.append(f"System: {system_message}\n")
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'user':
                    parts.append(f"User: {content}")
                elif role == 'assistant':
                    parts.append(f"Assistant: {content}")
            parts.append("Assistant:")
            return "\n".join(parts)

    def chat(
            self,
            messages: List[Dict[str, str]],
            system_message: Optional[str] = "You are a helpful AI assistant.",
            **kwargs
    ) -> str:
        """
        Interface de chat com histórico.

        Args:
            messages: Lista de mensagens
            system_message: Mensagem de sistema
            **kwargs: Parâmetros adicionais

        Returns:
            Resposta do assistente
        """
        try:
            chat_prompt = self._format_chat_prompt(messages, system_message)
            response = self._client.invoke(chat_prompt)
            self.logger.debug(f"Tipo de resposta (chat): {type(response)} | backend={self._active_backend} repo={self._active_repo_id}")
            if isinstance(response, str):
                return response
            if hasattr(response, 'content'):
                return response.content
            if isinstance(response, dict):
                return response.get('generated_text') or response.get('text') or str(response)
            if hasattr(response, 'text'):
                try:
                    return getattr(response, 'text')
                except Exception:
                    pass
            return str(response)
        except Exception as e:
            msg = str(e)
            if any(code in msg for code in [" 404", "404", " 401", "401", " 403", "403"]):
                fb = getattr(settings, 'DEFAULT_HF_FALLBACK_MODEL', None)
                if fb and not self._used_fallback and fb != self._active_repo_id:
                    safe_msg = self._sanitize_hf_error_url(msg)
                    self.logger.warning(
                        f"Falha no modelo '{self._active_repo_id}' durante chat ({safe_msg}). Tentando fallback '{fb}'."
                    )
                    hf_token = settings.HUGGINGFACEHUB_API_TOKEN or settings.HUGGINGFACE_API_KEY
                    try:
                        self._client = self._build_client(fb, hf_token)
                        self._active_repo_id = fb
                        self._used_fallback = True
                        response = self._client.invoke(chat_prompt)
                        if isinstance(response, str):
                            return response
                        if hasattr(response, 'content'):
                            return response.content
                        if isinstance(response, dict):
                            return response.get('generated_text') or response.get('text') or str(response)
                        return str(response)
                    except Exception as e2:
                        self.logger.error(f"Falha também no fallback '{fb}' (chat): {e2}")
                        # Continua para formar dica abaixo
                hint = (
                    f"Falha ao acessar o modelo '{self.config.model_name}' no chat.\n"
                    f"- Repo: https://huggingface.co/{self.config.model_name}\n"
                    f"- Buscar: {self._build_hf_search_url(self.config.model_name)}\n"
                    "Dica: verifique o token (HUGGINGFACEHUB_API_TOKEN)."
                )
                safe_msg = self._sanitize_hf_error_url(msg)
                self.logger.error(f"Erro no chat: {safe_msg} | {hint}")
                raise ValueError(hint) from e
            self.logger.error(f"Erro no chat: {e}")
            raise