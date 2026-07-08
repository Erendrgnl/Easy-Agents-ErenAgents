from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chat_models import init_chat_model


class ModelRouter:
    """Resolve a ``provider:model`` string into a LangChain chat model.

    Supports the major hosted providers plus OpenAI-compatible endpoints
    (vLLM, local servers) via ``base_url``/``api_key``.
    """

    SUPPORTED_PROVIDERS = ["openai", "anthropic", "google", "gemini", "vllm"]

    # Map user-facing provider names to the ``model_provider`` value expected
    # by ``init_chat_model``.
    PROVIDER_ALIASES = {
        "google": "google_genai",
        "gemini": "google_genai",
        # OpenAI-compatible servers (vLLM, LM Studio, etc.)
        "vllm": "openai",
        "openai-compatible": "openai",
        "custom": "openai",
    }

    # Providers that talk the OpenAI API but are reached through a custom URL.
    OPENAI_COMPATIBLE = {"vllm", "openai-compatible", "custom"}

    @staticmethod
    def infer_provider(model_name: str) -> Optional[str]:
        """Best-effort provider guess from a bare model name."""
        name = model_name.lower()
        if name.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
            return "openai"
        if name.startswith("claude"):
            return "anthropic"
        if name.startswith("gemini"):
            return "google"
        return None

    @staticmethod
    def parse_model_string(model: str) -> tuple[str, str]:
        """Parse ``provider:model`` (or a bare model name) into ``(provider, model_name)``.

        If no provider prefix is given, the provider is inferred from the model
        name when possible (e.g. ``gpt-4o`` -> ``openai``).
        """
        if ":" in model:
            provider, model_name = model.split(":", 1)
            return provider.lower(), model_name

        inferred = ModelRouter.infer_provider(model)
        if inferred:
            return inferred, model

        raise ValueError(
            f"Could not determine provider for model '{model}'. "
            f"Use 'provider:model' format, e.g. 'openai:{model}'. "
            f"Available providers: {', '.join(ModelRouter.SUPPORTED_PROVIDERS)}"
        )

    @staticmethod
    def get_llm(
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> tuple[BaseChatModel, str]:
        """Return ``(chat_model, effective_provider)`` for the given model string.

        Args:
            model: ``provider:model`` string or bare model name.
            base_url: Custom endpoint (e.g. a vLLM server's ``/v1`` URL).
            api_key: API key; for OpenAI-compatible servers it defaults to
                ``"EMPTY"`` when a ``base_url`` is supplied.
            **kwargs: Extra parameters forwarded to ``init_chat_model``.

        Returns:
            Tuple of the chat model and the *effective* provider name used for
            downstream behaviour (e.g. structured-output method selection).
        """
        provider, model_name = ModelRouter.parse_model_string(model)
        init_provider = ModelRouter.PROVIDER_ALIASES.get(provider, provider)

        if base_url is not None:
            kwargs["base_url"] = base_url
        if api_key is not None:
            kwargs["api_key"] = api_key
        elif provider in ModelRouter.OPENAI_COMPATIBLE and "api_key" not in kwargs:
            # vLLM and similar servers accept any non-empty key by default.
            kwargs["api_key"] = "EMPTY"

        try:
            llm = init_chat_model(
                model=model_name,
                model_provider=init_provider,
                **kwargs,
            )
            return llm, init_provider
        except Exception as e:
            raise ValueError(
                f"Model '{model}' could not be initialized. Error: {str(e)}."
            ) from e
