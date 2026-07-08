"""Observability integrations for ErenAgents.

Provides optional, pluggable tracing/cost backends via LangChain callbacks.
Currently ships a Langfuse helper; any LangChain ``BaseCallbackHandler`` can
also be passed directly through ``Agent(callbacks=[...])``.
"""

from typing import Any


def get_langfuse_handler(**kwargs: Any):
    """Return a Langfuse LangChain ``CallbackHandler``.

    Authentication is read from the environment by the Langfuse SDK
    (``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, ``LANGFUSE_HOST``) or from
    a previously configured Langfuse client. Extra keyword arguments are passed
    through to the handler.

    Requires the ``langfuse`` extra::

        pip install "erenagents[langfuse]"
    """
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Langfuse is not installed. Install it with: "
            'pip install "erenagents[langfuse]"'
        ) from e
    return CallbackHandler(**kwargs)


# Registry of named observability backends usable via ``Agent(observe=...)``.
_OBSERVERS = {
    "langfuse": get_langfuse_handler,
}


def get_observer(name: str, **kwargs: Any):
    """Return a callback handler for a named backend (e.g. ``"langfuse"``)."""
    try:
        factory = _OBSERVERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown observability backend '{name}'. "
            f"Available: {', '.join(_OBSERVERS)}"
        )
    return factory(**kwargs)


__all__ = ["get_langfuse_handler", "get_observer"]
