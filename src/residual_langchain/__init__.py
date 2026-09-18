"""residual-langchain: LangChain callback adapter for RESIDUAL attestation."""

from .handler import AsyncResidualCallbackHandler, ResidualCallbackHandler

__version__ = "1.0.0"
__all__ = ["ResidualCallbackHandler", "AsyncResidualCallbackHandler"]
