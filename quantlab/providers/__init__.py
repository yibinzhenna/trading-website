"""Data providers. The engine depends on this interface, never on a vendor."""

from .base import DataProvider, ProviderError
from .local import LocalProvider

__all__ = ["DataProvider", "ProviderError", "LocalProvider", "get_provider"]


def get_provider(name="local", **kwargs):
    """Construct a provider by name.

    Imported lazily so a missing optional dependency or absent API key only
    matters if you actually ask for that provider.
    """
    key = (name or "local").lower()
    if key == "local":
        return LocalProvider(**kwargs)
    if key in ("alphavantage", "av"):
        from .alphavantage import AlphaVantageProvider
        return AlphaVantageProvider(**kwargs)
    raise ProviderError(f"Unknown provider {name!r}")
