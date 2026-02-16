"""agos — Agentic OS. An intelligence layer, not a library."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("agos")
except PackageNotFoundError:
    __version__ = "0.1.0"  # fallback for development
