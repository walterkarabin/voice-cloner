"""
Helper module to load services with hyphenated directory names.

Python doesn't allow importing from directories with hyphens, so this module
provides a way to load service clients using importlib.
"""

import importlib.util
import sys
from pathlib import Path

# Ensure workspace is in path
if '/workspace' not in sys.path:
    sys.path.insert(0, '/workspace')


def load_service_client(service_name: str, client_class_name: str):
    """
    Load a service client from a hyphenated service directory.

    Args:
        service_name: Service directory name (with hyphens, e.g., 'embed-loader')
        client_class_name: Name of the client class to load (e.g., 'EmbedLoaderClient')

    Returns:
        The client class from the service module

    Example:
        EmbedLoaderClient = load_service_client('embed-loader', 'EmbedLoaderClient')
        client = EmbedLoaderClient()
    """
    service_path = Path('/workspace/services') / service_name / 'client.py'

    if not service_path.exists():
        raise FileNotFoundError(f"Service client not found: {service_path}")

    # Create a unique module name by replacing hyphens with underscores
    module_name = f"{service_name.replace('-', '_')}_client"

    spec = importlib.util.spec_from_file_location(module_name, str(service_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {service_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, client_class_name):
        raise AttributeError(
            f"Client class '{client_class_name}' not found in {service_path}"
        )

    return getattr(module, client_class_name)


# Pre-load common clients for convenience
try:
    EmbedLoaderClient = load_service_client('embed-loader', 'EmbedLoaderClient')
except (FileNotFoundError, ImportError, AttributeError):
    EmbedLoaderClient = None

try:
    WhisperSTTClient = load_service_client('stt-whisper', 'WhisperSTTClient')
except (FileNotFoundError, ImportError, AttributeError):
    WhisperSTTClient = None

try:
    LLMRewriterClient = load_service_client('rewriter-llm', 'LLMRewriterClient')
except (FileNotFoundError, ImportError, AttributeError):
    LLMRewriterClient = None


__all__ = [
    'load_service_client',
    'EmbedLoaderClient',
    'WhisperSTTClient',
    'LLMRewriterClient',
]
