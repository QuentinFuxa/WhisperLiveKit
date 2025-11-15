import importlib.util
import logging
import platform

logger = logging.getLogger(__name__)


def module_available(module_name):
    """Return True if the given module can be imported."""
    return importlib.util.find_spec(module_name) is not None


def mlx_backend_available(warn_on_missing = False):
    is_macos = platform.system() == "Darwin"
    is_arm = platform.machine() == "arm64"
    available = (
        is_macos
        and is_arm
        and module_available("mlx_whisper")
    )
    if not available and warn_on_missing and is_macos and is_arm:
        logger.warning(
            "=" * 50
            + "\nMLX Whisper not found but you are on Apple Silicon. "
              "Consider installing mlx-whisper for better performance: "
              "`pip install mlx-whisper`\n"
            + "=" * 50
        )
    return available


def faster_backend_available(warn_on_missing = False):
    available = module_available("faster_whisper")
    if not available and warn_on_missing and platform.system() != "Darwin":
        logger.warning(
            "=" * 50
            + "\nFaster-Whisper not found. Consider installing faster-whisper "
              "for better performance: `pip install faster-whisper`\n"
            + "=" * 50
        )
    return available
