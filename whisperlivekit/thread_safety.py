"""
Thread Safety Configuration for WhisperLiveKit

This module provides thread safety configuration and utilities.

Environment Variables:
    WHISPERLIVEKIT_MODEL_LOCK: Enable/disable model locking (default: 1)
        Set to "0" to disable for single-connection deployments

    WHISPERLIVEKIT_LOCK_TIMEOUT: Lock acquisition timeout in seconds (default: 30)

Usage:
    # Enable model locking (default)
    export WHISPERLIVEKIT_MODEL_LOCK=1

    # Disable for single-connection deployment
    export WHISPERLIVEKIT_MODEL_LOCK=0

    # Custom timeout
    export WHISPERLIVEKIT_LOCK_TIMEOUT=60
"""

import os
import logging
import threading

logger = logging.getLogger(__name__)

# Configuration
USE_MODEL_LOCK = os.environ.get("WHISPERLIVEKIT_MODEL_LOCK", "1") == "1"
LOCK_TIMEOUT = float(os.environ.get("WHISPERLIVEKIT_LOCK_TIMEOUT", "30.0"))

# Global model lock
_model_lock = threading.Lock()

# Log configuration on import
if USE_MODEL_LOCK:
    logger.info(f"Model locking ENABLED (timeout: {LOCK_TIMEOUT}s)")
    logger.info("For single-connection deployments, set WHISPERLIVEKIT_MODEL_LOCK=0")
else:
    logger.warning("Model locking DISABLED - only safe for single-connection deployments")


def get_model_lock():
    """Get the global model lock instance"""
    return _model_lock


def acquire_model_lock(timeout=None):
    """
    Acquire model lock with timeout.

    Args:
        timeout: Lock acquisition timeout (default: use LOCK_TIMEOUT)

    Returns:
        bool: True if lock acquired, False on timeout
    """
    if not USE_MODEL_LOCK:
        return True

    timeout = timeout or LOCK_TIMEOUT
    acquired = _model_lock.acquire(timeout=timeout)

    if not acquired:
        logger.error(f"Failed to acquire model lock within {timeout}s")

    return acquired


def release_model_lock():
    """Release model lock"""
    if not USE_MODEL_LOCK:
        return

    try:
        _model_lock.release()
    except RuntimeError:
        # Lock not held - this is fine
        pass


class ModelLockContext:
    """Context manager for model lock"""

    def __init__(self, timeout=None):
        self.timeout = timeout
        self.acquired = False

    def __enter__(self):
        self.acquired = acquire_model_lock(self.timeout)
        return self.acquired

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            release_model_lock()
        return False


# Concurrency recommendations
RECOMMENDED_CONNECTIONS_PER_WORKER = 1 if USE_MODEL_LOCK else 1
RECOMMENDED_WORKERS = 4

def print_deployment_recommendations():
    """Print recommended deployment configuration"""
    print("\n" + "="*60)
    print("WhisperLiveKit Deployment Recommendations")
    print("="*60)

    if USE_MODEL_LOCK:
        print("⚠️  Model locking is ENABLED")
        print("   This serializes inference across connections.")
        print()
        print("Recommended deployment:")
        print(f"  gunicorn -w {RECOMMENDED_WORKERS} \\")
        print("    -k uvicorn.workers.UvicornWorker \\")
        print("    --worker-connections 1 \\")
        print("    whisperlivekit.basic_server:app")
        print()
        print("Expected capacity:")
        print(f"  - {RECOMMENDED_WORKERS} concurrent users (1 per worker)")
        print(f"  - Memory: ~{RECOMMENDED_WORKERS}x model size")
    else:
        print("✅ Model locking is DISABLED")
        print("   ⚠️  ONLY safe for single-connection deployments")
        print()
        print("Recommended deployment:")
        print("  uvicorn whisperlivekit.basic_server:app \\")
        print("    --host 0.0.0.0 --port 8000 \\")
        print("    --workers 1")
        print()
        print("Expected capacity:")
        print("  - 1 concurrent user only")

    print("="*60 + "\n")


if __name__ == "__main__":
    print_deployment_recommendations()
