"""
kg-service entry point

Start the FastAPI server
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings, LOGGING_CONFIG
import uvicorn

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def main():
    """Start the kg-service API server"""

    logger.info("=" * 70)
    logger.info(f"Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")
    logger.info("=" * 70)
    logger.info(f"Host: {settings.API_HOST}")
    logger.info(f"Port: {settings.API_PORT}")
    logger.info(f"Debug: {settings.DEBUG}")
    logger.info(f"Log Level: {settings.LOG_LEVEL}")
    logger.info("=" * 70)

    try:
        uvicorn.run(
            "api.server:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=settings.DEBUG,
            log_level=settings.LOG_LEVEL.lower(),
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("\n✓ Shutting down gracefully...")
    except Exception as e:
        logger.error(f"✗ Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
