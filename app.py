"""
Nexus Application Entry Point
Run with: python app.py
Or: python app.py --port 8000
"""

import os
import sys
import argparse
import logging

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from nexus.api.routes import app
from nexus.core.config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger("nexus")


def main():
    parser = argparse.ArgumentParser(description="Nexus Relationship Intelligence API")
    parser.add_argument("--host", default=CONFIG.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=CONFIG.port, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", default=CONFIG.debug)
    parser.add_argument("--api-key", help="Anthropic API key", default=os.environ.get("ANTHROPIC_API_KEY", ""))
    args = parser.parse_args()

    if args.api_key:
        CONFIG.anthropic_api_key = args.api_key
        os.environ["ANTHROPIC_API_KEY"] = args.api_key

    logger.info("=" * 60)
    logger.info("  NEXUS Relationship Intelligence Platform")
    logger.info("  Version: 1.0.0")
    logger.info(f"  Starting on http://{args.host}:{args.port}")
    logger.info(f"  Debug mode: {args.debug}")
    logger.info(f"  LLM: {'Enabled (Claude API)' if CONFIG.anthropic_api_key else 'Disabled (template fallback)'}")
    logger.info(f"  DB: {CONFIG.db_path}")
    logger.info("=" * 60)

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
