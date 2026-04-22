#!/usr/bin/env python
"""Build all demo outputs using synthetic data.

Generates realistic synthetic prices, runs the full pipeline,
and produces all charts and tables for the README and docs/.

Run once after cloning the repo to populate outputs/:
    python scripts/build_demo_outputs.py

No LSEG access required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils.logging_utils import configure_root_logger, get_logger

configure_root_logger("INFO")
logger = get_logger(__name__)


def main():
    logger.info("Building demo outputs (synthetic data, no LSEG required) ...")

    # Step 1: Generate synthetic price data
    from data.sample.generate_demo_data import generate_and_save
    generate_and_save()

    # Step 2: Run the full backtest pipeline in demo mode
    import subprocess
    result = subprocess.run(
        [
            sys.executable, "scripts/run_backtest.py",
            "--demo",
            "--portfolio", "all",
            "--start", "2007-01-01",
            "--log-level", "INFO",
        ],
        cwd=Path(__file__).parents[1],
    )

    if result.returncode != 0:
        logger.error("Demo backtest run failed. Check logs above.")
        sys.exit(1)

    logger.info("Demo outputs built successfully.")
    logger.info("  Synthetic data: data/sample/demo_prices.parquet")
    logger.info("  Charts: outputs/charts/")
    logger.info("  Tables: outputs/tables/")
    logger.info("  README images: docs/images/")


if __name__ == "__main__":
    main()
