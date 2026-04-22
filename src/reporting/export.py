"""Output export utilities.

Handles saving DataFrames to CSV and figures to PNG, maintaining
consistent output directory structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_TABLES_DIR = Path("outputs/tables")
_CHARTS_DIR = Path("outputs/charts")
_DOCS_IMAGES_DIR = Path("docs/images")


def save_table(
    df: pd.DataFrame,
    filename: str,
    output_dir: Optional[Path] = None,
    index: bool = True,
) -> Path:
    """Save a DataFrame to CSV.

    Args:
        df: DataFrame to save.
        filename: File name (with or without .csv extension).
        output_dir: Target directory. Defaults to outputs/tables/.
        index: Whether to write the row index.

    Returns:
        Path to the saved file.
    """
    if not filename.endswith(".csv"):
        filename += ".csv"
    out = (output_dir or _TABLES_DIR) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=index)
    logger.info("Table saved: %s", out)
    return out


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Optional[Path] = None,
    dpi: int = 150,
    also_save_to_docs: bool = False,
) -> Path:
    """Save a matplotlib Figure to PNG.

    Args:
        fig: Matplotlib Figure to save.
        filename: File name (with or without .png extension).
        output_dir: Target directory. Defaults to outputs/charts/.
        dpi: Output DPI.
        also_save_to_docs: If True, also copies to docs/images/ for README.

    Returns:
        Path to the saved file.
    """
    if not filename.endswith(".png"):
        filename += ".png"
    out = (output_dir or _CHARTS_DIR) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    logger.info("Chart saved: %s", out)

    if also_save_to_docs:
        docs_path = _DOCS_IMAGES_DIR / filename
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(docs_path, dpi=dpi, bbox_inches="tight")
        logger.info("Chart copied to docs: %s", docs_path)

    plt.close(fig)
    return out


def export_all_tables(
    tables: Dict[str, pd.DataFrame],
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Save a dict of DataFrames to CSV files.

    Args:
        tables: Dict of {filename_stem: DataFrame}.
        output_dir: Target directory.

    Returns:
        Dict of {name: saved_path}.
    """
    paths = {}
    for name, df in tables.items():
        paths[name] = save_table(df, name, output_dir)
    return paths


def export_all_figures(
    figures: Dict[str, plt.Figure],
    output_dir: Optional[Path] = None,
    docs_figures: Optional[list] = None,
    dpi: int = 150,
) -> Dict[str, Path]:
    """Save a dict of Figures to PNG files.

    Args:
        figures: Dict of {filename_stem: Figure}.
        output_dir: Target directory.
        docs_figures: List of figure names to also copy to docs/images/.
        dpi: Output DPI.

    Returns:
        Dict of {name: saved_path}.
    """
    docs_set = set(docs_figures or [])
    paths = {}
    for name, fig in figures.items():
        paths[name] = save_figure(
            fig, name, output_dir, dpi=dpi,
            also_save_to_docs=(name in docs_set),
        )
    return paths
