"""Universe mapping: loads sleeve definitions from config/universe.yaml.

Provides lookup helpers used by the discovery and loader modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_UNIVERSE_PATH = Path(__file__).parents[2] / "config" / "universe.yaml"


def load_universe(path: Optional[Path] = None) -> Dict:
    """Load and return the raw universe YAML as a dict.

    Args:
        path: Override path to universe.yaml. Uses default location if None.
    """
    config_path = path or _UNIVERSE_PATH
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def get_sleeve_list(path: Optional[Path] = None) -> List[Dict]:
    """Return the list of sleeve definition dicts from universe.yaml."""
    return load_universe(path).get("sleeves", [])


def get_sleeve_by_id(sleeve_id: str, path: Optional[Path] = None) -> Optional[Dict]:
    """Return the sleeve definition for *sleeve_id*, or None if not found."""
    for sleeve in get_sleeve_list(path):
        if sleeve["id"] == sleeve_id:
            return sleeve
    logger.warning("Sleeve id '%s' not found in universe config.", sleeve_id)
    return None


def get_all_rics(path: Optional[Path] = None) -> Dict[str, str]:
    """Return {sleeve_id: primary_ric} for all sleeves."""
    return {s["id"]: s["primary_ric"] for s in get_sleeve_list(path)}


def get_fallback_rics(path: Optional[Path] = None) -> Dict[str, str]:
    """Return {sleeve_id: fallback_ric} for all sleeves."""
    return {s["id"]: s["fallback_ric"] for s in get_sleeve_list(path)}


def get_asset_class_map(path: Optional[Path] = None) -> Dict[str, str]:
    """Return {sleeve_id: asset_class} for all sleeves."""
    return {s["id"]: s["asset_class"] for s in get_sleeve_list(path)}


def get_currency_map(path: Optional[Path] = None) -> Dict[str, str]:
    """Return {sleeve_id: currency} for all sleeves."""
    return {s["id"]: s["currency"] for s in get_sleeve_list(path)}


def get_name_map(path: Optional[Path] = None) -> Dict[str, str]:
    """Return {sleeve_id: human-readable name} for all sleeves."""
    return {s["id"]: s["name"] for s in get_sleeve_list(path)}


def get_base_currency(path: Optional[Path] = None) -> str:
    """Return the base currency defined in universe.yaml."""
    return load_universe(path).get("base_currency", "USD")


def build_ric_to_sleeve_map(path: Optional[Path] = None) -> Dict[str, str]:
    """Return {ric: sleeve_id} for both primary and fallback RICs."""
    result = {}
    for sleeve in get_sleeve_list(path):
        result[sleeve["primary_ric"]] = sleeve["id"]
        result[sleeve["fallback_ric"]] = sleeve["id"]
    return result
