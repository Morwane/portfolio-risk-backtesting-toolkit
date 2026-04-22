"""LSEG session management.

Tries to connect using available libraries in preference order:
  1. lseg.data (newest LSEG Data Library branding)
  2. refinitiv.data (previous branding, same underlying library)
  3. eikon (legacy Refinitiv Eikon Python API)

Each adapter is wrapped in a thin class that exposes a uniform interface
for the rest of the toolkit. If no LSEG library is installed or reachable,
an LSEGUnavailableError is raised.

ENTITLEMENT NOTE:
  - Platform sessions require Eikon/Workspace desktop to be running.
  - Deployed sessions require LSEG_CLIENT_ID / LSEG_CLIENT_SECRET env vars.
  - If credentials are missing, demo mode should be used instead.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LSEGUnavailableError(RuntimeError):
    """Raised when no LSEG library is installed or no session can be opened."""


class _RDSession:
    """Adapter for refinitiv.data / lseg.data libraries."""

    def __init__(self, lib: Any, session_type: str) -> None:
        self._lib = lib
        self._session_type = session_type
        self._session: Optional[Any] = None

    def open(self) -> None:
        try:
            if self._session_type == "deployed":
                self._session = self._lib.session.platform.Definition(
                    signon_control=False,
                    grant=self._lib.session.platform.GrantPassword(
                        username=os.getenv("LSEG_USERNAME", ""),
                        password=os.getenv("LSEG_PASSWORD", ""),
                    ),
                )
            else:
                self._session = self._lib.open_session()
            logger.info("LSEG Data Library session opened (%s).", self._session_type)
        except Exception as exc:
            raise LSEGUnavailableError(
                f"Failed to open LSEG Data Library session: {exc}"
            ) from exc

    def close(self) -> None:
        try:
            self._lib.close_session()
            logger.info("LSEG Data Library session closed.")
        except Exception:
            pass

    @property
    def lib(self) -> Any:
        return self._lib

    @property
    def backend(self) -> str:
        return "rd"


class _EikonSession:
    """Adapter for the legacy Eikon Python API."""

    def __init__(self, ek: Any, app_key: str) -> None:
        self._ek = ek
        self._app_key = app_key

    def open(self) -> None:
        try:
            self._ek.set_app_key(self._app_key)
            logger.info("Eikon session initialised.")
        except Exception as exc:
            raise LSEGUnavailableError(
                f"Failed to initialise Eikon session: {exc}"
            ) from exc

    def close(self) -> None:
        logger.info("Eikon does not require explicit session close.")

    @property
    def lib(self) -> Any:
        return self._ek

    @property
    def backend(self) -> str:
        return "eikon"


def get_session(session_type: str = "platform") -> Any:
    """Return an active LSEG session adapter.

    Tries lseg.data -> refinitiv.data -> eikon in that order.
    Raises :class:`LSEGUnavailableError` if nothing is available.

    Args:
        session_type: ``"platform"`` (desktop) or ``"deployed"`` (server).

    Returns:
        An opened session adapter with ``.lib`` and ``.backend`` properties.
    """
    # ── Attempt 1: lseg.data ────────────────────────────────────────────────
    try:
        import lseg.data as ld  # type: ignore[import]
        session = _RDSession(ld, session_type)
        session.open()
        return session
    except ImportError:
        logger.debug("lseg.data not installed.")
    except LSEGUnavailableError as exc:
        logger.warning("lseg.data available but session failed: %s", exc)

    # ── Attempt 2: refinitiv.data ────────────────────────────────────────────
    try:
        import refinitiv.data as rd  # type: ignore[import]
        session = _RDSession(rd, session_type)
        session.open()
        return session
    except ImportError:
        logger.debug("refinitiv.data not installed.")
    except LSEGUnavailableError as exc:
        logger.warning("refinitiv.data available but session failed: %s", exc)

    # ── Attempt 3: eikon ────────────────────────────────────────────────────
    try:
        import eikon as ek  # type: ignore[import]
        app_key = os.getenv("LSEG_APP_KEY", "")
        if not app_key:
            logger.warning(
                "LSEG_APP_KEY environment variable not set. Eikon may reject requests."
            )
        session = _EikonSession(ek, app_key)
        session.open()
        return session
    except ImportError:
        logger.debug("eikon not installed.")
    except LSEGUnavailableError as exc:
        logger.warning("eikon available but session failed: %s", exc)

    raise LSEGUnavailableError(
        "No LSEG library is installed or reachable.\n"
        "Install one of: lseg-data, refinitiv-data, eikon.\n"
        "Alternatively, enable demo_mode in config/settings.yaml."
    )
