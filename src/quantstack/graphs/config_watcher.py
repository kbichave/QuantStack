"""Hot-reload config watcher with file-watch (dev) and SIGHUP (prod) support."""
from __future__ import annotations

import logging
import signal
import threading
from pathlib import Path

from quantstack.graphs.config import AgentConfig, load_agent_configs, load_blocked_tools

logger = logging.getLogger(__name__)


class ConfigWatcher:
    """Loads agent configs and supports hot-reload.

    Two reload mechanisms:
    - Dev mode (watch=True): uses watchdog to monitor YAML files for changes.
    - Prod mode (SIGHUP): registers a signal handler that sets a reload flag.

    In both modes, the actual config swap happens only when the caller invokes
    apply_pending_reload(). This ensures reload occurs at cycle boundaries.
    """

    def __init__(self, yaml_path: Path, *, watch: bool = False) -> None:
        self._yaml_path = yaml_path
        self._lock = threading.Lock()
        self._configs = load_agent_configs(yaml_path)
        self._blocked_tools = load_blocked_tools(yaml_path)
        self._pending_configs: dict[str, AgentConfig] | None = None
        self._pending_blocked_tools: frozenset[str] | None = None
        self._observer = None

        if watch:
            self._start_file_watcher()

    def get_config(self, agent_name: str) -> AgentConfig:
        """Return the current config for the named agent. Thread-safe."""
        with self._lock:
            return self._configs[agent_name]

    def get_all_configs(self) -> dict[str, AgentConfig]:
        """Return all current configs. Thread-safe."""
        with self._lock:
            return dict(self._configs)

    def get_blocked_tools(self) -> frozenset[str]:
        """Return the current graph-level blocked tools set. Thread-safe."""
        with self._lock:
            return self._blocked_tools

    def apply_pending_reload(self) -> bool:
        """If a reload is pending, atomically swap configs.

        Returns True if configs were reloaded, False if no change pending.
        Called by the runner at the start of each cycle.
        """
        with self._lock:
            if self._pending_configs is None:
                return False
            self._configs = self._pending_configs
            self._pending_configs = None
            if self._pending_blocked_tools is not None:
                self._blocked_tools = self._pending_blocked_tools
                self._pending_blocked_tools = None
            logger.info("Agent configs reloaded from %s", self._yaml_path)
            return True

    def _stage_reload(self) -> None:
        """Parse YAML and stage new configs (does NOT apply them yet)."""
        try:
            new_configs = load_agent_configs(self._yaml_path)
            new_blocked = load_blocked_tools(self._yaml_path)
            with self._lock:
                self._pending_configs = new_configs
                self._pending_blocked_tools = new_blocked
            logger.debug("Staged config reload from %s", self._yaml_path)
        except Exception:
            logger.exception("Failed to reload configs from %s — keeping current config", self._yaml_path)

    def register_sighup_handler(self) -> None:
        """Register SIGHUP handler for prod-mode reload."""
        try:
            signal.signal(signal.SIGHUP, self._sighup_handler)
        except (AttributeError, OSError):
            logger.warning("SIGHUP not available on this platform — use file-watch mode instead")

    def _sighup_handler(self, signum, frame) -> None:
        self._stage_reload()

    def _start_file_watcher(self) -> None:
        """Start watchdog file observer for dev-mode hot-reload."""
        try:
            from watchdog.events import FileModifiedEvent, FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            logger.warning("watchdog not installed — file-watch mode unavailable")
            return

        watcher = self
        yaml_name = self._yaml_path.name

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if isinstance(event, FileModifiedEvent) and event.src_path.endswith(yaml_name):
                    watcher._stage_reload()

        self._observer = Observer()
        self._observer.schedule(_Handler(), str(self._yaml_path.parent), recursive=False)
        self._observer.daemon = True
        self._observer.start()
        logger.info("File watcher started for %s", self._yaml_path)

    def stop(self) -> None:
        """Stop file watcher if running."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
