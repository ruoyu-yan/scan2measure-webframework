"""
Plugin registry: discovers, loads, and manages installed plugins.
"""

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from plugins.base import Plugin, PluginMeta, PluginType

# Default directories
PLUGINS_DIR = Path(__file__).parent
INSTALLED_DIR = PLUGINS_DIR / "installed"
REGISTRY_FILE = PLUGINS_DIR / "registry.json"


class PluginRegistry:
    """
    Manages plugin discovery, installation tracking, and loading.
    """

    def __init__(self, registry_path: Optional[Path] = None,
                 installed_dir: Optional[Path] = None):
        self.registry_path = registry_path or REGISTRY_FILE
        self.installed_dir = installed_dir or INSTALLED_DIR
        self._plugins: Dict[str, Plugin] = {}
        self._metadata: Dict[str, dict] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load the persisted registry of installed plugins."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                self._metadata = json.load(f)

    def _save_registry(self) -> None:
        """Persist the registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def register(self, plugin: Plugin) -> None:
        """Register a plugin instance."""
        meta = plugin.get_meta()
        self._plugins[meta.name] = plugin
        self._metadata[meta.name] = {
            **meta.to_dict(),
            "enabled": True,
        }
        self._save_registry()

    def unregister(self, name: str) -> bool:
        """Unregister a plugin by name."""
        if name in self._plugins:
            self._plugins[name].on_uninstall()
            del self._plugins[name]
        if name in self._metadata:
            del self._metadata[name]
            self._save_registry()
            return True
        return False

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a loaded plugin by name."""
        return self._plugins.get(name)

    def list_installed(self) -> List[dict]:
        """List all installed plugin metadata."""
        return list(self._metadata.values())

    def list_by_type(self, plugin_type: PluginType) -> List[dict]:
        """List installed plugins filtered by type."""
        return [
            m for m in self._metadata.values()
            if m.get("plugin_type") == plugin_type.value
        ]

    def is_installed(self, name: str) -> bool:
        """Check if a plugin is installed."""
        return name in self._metadata

    def enable(self, name: str) -> bool:
        """Enable a plugin."""
        if name in self._metadata:
            self._metadata[name]["enabled"] = True
            self._save_registry()
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a plugin without uninstalling."""
        if name in self._metadata:
            self._metadata[name]["enabled"] = False
            self._save_registry()
            return True
        return False

    def discover_plugins(self, search_dir: Optional[Path] = None) -> List[str]:
        """
        Discover plugin modules in a directory.
        Returns list of discovered module names.
        """
        target_dir = search_dir or self.installed_dir
        if not target_dir.exists():
            return []

        discovered = []
        for item in target_dir.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                discovered.append(item.name)
            elif item.is_file() and item.suffix == ".py" and item.name != "__init__.py":
                discovered.append(item.stem)
        return discovered

    def load_plugin(self, module_name: str, search_dir: Optional[Path] = None) -> Optional[Plugin]:
        """
        Dynamically load a plugin from a module.
        The module must contain a class that subclasses Plugin.
        """
        target_dir = search_dir or self.installed_dir
        if str(target_dir) not in sys.path:
            sys.path.insert(0, str(target_dir))

        try:
            module = importlib.import_module(module_name)
            # Find Plugin subclasses in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type)
                        and issubclass(attr, Plugin)
                        and attr is not Plugin):
                    instance = attr()
                    self.register(instance)
                    instance.on_install()
                    return instance
        except Exception as e:
            print(f"Error loading plugin '{module_name}': {e}")
        return None

    def load_all(self, search_dir: Optional[Path] = None) -> int:
        """Load all discovered plugins. Returns count of loaded plugins."""
        modules = self.discover_plugins(search_dir)
        loaded = 0
        for mod in modules:
            if self.load_plugin(mod, search_dir):
                loaded += 1
        return loaded
