"""
Plugin Marketplace: browse, search, install, and manage plugins
for the Scan2Measure pipeline.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from plugins.base import Plugin, PluginMeta, PluginType
from plugins.registry import PluginRegistry

# Built-in catalog of available plugins
MARKETPLACE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "ransac_feature_extractor",
        "version": "1.0.0",
        "description": "RANSAC-based robust feature extraction for noisy point clouds. "
                       "Improves line detection in cluttered environments.",
        "author": "Scan2Measure Team",
        "plugin_type": "feature_extraction",
        "dependencies": ["numpy", "scipy"],
        "tags": ["ransac", "robust", "point-cloud", "feature-extraction"],
        "homepage": "",
        "license": "MIT",
        "source": "bundled",
    },
    {
        "name": "superpoint_feature_matcher",
        "version": "1.0.0",
        "description": "SuperPoint + SuperGlue neural feature matching for improved "
                       "camera pose estimation accuracy.",
        "author": "Scan2Measure Team",
        "plugin_type": "pose_estimation",
        "dependencies": ["torch", "numpy"],
        "tags": ["superpoint", "superglue", "deep-learning", "pose"],
        "homepage": "",
        "license": "MIT",
        "source": "bundled",
    },
    {
        "name": "open3d_visualizer",
        "version": "1.0.0",
        "description": "Interactive 3D visualization of point clouds, wireframes, "
                       "and camera poses using Open3D.",
        "author": "Scan2Measure Team",
        "plugin_type": "visualization",
        "dependencies": ["open3d", "numpy"],
        "tags": ["3d", "visualization", "interactive", "open3d"],
        "homepage": "",
        "license": "MIT",
        "source": "bundled",
    },
    {
        "name": "ply_las_converter",
        "version": "1.0.0",
        "description": "Convert between PLY, LAS, LAZ, XYZ, and OBJ point cloud formats.",
        "author": "Scan2Measure Team",
        "plugin_type": "data_io",
        "dependencies": ["numpy"],
        "tags": ["converter", "ply", "las", "xyz", "obj", "io"],
        "homepage": "",
        "license": "MIT",
        "source": "bundled",
    },
    {
        "name": "noise_filter",
        "version": "1.0.0",
        "description": "Statistical outlier removal and voxel downsampling for "
                       "point cloud preprocessing.",
        "author": "Scan2Measure Team",
        "plugin_type": "preprocessing",
        "dependencies": ["numpy", "scipy"],
        "tags": ["filter", "denoise", "outlier", "downsample", "preprocessing"],
        "homepage": "",
        "license": "MIT",
        "source": "bundled",
    },
    {
        "name": "floorplan_exporter",
        "version": "1.0.0",
        "description": "Export reconstructed floorplans to DXF, SVG, and PDF formats "
                       "for CAD integration.",
        "author": "Scan2Measure Team",
        "plugin_type": "postprocessing",
        "dependencies": ["numpy", "shapely"],
        "tags": ["floorplan", "export", "dxf", "svg", "cad"],
        "homepage": "",
        "license": "MIT",
        "source": "bundled",
    },
    {
        "name": "plane_segmentation",
        "version": "1.0.0",
        "description": "Deep learning-based plane segmentation for structured "
                       "environments (walls, floors, ceilings).",
        "author": "Scan2Measure Team",
        "plugin_type": "ml_inference",
        "dependencies": ["torch", "numpy"],
        "tags": ["plane", "segmentation", "deep-learning", "walls"],
        "homepage": "",
        "license": "MIT",
        "source": "bundled",
    },
    {
        "name": "icp_refinement",
        "version": "1.0.0",
        "description": "Iterative Closest Point (ICP) algorithm variants for fine-grained "
                       "point cloud registration and pose refinement.",
        "author": "Scan2Measure Team",
        "plugin_type": "geometry_processing",
        "dependencies": ["numpy", "scipy"],
        "tags": ["icp", "registration", "alignment", "refinement"],
        "homepage": "",
        "license": "MIT",
        "source": "bundled",
    },
]


class PluginMarketplace:
    """
    Browse, search, and install plugins from the marketplace catalog.
    """

    def __init__(self, registry: Optional[PluginRegistry] = None):
        self.registry = registry or PluginRegistry()
        self._catalog = list(MARKETPLACE_CATALOG)

    def browse(self, plugin_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Browse available plugins, optionally filtered by type.

        Args:
            plugin_type: Filter by PluginType value (e.g. "feature_extraction").

        Returns:
            List of plugin catalog entries.
        """
        if plugin_type:
            return [p for p in self._catalog if p["plugin_type"] == plugin_type]
        return list(self._catalog)

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search plugins by name, description, or tags.

        Args:
            query: Search term (case-insensitive).

        Returns:
            Matching plugin catalog entries.
        """
        query_lower = query.lower()
        results = []
        for plugin in self._catalog:
            if (query_lower in plugin["name"].lower()
                    or query_lower in plugin["description"].lower()
                    or any(query_lower in tag for tag in plugin.get("tags", []))):
                results.append(plugin)
        return results

    def get_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific plugin."""
        for plugin in self._catalog:
            if plugin["name"] == name:
                info = dict(plugin)
                info["installed"] = self.registry.is_installed(name)
                return info
        return None

    def install(self, name: str) -> bool:
        """
        Install a plugin from the marketplace.

        Args:
            name: Plugin name to install.

        Returns:
            True if installed successfully.
        """
        if self.registry.is_installed(name):
            print(f"Plugin '{name}' is already installed.")
            return False

        catalog_entry = self.get_info(name)
        if not catalog_entry:
            print(f"Plugin '{name}' not found in marketplace.")
            return False

        # Check for bundled plugin source
        bundled_dir = Path(__file__).parent / "available" / name
        installed_dir = Path(__file__).parent / "installed"
        installed_dir.mkdir(parents=True, exist_ok=True)

        if bundled_dir.exists():
            # Copy bundled plugin to installed directory
            dest = installed_dir / name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(bundled_dir, dest)
            # Load the plugin
            plugin = self.registry.load_plugin(name, installed_dir)
            if plugin:
                print(f"Successfully installed '{name}' v{catalog_entry['version']}.")
                return True
            else:
                print(f"Installed files for '{name}' but failed to load plugin class.")
                # Still register as installed from metadata
                meta_entry = {k: v for k, v in catalog_entry.items()
                              if k != "installed" and k != "source"}
                meta_entry["enabled"] = True
                self.registry._metadata[name] = meta_entry
                self.registry._save_registry()
                return True
        else:
            # Register from catalog metadata (stub install)
            meta_entry = {k: v for k, v in catalog_entry.items()
                          if k != "installed" and k != "source"}
            meta_entry["enabled"] = True
            self.registry._metadata[name] = meta_entry
            self.registry._save_registry()
            print(f"Registered '{name}' v{catalog_entry['version']} (source not bundled, "
                  f"metadata registered).")
            return True

    def uninstall(self, name: str) -> bool:
        """
        Uninstall a plugin.

        Args:
            name: Plugin name to uninstall.

        Returns:
            True if uninstalled successfully.
        """
        if not self.registry.is_installed(name):
            print(f"Plugin '{name}' is not installed.")
            return False

        # Remove installed files if present
        installed_path = Path(__file__).parent / "installed" / name
        if installed_path.exists():
            shutil.rmtree(installed_path)

        self.registry.unregister(name)
        print(f"Uninstalled '{name}'.")
        return True

    def list_installed(self) -> List[Dict[str, Any]]:
        """List all installed plugins with their status."""
        installed = self.registry.list_installed()
        for entry in installed:
            entry["installed"] = True
        return installed

    def get_categories(self) -> List[str]:
        """Get all available plugin categories."""
        return [t.value for t in PluginType]

    def print_catalog(self, plugin_type: Optional[str] = None) -> None:
        """Pretty-print the marketplace catalog."""
        plugins = self.browse(plugin_type)
        if not plugins:
            print("No plugins found.")
            return

        if plugin_type:
            print(f"\n  Marketplace — {plugin_type.replace('_', ' ').title()} Plugins")
        else:
            print("\n  Marketplace — All Plugins")
        print("  " + "=" * 60)

        for p in plugins:
            installed = self.registry.is_installed(p["name"])
            status = " [installed]" if installed else ""
            print(f"\n  {p['name']} v{p['version']}{status}")
            print(f"    {p['description']}")
            print(f"    Type: {p['plugin_type']} | Author: {p['author']}")
            if p.get("tags"):
                print(f"    Tags: {', '.join(p['tags'])}")

        print()

    def print_installed(self) -> None:
        """Pretty-print installed plugins."""
        installed = self.list_installed()
        if not installed:
            print("\n  No plugins installed. Use 'marketplace add <name>' to install one.")
            return

        print("\n  Installed Plugins")
        print("  " + "=" * 60)

        for p in installed:
            enabled = p.get("enabled", True)
            status = "enabled" if enabled else "disabled"
            print(f"\n  {p['name']} v{p.get('version', '?')} [{status}]")
            print(f"    {p.get('description', '')}")
            print(f"    Type: {p.get('plugin_type', '?')}")

        print()
