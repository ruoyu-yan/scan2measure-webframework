"""
Scan2Measure Plugin System

A modular plugin architecture for extending the scan-to-measure pipeline.
Plugins can hook into feature extraction, geometry processing, pose estimation,
ML inference, and visualization stages.
"""

from plugins.base import Plugin, PluginType, PluginMeta
from plugins.registry import PluginRegistry
from plugins.marketplace import PluginMarketplace

__all__ = ["Plugin", "PluginType", "PluginMeta", "PluginRegistry", "PluginMarketplace"]
