"""
Base plugin interface and types for the Scan2Measure plugin system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PluginType(Enum):
    """Categories of plugins in the scan-to-measure pipeline."""
    FEATURE_EXTRACTION = "feature_extraction"
    GEOMETRY_PROCESSING = "geometry_processing"
    POSE_ESTIMATION = "pose_estimation"
    ML_INFERENCE = "ml_inference"
    VISUALIZATION = "visualization"
    DATA_IO = "data_io"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"


@dataclass
class PluginMeta:
    """Metadata describing a plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    homepage: str = ""
    license: str = "MIT"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "plugin_type": self.plugin_type.value,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "homepage": self.homepage,
            "license": self.license,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginMeta":
        data = dict(data)
        data["plugin_type"] = PluginType(data["plugin_type"])
        return cls(**data)


class Plugin(ABC):
    """
    Base class for all Scan2Measure plugins.

    Subclass this and implement the required methods to create a plugin.
    Each plugin must declare its metadata and implement execute().
    """

    @abstractmethod
    def get_meta(self) -> PluginMeta:
        """Return metadata describing this plugin."""
        ...

    @abstractmethod
    def execute(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the plugin's main logic.

        Args:
            data: Input data from the pipeline (point clouds, images, features, etc.)
            config: Configuration parameters for this plugin run.

        Returns:
            Output data to pass to the next pipeline stage.
        """
        ...

    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Optional: validate that the input data has required keys."""
        return True

    def on_install(self) -> None:
        """Optional: hook called when the plugin is first installed."""
        pass

    def on_uninstall(self) -> None:
        """Optional: hook called when the plugin is uninstalled."""
        pass

    def get_default_config(self) -> Dict[str, Any]:
        """Optional: return default configuration for this plugin."""
        return {}

    def __repr__(self) -> str:
        meta = self.get_meta()
        return f"<Plugin: {meta.name} v{meta.version} ({meta.plugin_type.value})>"
