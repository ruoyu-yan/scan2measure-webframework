"""
Noise Filter Plugin: statistical outlier removal and voxel downsampling.
"""

from typing import Any, Dict

import numpy as np

from plugins.base import Plugin, PluginMeta, PluginType


class NoiseFilterPlugin(Plugin):
    """Remove outliers and downsample point clouds."""

    def get_meta(self) -> PluginMeta:
        return PluginMeta(
            name="noise_filter",
            version="1.0.0",
            description="Statistical outlier removal and voxel downsampling "
                        "for point cloud preprocessing.",
            author="Scan2Measure Team",
            plugin_type=PluginType.PREPROCESSING,
            dependencies=["numpy", "scipy"],
            tags=["filter", "denoise", "outlier", "downsample", "preprocessing"],
        )

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "method": "statistical",  # "statistical" or "voxel"
            "nb_neighbors": 20,       # for statistical method
            "std_ratio": 2.0,         # for statistical method
            "voxel_size": 0.05,       # for voxel downsampling
        }

    def validate_input(self, data: Dict[str, Any]) -> bool:
        return "points" in data

    def execute(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = {**self.get_default_config(), **config}
        points = np.asarray(data["points"])

        if cfg["method"] == "statistical":
            points = self._statistical_outlier_removal(
                points, cfg["nb_neighbors"], cfg["std_ratio"]
            )
        elif cfg["method"] == "voxel":
            points = self._voxel_downsample(points, cfg["voxel_size"])

        result = dict(data)
        result["points"] = points
        result["noise_filter_applied"] = cfg["method"]
        return result

    @staticmethod
    def _statistical_outlier_removal(points: np.ndarray, nb_neighbors: int,
                                     std_ratio: float) -> np.ndarray:
        """Remove points whose mean distance to neighbors exceeds threshold."""
        from scipy.spatial import KDTree

        tree = KDTree(points[:, :3])
        dists, _ = tree.query(points[:, :3], k=nb_neighbors + 1)
        mean_dists = dists[:, 1:].mean(axis=1)  # exclude self
        threshold = mean_dists.mean() + std_ratio * mean_dists.std()
        mask = mean_dists < threshold
        return points[mask]

    @staticmethod
    def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
        """Downsample by averaging points within each voxel."""
        coords = points[:, :3]
        voxel_indices = np.floor(coords / voxel_size).astype(int)
        # Use unique voxel keys
        _, unique_idx = np.unique(
            voxel_indices, axis=0, return_index=True
        )
        return points[np.sort(unique_idx)]
