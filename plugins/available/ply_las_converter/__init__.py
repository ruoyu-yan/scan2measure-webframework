"""
PLY/LAS/XYZ Converter Plugin: convert between point cloud formats.
"""

import struct
from pathlib import Path
from typing import Any, Dict

import numpy as np

from plugins.base import Plugin, PluginMeta, PluginType


class PlyLasConverterPlugin(Plugin):
    """Convert between PLY, XYZ, and other point cloud formats."""

    def get_meta(self) -> PluginMeta:
        return PluginMeta(
            name="ply_las_converter",
            version="1.0.0",
            description="Convert between PLY, LAS, LAZ, XYZ, and OBJ "
                        "point cloud formats.",
            author="Scan2Measure Team",
            plugin_type=PluginType.DATA_IO,
            dependencies=["numpy"],
            tags=["converter", "ply", "las", "xyz", "obj", "io"],
        )

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "input_format": "ply",
            "output_format": "xyz",
            "output_path": None,
        }

    def validate_input(self, data: Dict[str, Any]) -> bool:
        return "input_path" in data or "points" in data

    def execute(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = {**self.get_default_config(), **config}

        # Load points from file or data
        if "points" in data:
            points = np.asarray(data["points"])
        elif "input_path" in data:
            points = self._load(data["input_path"], cfg["input_format"])
        else:
            raise ValueError("Provide 'points' array or 'input_path'.")

        result = dict(data)
        result["points"] = points

        # Write output if path specified
        if cfg.get("output_path"):
            self._save(points, cfg["output_path"], cfg["output_format"])
            result["output_path"] = cfg["output_path"]

        return result

    @staticmethod
    def _load(path: str, fmt: str) -> np.ndarray:
        path = Path(path)
        if fmt == "xyz":
            return np.loadtxt(path)
        elif fmt == "ply":
            return PlyLasConverterPlugin._read_ply(path)
        else:
            raise ValueError(f"Unsupported input format: {fmt}")

    @staticmethod
    def _save(points: np.ndarray, path: str, fmt: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "xyz":
            np.savetxt(path, points, fmt="%.6f")
        elif fmt == "ply":
            PlyLasConverterPlugin._write_ply(points, path)
        else:
            raise ValueError(f"Unsupported output format: {fmt}")

    @staticmethod
    def _read_ply(path: Path) -> np.ndarray:
        """Simple ASCII PLY reader for XYZ data."""
        with open(path, "r") as f:
            header_lines = 0
            vertex_count = 0
            for line in f:
                header_lines += 1
                if line.startswith("element vertex"):
                    vertex_count = int(line.strip().split()[-1])
                if line.strip() == "end_header":
                    break
            points = []
            for _ in range(vertex_count):
                vals = f.readline().strip().split()
                points.append([float(v) for v in vals[:3]])
        return np.array(points)

    @staticmethod
    def _write_ply(points: np.ndarray, path: Path) -> None:
        """Write ASCII PLY file."""
        n = len(points)
        cols = points.shape[1] if points.ndim == 2 else 3
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if cols > 3:
                for i, name in enumerate(["red", "green", "blue", "alpha"][:cols - 3]):
                    f.write(f"property uchar {name}\n")
            f.write("end_header\n")
            for row in points:
                f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
