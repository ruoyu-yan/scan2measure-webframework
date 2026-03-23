import numpy as np
import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "meshing"))
from export_gltf import export_textured_glb


def test_export_textured_glb_creates_file():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    uv_coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float64)
    texture = np.full((64, 64, 3), [128, 64, 32], dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        glb_path = Path(tmpdir) / "test.glb"
        export_textured_glb(vertices, faces, uv_coords, normals, texture_images=[texture], output_path=glb_path)
        assert glb_path.exists()
        assert glb_path.stat().st_size > 100


def test_export_textured_glb_preserves_metric_metadata():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    uv_coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    normals = np.array([[0, 0, 1]] * 3, dtype=np.float64)
    texture = np.full((64, 64, 3), 128, dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        glb_path = Path(tmpdir) / "test.glb"
        export_textured_glb(vertices, faces, uv_coords, normals, texture_images=[texture], output_path=glb_path)
        data = glb_path.read_bytes()
        json_len = struct.unpack_from("<I", data, 12)[0]
        json_str = data[20:20+json_len].decode("utf-8").rstrip("\x00 ")
        gltf = json.loads(json_str)
        assert gltf["asset"]["extras"]["unit"] == "meter"
        assert gltf["asset"]["extras"]["scale"] == 1.0
