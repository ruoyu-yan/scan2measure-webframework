"""GLB export for textured meshes with metric metadata.

Exports UV-textured meshes as GLB (binary glTF) preserving metric scale.
The glTF specification defines 1 unit = 1 meter, matching TLS data directly.
"""

import json
import struct
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def export_textured_glb(vertices, faces, uv_coords, normals,
                        texture_images, output_path, mesh_name="tls_mesh"):
    """Export a UV-textured mesh as GLB with metric metadata."""
    output_path = Path(output_path)

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(texture_images[0]),
    )
    visual = trimesh.visual.TextureVisuals(
        uv=uv_coords.astype(np.float64),
        material=material,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        visual=visual,
        process=False,
    )
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name=mesh_name)

    glb_bytes = scene.export(file_type="glb")

    glb_bytes = _inject_gltf_metadata(glb_bytes, {
        "unit": "meter",
        "scale": 1.0,
        "source": "TLS_point_cloud",
        "note": "1 unit = 1 meter. No rescaling applied. Metric-accurate.",
    })

    with open(str(output_path), "wb") as f:
        f.write(glb_bytes)
    return str(output_path)


def export_vertex_color_ply(o3d_mesh, output_path):
    """Export Open3D mesh as PLY with vertex colors (for CloudCompare)."""
    import open3d as o3d
    o3d.io.write_triangle_mesh(str(output_path), o3d_mesh)


def _inject_gltf_metadata(glb_bytes, metadata):
    """Inject metadata into the glTF JSON chunk of a GLB binary."""
    if isinstance(glb_bytes, bytes):
        glb_bytes = bytearray(glb_bytes)
    magic, version, total_length = struct.unpack_from("<III", glb_bytes, 0)
    if magic != 0x46546C67:
        return bytes(glb_bytes)
    json_chunk_length = struct.unpack_from("<I", glb_bytes, 12)[0]
    json_chunk_type = struct.unpack_from("<I", glb_bytes, 16)[0]
    if json_chunk_type != 0x4E4F534A:
        return bytes(glb_bytes)
    json_data = glb_bytes[20:20 + json_chunk_length].decode("utf-8").rstrip("\x00")
    gltf = json.loads(json_data)
    if "asset" not in gltf:
        gltf["asset"] = {"version": "2.0", "generator": "scan2measure"}
    gltf["asset"]["extras"] = metadata
    new_json = json.dumps(gltf, separators=(",", ":"))
    while len(new_json) % 4 != 0:
        new_json += " "
    new_json_bytes = new_json.encode("utf-8")
    new_json_length = len(new_json_bytes)
    remaining_start = 20 + json_chunk_length
    remaining = glb_bytes[remaining_start:]
    new_total_length = 12 + 8 + new_json_length + len(remaining)
    result = bytearray()
    result += struct.pack("<III", magic, version, new_total_length)
    result += struct.pack("<II", new_json_length, 0x4E4F534A)
    result += new_json_bytes
    result += remaining
    return bytes(result)
