"""Debug: project mesh wireframe onto cubemap face images.

For each panorama, projects a subset of mesh edges onto the 4 horizontal
cubemap faces and overlays them in red. If the projections land on the
correct geometry in the images, the cam files are correct.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SRC_ROOT.parent

SCENE_DIR = _PROJECT_ROOT / "data" / "mesh" / "tmb_office_one_corridor_bigger_noRGB" / "texrecon_scene"
MESH_PLY = _PROJECT_ROOT / "data" / "mesh" / "tmb_office_one_corridor_bigger_noRGB" / "texrecon_input_mesh.ply"
POSE_JSON = _PROJECT_ROOT / "data" / "pose_estimates" / "multiroom" / "local_filter_results.json"
OUT_DIR = _PROJECT_ROOT / "data" / "mesh" / "tmb_office_one_corridor_bigger_noRGB" / "debug_projection"

FACE_NAMES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


def load_cam(cam_path):
    """Load a .cam file, return R (3x3), t (3,), flen."""
    with open(cam_path) as f:
        line1 = f.readline().split()
        line2 = f.readline().split()
    t = np.array([float(x) for x in line1[:3]])
    R = np.array([float(x) for x in line1[3:12]]).reshape(3, 3)
    flen = float(line2[0])
    return R, t, flen


def project_points(pts_3d, R, t, flen, img_size):
    """Project 3D points to pixel coords. Returns (N,2) and validity mask."""
    p_cam = (R @ pts_3d.T).T + t  # (N, 3)
    depth = p_cam[:, 2]
    valid = depth > 0.1  # in front of camera

    f_px = flen * img_size
    cx = img_size / 2.0
    cy = img_size / 2.0

    px = f_px * p_cam[:, 0] / depth + cx
    py = f_px * p_cam[:, 1] / depth + cy

    valid &= (px >= 0) & (px < img_size) & (py >= 0) & (py < img_size)
    return np.stack([px, py], axis=1), valid


def main():
    import open3d as o3d

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load mesh
    print("Loading mesh...")
    mesh = o3d.io.read_triangle_mesh(str(MESH_PLY))
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    print(f"  {len(verts):,} vertices, {len(tris):,} triangles")

    # Subsample edges for visualization (every 20th triangle)
    step = 20
    sub_tris = tris[::step]
    edges = set()
    for tri in sub_tris:
        for i in range(3):
            a, b = sorted([tri[i], tri[(i + 1) % 3]])
            edges.add((a, b))
    edges = np.array(list(edges))
    print(f"  {len(edges):,} edges for visualization")

    # Load poses
    with open(POSE_JSON) as f:
        poses = json.load(f)
    pano_names = list(poses.keys())
    print(f"  Panoramas: {pano_names}")

    # For each panorama, project onto horizontal faces (skip +Z/-Z)
    view_idx = 0
    for pano_name in pano_names:
        t_pano = np.array(poses[pano_name]["t"])
        print(f"\n--- {pano_name} (center: {t_pano}) ---")

        for face_idx in range(6):
            cam_path = SCENE_DIR / f"view_{view_idx:04d}.cam"
            img_path = SCENE_DIR / f"view_{view_idx:04d}.jpg"

            R, t, flen = load_cam(cam_path)
            img = cv2.imread(str(img_path))
            img_size = img.shape[0]

            # Project all vertices
            pts_2d, valid = project_points(verts, R, t, flen, img_size)

            # Draw visible edges
            n_drawn = 0
            overlay = img.copy()
            for a, b in edges:
                if valid[a] and valid[b]:
                    p1 = tuple(pts_2d[a].astype(int))
                    p2 = tuple(pts_2d[b].astype(int))
                    cv2.line(overlay, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)
                    n_drawn += 1

            n_visible = valid.sum()
            face_name = FACE_NAMES[face_idx]
            print(f"  view_{view_idx:04d} ({face_name}): {n_visible:,} visible verts, {n_drawn:,} edges drawn")

            # Add label
            cv2.putText(overlay, f"{pano_name} {face_name} (view_{view_idx:04d})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(overlay, f"{n_visible:,} verts, {n_drawn:,} edges",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            out_path = OUT_DIR / f"{pano_name}_{face_name.replace('+','p').replace('-','m')}_view{view_idx:04d}.jpg"
            cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])

            view_idx += 1

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
