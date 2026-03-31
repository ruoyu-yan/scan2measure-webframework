"""Per-face visibility check via depth-buffer rendering.

Renders depth maps from each cubemap face camera using Open3D raycasting,
then determines which mesh faces are visible from which cameras. Produces
a labeling file for texrecon's --labeling_file option, bypassing its
broken built-in visibility test.
"""

import time

import numpy as np
import open3d as o3d


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def render_depth_maps(verts, tris, intrinsics, extrinsics, width, height):
    """Render depth maps from each camera using Open3D raycasting.

    Args:
        verts: (V, 3) float64 mesh vertices
        tris: (F, 3) int32 mesh triangle indices
        intrinsics: list of N (3, 3) pinhole intrinsic matrices
        extrinsics: list of N (4, 4) world-to-camera extrinsic matrices
        width, height: image dimensions

    Returns:
        list of N (height, width) float32 depth maps (inf where no hit)
    """
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh()
    mesh_t.vertex.positions = o3d.core.Tensor(verts.astype(np.float32))
    mesh_t.triangle.indices = o3d.core.Tensor(tris.astype(np.int32))
    scene.add_triangles(mesh_t)

    depth_maps = []
    for K, ext in zip(intrinsics, extrinsics):
        K_t = o3d.core.Tensor(K.astype(np.float64))
        ext_t = o3d.core.Tensor(ext.astype(np.float64))
        rays = scene.create_rays_pinhole(K_t, ext_t, width, height)
        result = scene.cast_rays(rays)
        depth_maps.append(result['t_hit'].numpy())

    return depth_maps


def compute_face_visibility(verts, tris, face_normals, intrinsics, extrinsics,
                            depth_maps, width, height,
                            depth_margin=0.05, max_angle_deg=75.0):
    """Determine which faces are visible from which cameras.

    Args:
        verts: (V, 3) mesh vertices
        tris: (F, 3) triangle indices
        face_normals: (F, 3) per-face normal vectors
        intrinsics: list of N (3, 3) intrinsic matrices
        extrinsics: list of N (4, 4) extrinsic matrices
        depth_maps: list of N (H, W) depth images
        width, height: image dimensions
        depth_margin: tolerance in meters for depth comparison
        max_angle_deg: maximum viewing angle (degrees)

    Returns:
        visible: (F, N) bool array — visible[i, j] = face i visible from camera j
    """
    n_faces = len(tris)
    n_cams = len(intrinsics)
    cos_max_angle = np.cos(np.radians(max_angle_deg))

    # Compute face centroids
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0  # (F, 3)

    visible = np.zeros((n_faces, n_cams), dtype=bool)

    for j in range(n_cams):
        R = extrinsics[j][:3, :3]
        t = extrinsics[j][:3, 3]
        K = intrinsics[j]

        # Transform centroids to camera frame
        p_cam = (R @ centroids.T).T + t  # (F, 3)

        # Check depth > 0 (in front of camera)
        depth = p_cam[:, 2]
        in_front = depth > 0.1

        # Pinhole projection
        px = K[0, 0] * p_cam[:, 0] / depth + K[0, 2]
        py = K[1, 1] * p_cam[:, 1] / depth + K[1, 2]

        # Check within image bounds
        in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)

        # Candidate mask (in front + in bounds)
        candidates = in_front & in_bounds
        idx = np.where(candidates)[0]
        if len(idx) == 0:
            continue

        # Depth buffer comparison
        u_int = np.clip(px[idx].astype(np.int32), 0, width - 1)
        v_int = np.clip(py[idx].astype(np.int32), 0, height - 1)
        rendered_depth = depth_maps[j][v_int, u_int]
        depth_match = np.abs(depth[idx] - rendered_depth) < depth_margin

        # Viewing angle check: angle between face normal and camera-to-face direction
        # Camera position in world: cam_pos = -R^T @ t
        cam_pos = -R.T @ t
        to_face = centroids[idx] - cam_pos
        to_face_norm = to_face / (np.linalg.norm(to_face, axis=1, keepdims=True) + 1e-10)
        cos_angle = np.abs(np.sum(face_normals[idx] * to_face_norm, axis=1))
        angle_ok = cos_angle > cos_max_angle

        # Combine all checks
        pass_all = depth_match & angle_ok
        visible[idx[pass_all], j] = True

    return visible


def select_best_views(visible, centroids, face_normals, extrinsics):
    """Select the best camera for each face from visible cameras.

    Uses cos(angle) / distance² as quality metric, matching texrecon's
    area data term behavior.

    Args:
        visible: (F, N) bool visibility matrix
        centroids: (F, 3) face centroids
        face_normals: (F, 3) face normals
        extrinsics: list of N (4, 4) extrinsic matrices

    Returns:
        labeling: (F,) uint64 — 0 = unseen, 1..N = 1-indexed camera ID
    """
    n_faces, n_cams = visible.shape

    # Precompute camera positions
    cam_positions = []
    for ext in extrinsics:
        R = ext[:3, :3]
        t = ext[:3, 3]
        cam_positions.append(-R.T @ t)
    cam_positions = np.array(cam_positions)  # (N, 3)

    # Compute quality for all face-camera pairs
    quality = np.full((n_faces, n_cams), -np.inf, dtype=np.float64)

    for j in range(n_cams):
        mask = visible[:, j]
        if not mask.any():
            continue
        idx = np.where(mask)[0]

        to_cam = cam_positions[j] - centroids[idx]
        dist_sq = np.sum(to_cam ** 2, axis=1)
        dist = np.sqrt(dist_sq)
        to_cam_dir = to_cam / (dist[:, np.newaxis] + 1e-10)

        cos_angle = np.abs(np.sum(face_normals[idx] * to_cam_dir, axis=1))
        q = cos_angle / (dist_sq + 1e-10)
        quality[idx, j] = q

    # Select best camera per face
    labeling = np.zeros(n_faces, dtype=np.uint64)
    has_visible = visible.any(axis=1)
    best_cam = np.argmax(quality, axis=1)  # index of best camera
    labeling[has_visible] = best_cam[has_visible].astype(np.uint64) + 1  # 1-indexed

    return labeling


def write_labeling_file(labeling, path):
    """Write labeling as raw binary uint64 for texrecon --labeling_file."""
    labeling.astype(np.uint64).tofile(str(path))


def compute_and_write_labeling(mesh_ply_path, pano_names, pose_json_path,
                                cubemap_face_size, labeling_path):
    """Main entry point: compute visibility and write labeling file.

    Args:
        mesh_ply_path: path to the decimated mesh PLY
        pano_names: list of panorama name strings
        pose_json_path: path to local_filter_results.json
        cubemap_face_size: pixel size of cubemap faces (e.g. 1024)
        labeling_path: output path for the labeling .vec file

    Returns:
        dict with stats (n_visible_faces, n_unseen, camera_counts, etc.)
    """
    import json
    from pathlib import Path
    import sys

    _SCRIPT_DIR = Path(__file__).resolve().parent
    _SRC_ROOT = _SCRIPT_DIR.parent
    sys.path.insert(0, str(_SRC_ROOT / "meshing"))
    from cubemap_utils import build_cubemap_cameras

    # Load mesh
    log("Loading mesh for visibility computation...")
    mesh = o3d.io.read_triangle_mesh(str(mesh_ply_path))
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    n_faces = len(tris)
    log(f"  {len(verts):,} vertices, {n_faces:,} faces")

    # Compute face normals
    mesh.compute_triangle_normals()
    face_normals = np.asarray(mesh.triangle_normals)

    # Load poses
    with open(pose_json_path) as f:
        poses = json.load(f)

    # Build camera list (all cubemap face cameras across all panos)
    all_intrinsics = []
    all_extrinsics = []
    cam_labels = []

    for pano_name in pano_names:
        if pano_name not in poses:
            log(f"  WARNING: {pano_name} not in pose JSON, skipping")
            continue
        pose = poses[pano_name]
        R_pano = np.array(pose["R"], dtype=np.float64)
        t_pano = np.array(pose["t"], dtype=np.float64)

        intrinsics, extrinsics = build_cubemap_cameras(R_pano, t_pano,
                                                        cubemap_face_size)
        face_names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
        for i, (K, ext) in enumerate(zip(intrinsics, extrinsics)):
            all_intrinsics.append(K)
            all_extrinsics.append(ext)
            cam_labels.append(f"{pano_name} {face_names[i]}")

    n_cams = len(all_intrinsics)
    log(f"  {n_cams} cameras ({len(pano_names)} panos × 6 faces)")

    # Render depth maps
    t0 = time.time()
    log("Rendering depth maps...")
    depth_maps = render_depth_maps(verts, tris, all_intrinsics, all_extrinsics,
                                   cubemap_face_size, cubemap_face_size)
    log(f"  {n_cams} depth maps rendered in {time.time() - t0:.1f}s")

    # Compute centroids (needed for visibility and view selection)
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0

    # Compute visibility
    t0 = time.time()
    log("Computing per-face visibility...")
    visible = compute_face_visibility(
        verts, tris, face_normals,
        all_intrinsics, all_extrinsics, depth_maps,
        cubemap_face_size, cubemap_face_size
    )
    n_visible_any = visible.any(axis=1).sum()
    log(f"  {n_visible_any:,} / {n_faces:,} faces visible from at least 1 camera "
        f"({n_visible_any / n_faces * 100:.1f}%)")
    log(f"  Visibility computed in {time.time() - t0:.1f}s")

    # Per-camera stats
    for j, label in enumerate(cam_labels):
        n = visible[:, j].sum()
        log(f"    {label}: {n:,} visible faces")

    # Select best views
    log("Selecting best camera per face...")
    labeling = select_best_views(visible, centroids, face_normals,
                                 all_extrinsics)

    n_unseen = (labeling == 0).sum()
    n_textured = (labeling > 0).sum()
    log(f"  {n_textured:,} faces assigned a camera, {n_unseen:,} unseen")

    # Per-camera assignment counts
    for j, label in enumerate(cam_labels):
        n = (labeling == j + 1).sum()
        if n > 0:
            log(f"    {label}: {n:,} faces assigned")

    # Write labeling file
    write_labeling_file(labeling, labeling_path)
    log(f"  Labeling written: {labeling_path} ({labeling_path.stat().st_size:,} bytes)")

    return {
        "n_faces": n_faces,
        "n_cameras": n_cams,
        "n_visible_any": int(n_visible_any),
        "n_textured": int(n_textured),
        "n_unseen": int(n_unseen),
    }
