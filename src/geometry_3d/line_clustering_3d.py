"""Library for 3D line principal direction voting, classification, and intersection.

Mirrors FGPL algorithms from panoramic-localization/:
  - map_utils.py:61-74      — principal direction voting
  - edge_utils.py:76-90     — split_3d classification
  - fgpl/line_intersection.py:154-246 — 3D intersection
  - map_utils.py:87-101     — intersection mask construction

All functions take/return torch tensors. No classes.
"""

from pathlib import Path
import sys
import torch

_SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_ROOT / "utils"))

# Reuse existing icosphere code
from sphere_geometry import generate_sphere_points


# ── Principal direction voting ────────────────────────────────────────

def vote_principal_directions(dirs, sphere_level=5, n_directions=3,
                              suppression_thres=0.95):
    """Greedy icosphere voting for dominant 3D line directions.

    Args:
        dirs: (N, 3) unit direction vectors
        sphere_level: icosphere subdivision level (5 → ~2562 pts)
        n_directions: number of principal directions to extract
        suppression_thres: cosine threshold for inlier suppression

    Returns:
        principal_3d: (n_directions, 3) principal direction matrix
    """
    device = dirs.device
    sphere_pts = generate_sphere_points(sphere_level, device=device)
    # Keep upper hemisphere only — matches FGPL map_utils.py:139-140
    sphere_pts = sphere_pts[:sphere_pts.shape[0] // 2]

    principal = []
    vote_dirs = dirs.clone()
    for _ in range(n_directions):
        # Each line votes for the sphere point it aligns with most
        votes = torch.abs(vote_dirs @ sphere_pts.t()).argmax(-1)
        counts = votes.bincount(minlength=sphere_pts.shape[0])
        winner_idx = counts.argmax()
        principal.append(sphere_pts[winner_idx])
        # Suppress inliers aligned with this winner
        cosines = torch.abs(vote_dirs @ sphere_pts[winner_idx:winner_idx + 1].t())
        outlier = (cosines < suppression_thres).squeeze(-1)
        vote_dirs = vote_dirs[outlier]

    principal_3d = torch.stack(principal, dim=0)

    # Ensure right-handed frame — matches FGPL map_utils.py:181-182
    if torch.det(principal_3d) < 0:
        principal_3d[-1, :] *= -1

    return principal_3d


# ── Line classification ───────────────────────────────────────────────

def classify_lines_3d(dirs, principal_3d, inlier_thres=0.1):
    """Classify 3D lines by alignment with principal directions.

    Args:
        dirs: (N, 3) unit direction vectors
        principal_3d: (3, 3) principal directions
        inlier_thres: threshold — line is inlier if |dot| > 1 - inlier_thres

    Returns:
        edge_mask: (N, 3) bool tensor — True where line belongs to group k
    """
    # Mirrors FGPL edge_utils.py:88-89
    inner = torch.abs(dirs @ principal_3d.t())  # (N, 3)
    return inner > 1 - inlier_thres


# ── 3D line intersection ─────────────────────────────────────────────

def find_intersections_3d(dirs, starts, ends, principal_3d,
                          inlier_thres=0.1, intersect_thres=0.2):
    """Find pairwise intersections of 3D line segments from adjacent groups.

    For each adjacent group pair (i, (i+1)%3): meshgrid all pairs →
    closest-point formula → filter by segment bounds + distance.

    Mirrors FGPL fgpl/line_intersection.py:154-246.

    Args:
        dirs: (N, 3) unit direction vectors
        starts: (N, 3) segment start points
        ends: (N, 3) segment end points
        principal_3d: (3, 3) principal directions
        inlier_thres: classification threshold
        intersect_thres: max distance for valid intersection

    Returns:
        inter_pts_list: list of 3 tensors, each (M_k, 3) intersection points
        inter_idx_list: list of 3 tensors, each (M_k, 2) source line index pairs
    """
    device = dirs.device
    edge_mask = classify_lines_3d(dirs, principal_3d, inlier_thres)

    starts_p = [starts[edge_mask[:, i]] for i in range(3)]
    ends_p = [ends[edge_mask[:, i]] for i in range(3)]

    full_range = torch.arange(dirs.shape[0], device=device)
    ids_p = [full_range[edge_mask[:, i]] for i in range(3)]

    inter_pts_list = []
    inter_idx_list = []

    for i in range(3):
        j = (i + 1) % 3
        s0, e0, id0 = starts_p[i], ends_p[i], ids_p[i]
        s1, e1, id1 = starts_p[j], ends_p[j], ids_p[j]
        n0, n1 = s0.shape[0], s1.shape[0]

        if n0 == 0 or n1 == 0:
            inter_pts_list.append(torch.zeros((0, 3), device=device))
            inter_idx_list.append(torch.zeros((0, 2), device=device, dtype=torch.long))
            continue

        # Meshgrid all pairs
        gx, gy = torch.meshgrid(torch.arange(n0), torch.arange(n1), indexing='ij')
        idx_flat = torch.stack([gx, gy], dim=-1).reshape(-1, 2)

        s0_exp = s0[idx_flat[:, 0]]
        e0_exp = e0[idx_flat[:, 0]]
        s1_exp = s1[idx_flat[:, 1]]
        e1_exp = e1[idx_flat[:, 1]]
        id0_exp = id0[idx_flat[:, 0]]
        id1_exp = id1[idx_flat[:, 1]]

        # Closest-point formula
        d0 = e0_exp - s0_exp  # (P, 3)
        d1 = e1_exp - s1_exp
        cross = torch.cross(d0, d1, dim=-1)
        cross_norm = torch.norm(cross, dim=-1)  # (P,)

        diff = s1_exp - s0_exp
        sd1 = torch.cross(diff, d1, dim=-1)
        sd0 = torch.cross(diff, d0, dim=-1)
        u = (sd1 * cross).sum(dim=-1) / cross_norm.square()
        v = (sd0 * cross).sum(dim=-1) / cross_norm.square()

        len0 = torch.norm(d0, dim=1)
        len1 = torch.norm(d1, dim=1)

        # Check segment bounds (with tolerance)
        on0 = (u > -intersect_thres / len0) & (u < 1 + intersect_thres / len0)
        on1 = (v > -intersect_thres / len1) & (v < 1 + intersect_thres / len1)
        on_both = on0 & on1

        pt0 = s0_exp[on_both] + u[on_both].unsqueeze(-1) * d0[on_both]
        pt1 = s1_exp[on_both] + v[on_both].unsqueeze(-1) * d1[on_both]
        valid = torch.norm(pt0 - pt1, dim=-1) < intersect_thres
        intersects = pt0[valid]

        ids = torch.stack([id0_exp[on_both][valid],
                           id1_exp[on_both][valid]], dim=-1)

        inter_pts_list.append(intersects if intersects.shape[0] > 0
                              else torch.zeros((0, 3), device=device))
        inter_idx_list.append(ids if ids.shape[0] > 0
                              else torch.zeros((0, 2), device=device, dtype=torch.long))

    return inter_pts_list, inter_idx_list


# ── Intersection mask construction ────────────────────────────────────

def build_intersection_masks(inter_pts_list, device='cpu'):
    """Convert per-group-pair lists to concatenated tensors + bool mask.

    Mirrors FGPL map_utils.py:87-101.

    Args:
        inter_pts_list: list of 3 tensors from find_intersections_3d
        device: torch device

    Returns:
        inter_3d: (M, 3) concatenated intersection points
        inter_3d_mask: (M, 3) bool — which 2 axes each intersection lies along
    """
    masks = []
    for k in range(3):
        pts = inter_pts_list[k]
        if pts.shape[0] > 0:
            m = torch.zeros_like(pts, dtype=torch.bool)
            m[:, k] = True
            m[:, (k + 1) % 3] = True
            masks.append(m)
        else:
            masks.append(torch.zeros((0, 3), device=device, dtype=torch.bool))

    return torch.cat(masks, dim=0)


# ── OBJ export: colored lines ────────────────────────────────────────

_GROUP_COLORS = {
    'group_0': (1.0, 0.0, 0.0),
    'group_1': (0.0, 1.0, 0.0),
    'group_2': (0.0, 0.0, 1.0),
    'unclassified': (0.5, 0.5, 0.5),
}


def write_colored_lines_obj(filepath, starts, ends, group_mask, principal_3d):
    """Write OBJ + MTL with lines colored by principal direction group.

    Args:
        filepath: output .obj path (MTL written alongside)
        starts: (N, 3) segment start points
        ends: (N, 3) segment end points
        group_mask: (N, 3) bool classification mask
        principal_3d: (3, 3) principal directions (for MTL comments)
    """
    filepath = Path(filepath)
    mtl_path = filepath.with_suffix('.mtl')

    s = starts.cpu().numpy()
    e = ends.cpu().numpy()
    mask = group_mask.cpu().numpy()
    pd = principal_3d.cpu().numpy()

    # Write MTL
    with open(mtl_path, 'w') as f:
        for name, (r, g, b) in _GROUP_COLORS.items():
            f.write(f"newmtl {name}\n")
            f.write(f"Kd {r:.2f} {g:.2f} {b:.2f}\n\n")
        for i in range(3):
            d = pd[i]
            f.write(f"# group_{i} principal direction: "
                    f"[{d[0]:+.6f}, {d[1]:+.6f}, {d[2]:+.6f}]\n")

    # Assign each line to a group (first matching, or unclassified)
    N = s.shape[0]
    groups = {}  # group_name → list of line indices
    for name in _GROUP_COLORS:
        groups[name] = []
    for idx in range(N):
        assigned = False
        for g in range(3):
            if mask[idx, g]:
                groups[f'group_{g}'].append(idx)
                assigned = True
                break
        if not assigned:
            groups['unclassified'].append(idx)

    # Write OBJ
    with open(filepath, 'w') as f:
        f.write(f"mtllib {mtl_path.name}\n")
        # Write all vertices first (2 per segment)
        for idx in range(N):
            f.write(f"v {s[idx, 0]:.6f} {s[idx, 1]:.6f} {s[idx, 2]:.6f}\n")
            f.write(f"v {e[idx, 0]:.6f} {e[idx, 1]:.6f} {e[idx, 2]:.6f}\n")
        # Write lines by group
        for name in _GROUP_COLORS:
            if not groups[name]:
                continue
            f.write(f"\nusemtl {name}\n")
            for idx in groups[name]:
                v1 = idx * 2 + 1  # OBJ is 1-indexed
                v2 = idx * 2 + 2
                f.write(f"l {v1} {v2}\n")


# ── OBJ export: intersection octahedra ───────────────────────────────

_PAIR_COLORS = {
    'pair_01': (1.0, 0.5, 0.0),   # orange
    'pair_12': (0.0, 1.0, 1.0),   # cyan
    'pair_20': (1.0, 0.0, 1.0),   # magenta
}

# Octahedron: 6 vertices at ±1 along each axis, 8 triangular faces
_OCTA_VERTS = [
    ( 1,  0,  0), (-1,  0,  0),
    ( 0,  1,  0), ( 0, -1,  0),
    ( 0,  0,  1), ( 0,  0, -1),
]
_OCTA_FACES = [
    (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
    (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5),
]


def write_intersections_obj(filepath, inter_pts_list, marker_radius=0.03):
    """Write OBJ + MTL with octahedron markers at intersection points.

    Args:
        filepath: output .obj path
        inter_pts_list: list of 3 tensors, each (M_k, 3) intersection points
        marker_radius: size of each octahedron marker
    """
    filepath = Path(filepath)
    mtl_path = filepath.with_suffix('.mtl')

    pair_names = ['pair_01', 'pair_12', 'pair_20']

    # Write MTL
    with open(mtl_path, 'w') as f:
        for name, (r, g, b) in _PAIR_COLORS.items():
            f.write(f"newmtl {name}\n")
            f.write(f"Kd {r:.2f} {g:.2f} {b:.2f}\n\n")

    # Write OBJ
    with open(filepath, 'w') as f:
        f.write(f"mtllib {mtl_path.name}\n")
        vert_offset = 1  # OBJ is 1-indexed

        for k in range(3):
            pts = inter_pts_list[k]
            if pts.shape[0] == 0:
                continue
            pts_np = pts.cpu().numpy()
            f.write(f"\nusemtl {pair_names[k]}\n")

            for center in pts_np:
                # Write 6 octahedron vertices
                for vx, vy, vz in _OCTA_VERTS:
                    x = center[0] + vx * marker_radius
                    y = center[1] + vy * marker_radius
                    z = center[2] + vz * marker_radius
                    f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                # Write 8 triangular faces
                for i0, i1, i2 in _OCTA_FACES:
                    f.write(f"f {vert_offset + i0} "
                            f"{vert_offset + i1} "
                            f"{vert_offset + i2}\n")
                vert_offset += 6
