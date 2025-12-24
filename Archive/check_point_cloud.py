import open3d as o3d
import numpy as np

# Path to your original file
ply_path = "data/point_cloud/Area_3.ply"

print(f"Inspecting: {ply_path}")
pcd = o3d.io.read_point_cloud(ply_path)

# 1. Check Coordinates (Centering Issue)
points = np.asarray(pcd.points)
min_xyz = points.min(axis=0)
max_xyz = points.max(axis=0)
center = points.mean(axis=0)

print("\n--- COORDINATES ---")
print(f"Min (X,Y,Z): {min_xyz}")
print(f"Max (X,Y,Z): {max_xyz}")
print(f"Center     : {center}")

if np.abs(center).max() > 10.0:
    print(">> DIAGNOSIS: [PROBLEM FOUND] Coordinates are massive!")
    print("   The AI expects values near 0.0. Your data is far away.")
    print("   Fix: You MUST enable the 'Centering' fix in the script.")
else:
    print(">> DIAGNOSIS: Coordinates look centered.")

# 2. Check Colors (Normalization Issue)
if pcd.has_colors():
    colors = np.asarray(pcd.colors)
    max_color = colors.max()
    print("\n--- COLORS ---")
    print(f"Max Color Value: {max_color}")

    if max_color > 1.1:
        print(">> DIAGNOSIS: [PROBLEM FOUND] Colors are 0-255.")
        print("   The AI expects colors between 0.0 and 1.0.")
        print("   Fix: You MUST enable the 'Color Normalization' fix.")
    else:
        print(">> DIAGNOSIS: Colors are already normalized (0-1).")
else:
    print("\n--- COLORS ---")
    print(">> DIAGNOSIS: No colors found! The AI needs colors to work.")