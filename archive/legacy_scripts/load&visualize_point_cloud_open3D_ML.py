import open3d as o3d
import os

# 1. Define the path (using the Linux path format for WSL)
# Note: Python inside WSL sees '/home/ruoyu/...' not '\\wsl.localhost\...'
file_path = "/home/ruoyu/scan2measure-webframework/data/point_cloud/Area_3.ply"

# 2. Check if file exists to avoid confusion
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit()

print(f"Loading point cloud from: {file_path}")

# 3. Load the point cloud
pcd = o3d.io.read_point_cloud(file_path)

# 4. Print basic info (sanity check)
print(f"Loaded {len(pcd.points)} points.")
print(f"Has colors? {pcd.has_colors()}")

# 5. Visualize
# This opens a window. Use your mouse to rotate (Left Click), pan (Shift+Left), and zoom (Scroll).
o3d.visualization.draw_geometries([pcd], 
                                  window_name="Area 3 Visualization",
                                  width=1024, 
                                  height=768)