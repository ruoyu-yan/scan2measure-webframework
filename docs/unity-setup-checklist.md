# Unity Virtual Tour — Setup Checklist

All 13 C# scripts are already written at `unity/Assets/Scripts/`. This checklist covers the GUI-only steps needed in Unity Editor to wire everything together.

**Time estimate**: ~15 minutes

**Requires**: Unity Hub + Unity Editor 2022.3 LTS (any 2022.3.x build) installed on Windows

> **Note**: The Unity project lives on a Windows-native path, not inside WSL, because Unity
> cannot create projects on `\\wsl.localhost\` paths. The Electron app and `unity-launcher.ts`
> are configured to search both locations.

---

## A. Create Project (~2 min)

- [x] Open Unity Hub
- [x] Click "New Project"
- [x] Template: **3D (URP)** (Universal Render Pipeline)
- [x] Project name: `VirtualTour`
- [x] Location: `E:\OneDrive\File\Uni\KIT\WS25-26\Master Thesis\unity\`
  - Originally planned at `<repo_root>/unity/`, but Unity Hub cannot create projects on WSL network paths.
  - Actual project path: `E:\OneDrive\File\Uni\KIT\WS25-26\Master Thesis\unity\VirtualTour\`
- [x] Click "Create project" and wait for it to open

---

## B. Install Packages (~2 min)

### GLTFast (loads .glb mesh files at runtime)
- [ ] Window > Package Manager
- [ ] Click "+" > "Add package by name"
- [ ] Enter: `com.unity.cloud.gltfast` version `6.9.0`
- [ ] Click Add, wait for import

### TextMeshPro (renders distance labels and UI text)
- [ ] Window > Package Manager > Unity Registry tab
- [ ] Search "TextMeshPro", click Install
- [ ] When prompted, click "Import TMP Essential Resources"

---

## C. Copy Scripts (~1 min)

The scripts were written to `<repo_root>/unity/Assets/Scripts/` (WSL). Since the Unity project is at
`E:\OneDrive\File\Uni\KIT\WS25-26\Master Thesis\unity\VirtualTour\`, you need to copy them:
- [ ] Copy all 13 `.cs` files from `\\wsl.localhost\Ubuntu-22.04\home\ruoyu\scan2measure-webframework\unity\Assets\Scripts\`
      into `E:\OneDrive\File\Uni\KIT\WS25-26\Master Thesis\unity\VirtualTour\Assets\Scripts\`
- [ ] In Unity's Project panel, right-click Assets > "Refresh"
- [ ] Verify all 13 `.cs` files appear under `Assets/Scripts/`
- [ ] Check the Console panel — there should be zero compilation errors

---

## D. Set Up Scene (~5 min)

Open `Assets/Scenes/SampleScene` (the default scene).

### Lighting
- [ ] Delete the default "Directional Light" in the Hierarchy
- [ ] GameObject > Light > Directional Light
- [ ] Set Rotation: `(50, -30, 0)`, Intensity: `1.0`, Shadow Type: `Soft Shadows`

### Player
- [ ] Delete the default "Main Camera"
- [ ] GameObject > Create Empty, name it **"Player"**
- [ ] Set Player position: `(0, 1.6, 0)`
- [ ] Add Component: **CharacterController** — Height: `2.0`, Radius: `0.3`, Center: `(0, 0, 0)`
- [ ] Add Component: **FirstPersonController** (from Scripts)
- [ ] With Player selected: GameObject > Camera, name it **"PlayerCamera"**
  - It becomes a child of Player automatically
  - Set local position: `(0, 0.2, 0)`
  - Tag it as **"MainCamera"** (dropdown at top of Inspector)

### UI Canvas
- [ ] GameObject > UI > Canvas, name it **"UICanvas"**
- [ ] Set Render Mode: `Screen Space - Overlay`
- [ ] Add "Canvas Scaler" component (should already exist), set:
  - UI Scale Mode: `Scale With Screen Size`
  - Reference Resolution: `1920 x 1080`

---

## E. Create GameObjects & Attach Scripts (~5 min)

### AppBootstrap
- [ ] GameObject > Create Empty, name **"AppBootstrap"**
- [ ] Add Component: **AppBootstrap** (from Scripts)
- [ ] Edit > Project Settings > Script Execution Order: set AppBootstrap to **-100** (runs first)
- [ ] For Editor testing, set `Editor GLB Path` in Inspector to:
  ```
  <repo_root>/data/mesh/tmb_office_one_corridor_dense/tmb_office_one_corridor_dense_noRGB_textured.glb
  ```
  (Use your Windows path, e.g. `C:\Users\ruoyu\scan2measure-webframework\data\mesh\...`)
- [ ] Optionally set `Editor Minimap Path` to:
  ```
  <repo_root>/data/density_image/tmb_office_one_corridor_dense/tmb_office_one_corridor_dense.png
  ```

### GLBLoader
- [ ] GameObject > Create Empty, name **"GLBLoader"**
- [ ] Add Component: **GLBLoader** (from Scripts)
- [ ] Drag the **"Player"** GameObject into the `Player Transform` field in Inspector

### MeasurementManager
- [ ] GameObject > Create Empty, name **"MeasurementManager"**
- [ ] Add Component: **MeasurementManager**
- [ ] Add Component: **PointToPointTool**
- [ ] Add Component: **WallToWallTool**
- [ ] Add Component: **HeightTool**
  (All 4 scripts go on the same GameObject)

### UI Scripts (on UICanvas)
- [ ] Select **"UICanvas"** in Hierarchy
- [ ] Add Component: **ToolbarUI**
- [ ] Add Component: **CrosshairUI**
- [ ] Add Component: **MeshInfoPanel**

### MinimapController
- [ ] GameObject > Create Empty, name **"MinimapController"**
- [ ] Add Component: **MinimapController** (from Scripts)
- [ ] Drag the **"Player"** GameObject into the `Player Transform` field

---

## F. Test in Editor (~1 min)

- [ ] Press **Play** (the triangle button at top center)
- [ ] Expected: GLB mesh loads, player is positioned at mesh center, WASD moves, mouse looks around
- [ ] Press **1** — activates Point-to-Point tool, click two surfaces to measure distance
- [ ] Press **2** — activates Wall-to-Wall tool
- [ ] Press **3** — activates Height tool
- [ ] Press **Alt** — toggles cursor lock for UI interaction
- [ ] Press **Delete** — clears all measurements
- [ ] Check bottom-right for minimap (only appears if minimap path is set)
- [ ] Check bottom-left for mesh info panel (only appears if metadata path is set)

---

## G. Build Executable (~2 min)

- [ ] File > Build Settings
- [ ] Platform: **PC, Mac & Linux Standalone**
- [ ] Target Platform: **Windows**, Architecture: **x86_64**
- [ ] Click "Player Settings":
  - Product Name: `VirtualTour`
  - Company Name: `scan2measure`
  - Default Screen Width: `1920`, Height: `1080`
  - Check: `Resizable Window`, `Run In Background`
- [ ] Click "Build"
- [ ] Choose output folder: `E:\OneDrive\File\Uni\KIT\WS25-26\Master Thesis\unity\VirtualTour\Build\`
- [ ] Result: `E:\...\unity\VirtualTour\Build\VirtualTour.exe`

This executable is what the Electron app launches with (it searches both the repo `unity/Build/` and the external path):
```
VirtualTour.exe --glb "path/to/mesh.glb" --minimap "path/to/density.png" --metadata "path/to/metadata.json"
```

---

## Final Hierarchy

After setup, your scene Hierarchy should look like:

```
SampleScene
├── Directional Light
├── Player                      [CharacterController, FirstPersonController]
│   └── PlayerCamera            [Camera, tag: MainCamera]
├── AppBootstrap                [AppBootstrap]
├── GLBLoader                   [GLBLoader → playerTransform: Player]
├── MeasurementManager          [MeasurementManager, PointToPointTool, WallToWallTool, HeightTool]
├── UICanvas                    [Canvas, CanvasScaler, ToolbarUI, CrosshairUI, MeshInfoPanel]
│   └── EventSystem             (auto-created with Canvas)
└── MinimapController           [MinimapController → playerTransform: Player]
```
