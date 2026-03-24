# Unity Virtual Tour Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Unity virtual tour application with first-person navigation, collision detection, and measurement tools (point-to-point, wall-to-wall, height) that loads GLB meshes from command-line arguments.

**Architecture:** Single Unity scene with runtime GLB loading via GLTFast. CharacterController for first-person movement with mesh collider collision. Measurement system uses Physics.Raycast with a tool state machine. Canvas-based UI for toolbar and minimap.

**Tech Stack:** Unity 2022 LTS, C#, GLTFast, Unity UI Toolkit or UGUI

**Spec:** `docs/superpowers/specs/2026-03-24-desktop-app-design.md` (Section 7)

**Test GLB:** `data/mesh/tmb_office_one_corridor_dense/tmb_office_one_corridor_dense_noRGB_textured.glb` (15 MB, 500K triangles, 1 unit = 1 meter)

**Test density image:** `data/density_image/tmb_office_one_corridor_dense/tmb_office_one_corridor_dense.png` (256x256)

---

## File Structure

| File | Purpose |
|------|---------|
| `unity/Assets/Scripts/AppBootstrap.cs` | Entry point: parse CLI args, trigger GLB load |
| `unity/Assets/Scripts/GLBLoader.cs` | GLTFast runtime import, mesh collider generation |
| `unity/Assets/Scripts/FirstPersonController.cs` | WASD + mouse look, CharacterController, gravity, ground snap |
| `unity/Assets/Scripts/MeasurementManager.cs` | Tool state machine, raycast dispatcher, measurement storage |
| `unity/Assets/Scripts/PointToPointTool.cs` | Two-click point-to-point distance measurement |
| `unity/Assets/Scripts/WallToWallTool.cs` | Plane-fit wall measurement with perpendicular distance |
| `unity/Assets/Scripts/HeightTool.cs` | Floor/ceiling vertical distance measurement |
| `unity/Assets/Scripts/MeasurementRenderer.cs` | Dashed lines, distance labels, measurement visualization |
| `unity/Assets/Scripts/ToolbarUI.cs` | Top toolbar with tool buttons and status bar |
| `unity/Assets/Scripts/MinimapController.cs` | Secondary orthographic camera, density image ground plane, player dot |
| `unity/Assets/Scripts/MeshInfoPanel.cs` | Bottom-left text overlay showing mesh metadata (name, quality tier, triangle count) |

---

## Group A: Project Setup

### Task 1: Create Unity Project Structure

**Files:**
- Create: `unity/` project directory via Unity Hub or command line

- [ ] **Step 1: Create the Unity project**

Open Unity Hub and create a new 3D (URP) project at the path `<repo_root>/unity/`. Use Unity 2022.3 LTS. Name the project "VirtualTour".

Alternatively, from the command line (if Unity is in PATH):

```bash
"C:\Program Files\Unity\Hub\Editor\2022.3.XXf1\Editor\Unity.exe" -createProject "/home/ruoyu/scan2measure-webframework/unity" -quit -batchmode
```

- [ ] **Step 2: Add unity/ to .gitignore at repo root**

Append these lines to the repo root `.gitignore`:

```
# Unity
unity/Library/
unity/Temp/
unity/Obj/
unity/Build/
unity/Builds/
unity/Logs/
unity/UserSettings/
unity/*.csproj
unity/*.sln
unity/*.suo
unity/*.tmp
unity/*.user
unity/*.userprefs
unity/*.pidb
unity/*.booproj
unity/*.svd
unity/*.pdb
unity/*.mdb
unity/*.opendb
unity/*.VC.db
unity/**/*.meta
```

Note: `.meta` files ARE needed for Unity version control. If using Git for the Unity project, remove the last line and keep `.meta` files tracked. The recommendation is to keep `.meta` files tracked.

- [ ] **Step 3: Verify project opens**

Open the project in Unity Editor. Confirm it loads without errors and the default SampleScene exists.

---

### Task 2: Install GLTFast Package

**Files:**
- Modify: `unity/Packages/manifest.json`

- [ ] **Step 1: Add GLTFast via Package Manager**

In Unity Editor: Window > Package Manager > + > Add package by name > enter `com.unity.cloud.gltfast` > Add.

Alternatively, add this line to `unity/Packages/manifest.json` in the `dependencies` block:

```json
"com.unity.cloud.gltfast": "6.9.0"
```

- [ ] **Step 2: Verify GLTFast imported**

In the Unity console, confirm no errors. Check that `GLTFast` namespace is available by creating a temporary C# script with `using GLTFast;` and verifying it compiles.

---

### Task 3: Configure Scene and Lighting

**Files:**
- Modify: `unity/Assets/Scenes/SampleScene.unity` (via Editor)

- [ ] **Step 1: Set up the scene**

In SampleScene:
1. Delete the default Directional Light
2. Create a new Directional Light: Rotation (50, -30, 0), Intensity 1.0, Shadow Type: Soft Shadows
3. Set the camera background to solid color (dark gray: #1A1A1A)
4. Delete the default Main Camera (we will create one with the CharacterController)

- [ ] **Step 2: Create Player object**

1. Create an empty GameObject named "Player" at position (0, 1.6, 0)
2. Add a CharacterController component: Height=2.0, Radius=0.3, Center=(0, 0, 0)
3. Create a child Camera named "PlayerCamera" at local position (0, 0.2, 0) — this places the camera at 1.8m relative to feet at 1.6m start, simulating ~1.6m eye height when grounded
4. Tag the Camera as MainCamera

- [ ] **Step 3: Create UI Canvas**

1. Create a Canvas: Render Mode = Screen Space - Overlay, Canvas Scaler = Scale With Screen Size, Reference Resolution = 1920x1080
2. Name it "UICanvas"

---

## Group B: GLB Loading and Command-Line Interface

### Task 4: Command-Line Argument Parser

**Files:**
- Create: `unity/Assets/Scripts/AppBootstrap.cs`

- [ ] **Step 1: Create AppBootstrap.cs**

```csharp
using UnityEngine;
using System;
using System.IO;

public class AppBootstrap : MonoBehaviour
{
    public static string GLBPath { get; private set; }
    public static string MinimapPath { get; private set; }
    public static string MetadataPath { get; private set; }

    [Header("Fallback paths (Editor testing only)")]
    [SerializeField] private string editorGLBPath = "";
    [SerializeField] private string editorMinimapPath = "";

    void Awake()
    {
        ParseCommandLineArgs();

        // In Editor, use inspector fallback paths if no CLI args
        #if UNITY_EDITOR
        if (string.IsNullOrEmpty(GLBPath) && !string.IsNullOrEmpty(editorGLBPath))
            GLBPath = editorGLBPath;
        if (string.IsNullOrEmpty(MinimapPath) && !string.IsNullOrEmpty(editorMinimapPath))
            MinimapPath = editorMinimapPath;
        #endif

        if (string.IsNullOrEmpty(GLBPath))
        {
            Debug.LogError("No GLB path provided. Use --glb <path> or set editorGLBPath in Inspector.");
            #if !UNITY_EDITOR
            Application.Quit(1);
            #endif
            return;
        }

        if (!File.Exists(GLBPath))
        {
            Debug.LogError($"GLB file not found: {GLBPath}");
            #if !UNITY_EDITOR
            Application.Quit(1);
            #endif
            return;
        }

        Debug.Log($"GLB path: {GLBPath}");
        Debug.Log($"Minimap path: {(string.IsNullOrEmpty(MinimapPath) ? "(none)" : MinimapPath)}");
        Debug.Log($"Metadata path: {(string.IsNullOrEmpty(MetadataPath) ? "(none)" : MetadataPath)}");
    }

    private void ParseCommandLineArgs()
    {
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--glb" && i + 1 < args.Length)
                GLBPath = args[i + 1];
            else if (args[i] == "--minimap" && i + 1 < args.Length)
                MinimapPath = args[i + 1];
            else if (args[i] == "--metadata" && i + 1 < args.Length)
                MetadataPath = args[i + 1];
        }
    }
}
```

- [ ] **Step 2: Attach to scene**

Create an empty GameObject named "AppBootstrap" in SampleScene. Attach the `AppBootstrap` component. Set Script Execution Order to -100 (Edit > Project Settings > Script Execution Order) so it runs before other scripts.

---

### Task 5: GLB Runtime Loader

**Files:**
- Create: `unity/Assets/Scripts/GLBLoader.cs`

- [ ] **Step 1: Create GLBLoader.cs**

```csharp
using UnityEngine;
using GLTFast;
using System.Threading.Tasks;

public class GLBLoader : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private Transform playerTransform;

    private GameObject loadedRoot;

    async void Start()
    {
        string glbPath = AppBootstrap.GLBPath;
        if (string.IsNullOrEmpty(glbPath))
            return;

        await LoadGLB(glbPath);
    }

    private async Task LoadGLB(string path)
    {
        Debug.Log($"Loading GLB: {path}");

        var gltf = new GltfImport();
        bool success = await gltf.Load($"file://{path}");

        if (!success)
        {
            Debug.LogError("Failed to load GLB file.");
            return;
        }

        loadedRoot = new GameObject("LoadedMesh");
        await gltf.InstantiateMainSceneAsync(loadedRoot.transform);

        // GLTFast imports in glTF coordinate system (right-handed, Y-up).
        // Unity is left-handed Y-up. GLTFast handles this internally,
        // so no manual axis flip is needed.

        // Generate mesh colliders for all child meshes
        int colliderCount = 0;
        foreach (var mf in loadedRoot.GetComponentsInChildren<MeshFilter>())
        {
            var mc = mf.gameObject.AddComponent<MeshCollider>();
            mc.sharedMesh = mf.sharedMesh;
            colliderCount++;
        }
        Debug.Log($"GLB loaded: {colliderCount} mesh collider(s) generated.");

        // Position the player at the mesh center, slightly above the floor
        PositionPlayerAtMeshCenter();
    }

    private void PositionPlayerAtMeshCenter()
    {
        if (loadedRoot == null || playerTransform == null)
            return;

        // Compute combined bounds of all renderers
        Bounds combinedBounds = new Bounds();
        bool first = true;
        foreach (var renderer in loadedRoot.GetComponentsInChildren<Renderer>())
        {
            if (first)
            {
                combinedBounds = renderer.bounds;
                first = false;
            }
            else
            {
                combinedBounds.Encapsulate(renderer.bounds);
            }
        }

        if (first)
            return; // no renderers found

        // Place player at XZ center, Y at floor level + 1.6m
        float floorY = combinedBounds.min.y;
        Vector3 center = combinedBounds.center;
        playerTransform.position = new Vector3(center.x, floorY + 1.6f, center.z);

        Debug.Log($"Mesh bounds: min={combinedBounds.min}, max={combinedBounds.max}");
        Debug.Log($"Player placed at {playerTransform.position}");
    }
}
```

- [ ] **Step 2: Attach to scene**

Create an empty GameObject named "GLBLoader" in SampleScene. Attach the `GLBLoader` component. Drag the "Player" GameObject into the `playerTransform` field.

- [ ] **Step 3: Test with sample GLB**

Set `editorGLBPath` on AppBootstrap to:
```
/home/ruoyu/scan2measure-webframework/data/mesh/tmb_office_one_corridor_dense/tmb_office_one_corridor_dense_noRGB_textured.glb
```

On Windows, use the Windows-style path equivalent. Enter Play mode and verify:
- The mesh loads without errors
- Mesh colliders are generated (check Inspector on LoadedMesh children)
- The player is positioned near the center of the mesh
- Console shows bounds info confirming metric scale (bbox spans ~10m x 21m x 3.8m based on metadata)

---

## Group C: First-Person Navigation

### Task 6: First-Person Controller

**Files:**
- Create: `unity/Assets/Scripts/FirstPersonController.cs`

- [ ] **Step 1: Create FirstPersonController.cs**

```csharp
using UnityEngine;

[RequireComponent(typeof(CharacterController))]
public class FirstPersonController : MonoBehaviour
{
    [Header("Movement")]
    [SerializeField] private float moveSpeed = 3.0f;
    [SerializeField] private float sprintMultiplier = 2.0f;
    [SerializeField] private float gravity = -15.0f;

    [Header("Mouse Look")]
    [SerializeField] private float mouseSensitivity = 2.0f;
    [SerializeField] private float maxLookAngle = 85.0f;

    [Header("Ground Snap")]
    [SerializeField] private float targetEyeHeight = 1.6f;
    [SerializeField] private float snapSpeed = 8.0f;
    [SerializeField] private float groundCheckDistance = 3.0f;

    private CharacterController cc;
    private Transform cameraTransform;
    private float verticalVelocity;
    private float cameraPitch;
    private bool cursorLocked = true;

    void Start()
    {
        cc = GetComponent<CharacterController>();
        cameraTransform = GetComponentInChildren<Camera>().transform;
        LockCursor(true);
    }

    void Update()
    {
        HandleCursorToggle();

        if (cursorLocked)
        {
            HandleMouseLook();
            HandleMovement();
        }
    }

    private void HandleCursorToggle()
    {
        // Alt key toggles cursor lock (for UI interaction)
        if (Input.GetKeyDown(KeyCode.LeftAlt))
        {
            LockCursor(!cursorLocked);
        }
    }

    private void LockCursor(bool locked)
    {
        cursorLocked = locked;
        Cursor.lockState = locked ? CursorLockMode.Locked : CursorLockMode.None;
        Cursor.visible = !locked;
    }

    private void HandleMouseLook()
    {
        float mouseX = Input.GetAxis("Mouse X") * mouseSensitivity;
        float mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity;

        // Horizontal rotation on the Player object
        transform.Rotate(Vector3.up, mouseX);

        // Vertical rotation on the Camera (clamped)
        cameraPitch -= mouseY;
        cameraPitch = Mathf.Clamp(cameraPitch, -maxLookAngle, maxLookAngle);
        cameraTransform.localEulerAngles = new Vector3(cameraPitch, 0f, 0f);
    }

    private void HandleMovement()
    {
        float h = Input.GetAxisRaw("Horizontal");
        float v = Input.GetAxisRaw("Vertical");

        Vector3 moveDir = (transform.forward * v + transform.right * h).normalized;

        float speed = moveSpeed;
        if (Input.GetKey(KeyCode.LeftShift))
            speed *= sprintMultiplier;

        // Gravity
        if (cc.isGrounded)
        {
            verticalVelocity = -2.0f; // small downward force to keep grounded
        }
        else
        {
            verticalVelocity += gravity * Time.deltaTime;
        }

        Vector3 velocity = moveDir * speed;
        velocity.y = verticalVelocity;

        cc.Move(velocity * Time.deltaTime);

        // Ground snap: raycast down to find floor, adjust height smoothly
        SnapToGroundHeight();
    }

    private void SnapToGroundHeight()
    {
        // Cast a ray downward from the character's feet
        Vector3 rayOrigin = transform.position + Vector3.up * 0.1f;
        if (Physics.Raycast(rayOrigin, Vector3.down, out RaycastHit hit, groundCheckDistance))
        {
            float desiredY = hit.point.y + targetEyeHeight;
            float currentY = transform.position.y;

            // Only snap upward if we are on the ground, or snap down if floating
            if (cc.isGrounded || currentY > desiredY + 0.1f)
            {
                float newY = Mathf.Lerp(currentY, desiredY, snapSpeed * Time.deltaTime);
                Vector3 pos = transform.position;
                pos.y = newY;
                transform.position = pos;
            }
        }
    }

    /// <summary>
    /// Temporarily unlock cursor for UI interaction (called by ToolbarUI).
    /// </summary>
    public void SetCursorLocked(bool locked)
    {
        LockCursor(locked);
    }

    public bool IsCursorLocked => cursorLocked;
}
```

- [ ] **Step 2: Attach to Player**

Attach `FirstPersonController` to the "Player" GameObject in SampleScene. The CharacterController should already be on the Player from Task 3.

- [ ] **Step 3: Test navigation**

Enter Play mode with the GLB loaded. Verify:
- WASD moves the player through the corridor
- Mouse rotates the view
- The player collides with walls (cannot walk through)
- Gravity keeps the player on the floor
- Shift speeds up movement
- Alt toggles cursor lock

---

## Group D: Measurement System Core

### Task 7: Measurement Manager (State Machine)

**Files:**
- Create: `unity/Assets/Scripts/MeasurementManager.cs`

- [ ] **Step 1: Create MeasurementManager.cs**

```csharp
using UnityEngine;
using System;
using System.Collections.Generic;

public enum MeasurementTool
{
    None,
    PointToPoint,
    WallToWall,
    Height
}

public class MeasurementData
{
    public MeasurementTool Tool;
    public Vector3 PointA;
    public Vector3 PointB;
    public Vector3 NormalA;
    public Vector3 NormalB;
    public float Distance;
    public GameObject Visualization;
}

public class MeasurementManager : MonoBehaviour
{
    public static MeasurementManager Instance { get; private set; }

    [Header("Raycast")]
    [SerializeField] private float maxRaycastDistance = 50.0f;
    [SerializeField] private LayerMask raycastLayerMask = ~0;

    [Header("Visual Feedback")]
    [SerializeField] private GameObject hitMarkerPrefab;

    public MeasurementTool ActiveTool { get; private set; } = MeasurementTool.None;
    public List<MeasurementData> Measurements { get; private set; } = new List<MeasurementData>();

    // Events for UI updates
    public event Action<MeasurementTool> OnToolChanged;
    public event Action<string> OnStatusMessage;
    public event Action<MeasurementData> OnMeasurementCompleted;
    public event Action OnMeasurementsCleared;

    private IMeasurementTool activeTool;
    private PointToPointTool pointToPointTool;
    private WallToWallTool wallToWallTool;
    private HeightTool heightTool;

    void Awake()
    {
        if (Instance != null)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;

        pointToPointTool = GetComponent<PointToPointTool>();
        wallToWallTool = GetComponent<WallToWallTool>();
        heightTool = GetComponent<HeightTool>();
    }

    void Update()
    {
        if (ActiveTool == MeasurementTool.None)
            return;

        // ESC cancels current measurement
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            CancelCurrentMeasurement();
            return;
        }

        // Left click to select a point (only when cursor is locked / in FPS mode)
        if (Input.GetMouseButtonDown(0) && Cursor.lockState == CursorLockMode.Locked)
        {
            if (TryRaycast(out RaycastHit hit))
            {
                activeTool?.OnPointSelected(hit.point, hit.normal, hit.triangleIndex, hit.collider);
            }
        }
    }

    public bool TryRaycast(out RaycastHit hit)
    {
        Camera cam = Camera.main;
        Ray ray = cam.ViewportPointToRay(new Vector3(0.5f, 0.5f, 0));
        return Physics.Raycast(ray, out hit, maxRaycastDistance, raycastLayerMask);
    }

    public void SetTool(MeasurementTool tool)
    {
        // Cancel any in-progress measurement
        activeTool?.Cancel();

        ActiveTool = tool;

        switch (tool)
        {
            case MeasurementTool.PointToPoint:
                activeTool = pointToPointTool;
                break;
            case MeasurementTool.WallToWall:
                activeTool = wallToWallTool;
                break;
            case MeasurementTool.Height:
                activeTool = heightTool;
                break;
            default:
                activeTool = null;
                break;
        }

        activeTool?.Activate();
        OnToolChanged?.Invoke(tool);
    }

    public void RegisterMeasurement(MeasurementData data)
    {
        Measurements.Add(data);
        OnMeasurementCompleted?.Invoke(data);
    }

    public void SendStatusMessage(string message)
    {
        OnStatusMessage?.Invoke(message);
    }

    public void ClearAllMeasurements()
    {
        foreach (var m in Measurements)
        {
            if (m.Visualization != null)
                Destroy(m.Visualization);
        }
        Measurements.Clear();
        OnMeasurementsCleared?.Invoke();
    }

    private void CancelCurrentMeasurement()
    {
        activeTool?.Cancel();
        SendStatusMessage($"{ActiveTool} — cancelled. Click to start a new measurement.");
    }
}
```

- [ ] **Step 2: Create the IMeasurementTool interface**

Add this interface at the top of `MeasurementManager.cs` (above the class), or in a separate file. Keeping it in the same file for simplicity:

```csharp
public interface IMeasurementTool
{
    void Activate();
    void OnPointSelected(Vector3 point, Vector3 normal, int triangleIndex, Collider collider);
    void Cancel();
}
```

- [ ] **Step 3: Attach to scene**

Create an empty GameObject named "MeasurementManager" in SampleScene. Attach `MeasurementManager`, `PointToPointTool`, `WallToWallTool`, and `HeightTool` components (all on the same GameObject so GetComponent finds them).

---

### Task 8: Measurement Renderer

**Files:**
- Create: `unity/Assets/Scripts/MeasurementRenderer.cs`

- [ ] **Step 1: Create MeasurementRenderer.cs**

```csharp
using UnityEngine;
using TMPro;

public class MeasurementRenderer : MonoBehaviour
{
    [Header("Line Settings")]
    [SerializeField] private Material lineMaterial;
    [SerializeField] private float lineWidth = 0.01f;
    [SerializeField] private Color lineColor = Color.yellow;

    [Header("Label Settings")]
    [SerializeField] private GameObject distanceLabelPrefab;

    /// <summary>
    /// Create a dashed line between two points with a distance label.
    /// Returns the root GameObject containing all visuals.
    /// </summary>
    public static GameObject CreateMeasurementVisual(Vector3 pointA, Vector3 pointB, float distance)
    {
        GameObject root = new GameObject("Measurement");

        // Create line renderer for the dashed line
        GameObject lineObj = new GameObject("Line");
        lineObj.transform.SetParent(root.transform);

        LineRenderer lr = lineObj.AddComponent<LineRenderer>();
        lr.positionCount = 2;
        lr.SetPosition(0, pointA);
        lr.SetPosition(1, pointB);
        lr.startWidth = 0.008f;
        lr.endWidth = 0.008f;
        lr.material = new Material(Shader.Find("Sprites/Default"));
        lr.startColor = Color.yellow;
        lr.endColor = Color.yellow;
        lr.useWorldSpace = true;

        // Create spheres at endpoints
        CreateEndpointMarker(pointA, root.transform);
        CreateEndpointMarker(pointB, root.transform);

        // Create distance label at midpoint
        Vector3 midpoint = (pointA + pointB) / 2f;
        CreateDistanceLabel(midpoint, distance, root.transform);

        return root;
    }

    private static void CreateEndpointMarker(Vector3 position, Transform parent)
    {
        GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        marker.transform.SetParent(parent);
        marker.transform.position = position;
        marker.transform.localScale = Vector3.one * 0.03f;

        // Remove the collider so it does not interfere with raycasts
        Object.Destroy(marker.GetComponent<Collider>());

        var renderer = marker.GetComponent<Renderer>();
        renderer.material = new Material(Shader.Find("Sprites/Default"));
        renderer.material.color = Color.red;
    }

    private static void CreateDistanceLabel(Vector3 position, float distance, Transform parent)
    {
        // Create a world-space TextMeshPro label
        GameObject labelObj = new GameObject("DistanceLabel");
        labelObj.transform.SetParent(parent);
        labelObj.transform.position = position + Vector3.up * 0.05f;

        TextMeshPro tmp = labelObj.AddComponent<TextMeshPro>();
        tmp.text = $"{distance:F3} m";
        tmp.fontSize = 2.0f;
        tmp.alignment = TextAlignmentOptions.Center;
        tmp.color = Color.white;

        // Billboard: always face the camera
        labelObj.AddComponent<BillboardLabel>();
    }
}
```

- [ ] **Step 2: Create BillboardLabel.cs helper**

Create `unity/Assets/Scripts/BillboardLabel.cs`:

```csharp
using UnityEngine;

public class BillboardLabel : MonoBehaviour
{
    void LateUpdate()
    {
        if (Camera.main != null)
        {
            transform.LookAt(Camera.main.transform);
            transform.Rotate(0, 180, 0); // face toward camera, not away
        }
    }
}
```

- [ ] **Step 3: Add TextMeshPro package**

In Unity: Window > Package Manager > Unity Registry > TextMeshPro > Install. When prompted to import TMP Essential Resources, click "Import".

---

### Task 9: Crosshair

**Files:**
- Create: `unity/Assets/Scripts/CrosshairUI.cs`

- [ ] **Step 1: Create CrosshairUI.cs**

This provides a simple center-screen crosshair so the user knows where the raycast will hit.

```csharp
using UnityEngine;
using UnityEngine.UI;

public class CrosshairUI : MonoBehaviour
{
    void Start()
    {
        // Create crosshair as a child of the Canvas
        Canvas canvas = FindObjectOfType<Canvas>();
        if (canvas == null)
            return;

        GameObject crosshairObj = new GameObject("Crosshair");
        crosshairObj.transform.SetParent(canvas.transform, false);

        Image img = crosshairObj.AddComponent<Image>();
        img.color = new Color(1f, 1f, 1f, 0.7f);

        RectTransform rt = crosshairObj.GetComponent<RectTransform>();
        rt.anchorMin = new Vector2(0.5f, 0.5f);
        rt.anchorMax = new Vector2(0.5f, 0.5f);
        rt.sizeDelta = new Vector2(4, 4);
        rt.anchoredPosition = Vector2.zero;
    }
}
```

- [ ] **Step 2: Attach to scene**

Attach `CrosshairUI` to the UICanvas GameObject.

---

## Group E: Measurement Tools

### Task 10: Point-to-Point Tool

**Files:**
- Create: `unity/Assets/Scripts/PointToPointTool.cs`

- [ ] **Step 1: Create PointToPointTool.cs**

```csharp
using UnityEngine;

public class PointToPointTool : MonoBehaviour, IMeasurementTool
{
    private enum State { WaitingForFirstPoint, WaitingForSecondPoint }

    private State state;
    private Vector3 firstPoint;
    private GameObject firstMarker;

    public void Activate()
    {
        state = State.WaitingForFirstPoint;
        MeasurementManager.Instance.SendStatusMessage(
            "Point-to-Point: Click the first surface point.");
    }

    public void OnPointSelected(Vector3 point, Vector3 normal, int triangleIndex, Collider collider)
    {
        switch (state)
        {
            case State.WaitingForFirstPoint:
                firstPoint = point;

                // Show temporary marker at first point
                firstMarker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                firstMarker.transform.position = point;
                firstMarker.transform.localScale = Vector3.one * 0.04f;
                Object.Destroy(firstMarker.GetComponent<Collider>());
                firstMarker.GetComponent<Renderer>().material = new Material(Shader.Find("Sprites/Default"));
                firstMarker.GetComponent<Renderer>().material.color = Color.cyan;

                state = State.WaitingForSecondPoint;
                MeasurementManager.Instance.SendStatusMessage(
                    "Point-to-Point: Click the second surface point.");
                break;

            case State.WaitingForSecondPoint:
                Vector3 secondPoint = point;
                float distance = Vector3.Distance(firstPoint, secondPoint);

                // Clean up temporary marker
                if (firstMarker != null)
                    Object.Destroy(firstMarker);

                // Create persistent visualization
                GameObject vis = MeasurementRenderer.CreateMeasurementVisual(
                    firstPoint, secondPoint, distance);

                // Register measurement
                var data = new MeasurementData
                {
                    Tool = MeasurementTool.PointToPoint,
                    PointA = firstPoint,
                    PointB = secondPoint,
                    Distance = distance,
                    Visualization = vis,
                };
                MeasurementManager.Instance.RegisterMeasurement(data);

                MeasurementManager.Instance.SendStatusMessage(
                    $"Point-to-Point: {distance:F3} m. Click to start a new measurement.");

                // Reset for next measurement
                state = State.WaitingForFirstPoint;
                break;
        }
    }

    public void Cancel()
    {
        state = State.WaitingForFirstPoint;
        if (firstMarker != null)
            Object.Destroy(firstMarker);
    }
}
```

- [ ] **Step 2: Test point-to-point measurement**

Enter Play mode, select the Point-to-Point tool (via ToolbarUI once built, or by temporarily calling `MeasurementManager.Instance.SetTool(MeasurementTool.PointToPoint)` from a test script). Click two points on the mesh surface. Verify:
- Yellow line appears between the two points
- Red endpoint spheres appear
- Distance label shows a reasonable metric value (e.g., ~2m for a door width)
- Label faces the camera (billboard)

---

### Task 11: Wall-to-Wall Tool

**Files:**
- Create: `unity/Assets/Scripts/WallToWallTool.cs`

- [ ] **Step 1: Create WallToWallTool.cs**

```csharp
using UnityEngine;

public class WallToWallTool : MonoBehaviour, IMeasurementTool
{
    [Header("Plane Fitting")]
    [SerializeField] private float planeFitRadius = 0.3f;
    [SerializeField] private float maxOppositeWallDistance = 20.0f;

    public void Activate()
    {
        MeasurementManager.Instance.SendStatusMessage(
            "Wall-to-Wall: Click a wall surface to measure perpendicular distance to opposite wall.");
    }

    public void OnPointSelected(Vector3 point, Vector3 normal, int triangleIndex, Collider collider)
    {
        // Use the hit normal as the wall plane normal.
        // For mesh colliders, the triangle normal is already a good approximation.
        Vector3 wallNormal = normal.normalized;

        // Check that this is roughly a vertical surface (wall, not floor/ceiling)
        float verticalDot = Mathf.Abs(Vector3.Dot(wallNormal, Vector3.up));
        if (verticalDot > 0.5f)
        {
            MeasurementManager.Instance.SendStatusMessage(
                "Wall-to-Wall: Please click a vertical wall surface, not a floor or ceiling.");
            return;
        }

        // Snap normal to the nearest cardinal horizontal direction (Manhattan assumption)
        wallNormal = SnapToCardinalHorizontal(wallNormal);

        // Raycast in the opposite direction of the wall normal to find the opposite wall
        Ray oppositeRay = new Ray(point, -wallNormal);
        if (!Physics.Raycast(oppositeRay, out RaycastHit oppositeHit, maxOppositeWallDistance))
        {
            MeasurementManager.Instance.SendStatusMessage(
                "Wall-to-Wall: Could not find opposite wall. Try clicking a different surface.");
            return;
        }

        // Compute perpendicular distance: project the displacement onto the wall normal
        Vector3 displacement = oppositeHit.point - point;
        float perpendicularDistance = Mathf.Abs(Vector3.Dot(displacement, wallNormal));

        // The measurement line goes from the clicked point to the closest point on
        // the opposite wall along the normal direction
        Vector3 projectedPoint = point - wallNormal * perpendicularDistance;

        // Create visualization
        GameObject vis = MeasurementRenderer.CreateMeasurementVisual(
            point, oppositeHit.point, perpendicularDistance);

        // Register measurement
        var data = new MeasurementData
        {
            Tool = MeasurementTool.WallToWall,
            PointA = point,
            PointB = oppositeHit.point,
            NormalA = wallNormal,
            Distance = perpendicularDistance,
            Visualization = vis,
        };
        MeasurementManager.Instance.RegisterMeasurement(data);

        MeasurementManager.Instance.SendStatusMessage(
            $"Wall-to-Wall: {perpendicularDistance:F3} m. Click another wall to measure again.");
    }

    public void Cancel()
    {
        // Single-click tool, nothing to cancel
    }

    private Vector3 SnapToCardinalHorizontal(Vector3 normal)
    {
        // Project to XZ plane
        normal.y = 0;
        normal.Normalize();

        // Snap to nearest 90-degree direction
        float absX = Mathf.Abs(normal.x);
        float absZ = Mathf.Abs(normal.z);

        if (absX > absZ)
            return new Vector3(Mathf.Sign(normal.x), 0, 0);
        else
            return new Vector3(0, 0, Mathf.Sign(normal.z));
    }
}
```

- [ ] **Step 2: Test wall-to-wall measurement**

Enter Play mode, select Wall-to-Wall tool, click a wall in the corridor. Verify:
- The opposite wall is found via raycast
- Distance label shows a reasonable corridor width (e.g., ~1.5-2.5m)
- Clicking floor/ceiling shows a helpful error message instead

---

### Task 12: Height Tool

**Files:**
- Create: `unity/Assets/Scripts/HeightTool.cs`

- [ ] **Step 1: Create HeightTool.cs**

```csharp
using UnityEngine;

public class HeightTool : MonoBehaviour, IMeasurementTool
{
    [Header("Settings")]
    [SerializeField] private float maxVerticalDistance = 10.0f;

    public void Activate()
    {
        MeasurementManager.Instance.SendStatusMessage(
            "Height: Click a floor or ceiling surface to measure room height.");
    }

    public void OnPointSelected(Vector3 point, Vector3 normal, int triangleIndex, Collider collider)
    {
        Vector3 surfaceNormal = normal.normalized;

        // Determine if this is a floor (normal pointing up) or ceiling (normal pointing down)
        float upDot = Vector3.Dot(surfaceNormal, Vector3.up);

        Vector3 rayDirection;
        string surfaceType;

        if (upDot > 0.5f)
        {
            // Floor: cast up to find ceiling
            rayDirection = Vector3.up;
            surfaceType = "floor";
        }
        else if (upDot < -0.5f)
        {
            // Ceiling: cast down to find floor
            rayDirection = Vector3.down;
            surfaceType = "ceiling";
        }
        else
        {
            MeasurementManager.Instance.SendStatusMessage(
                "Height: Please click a horizontal surface (floor or ceiling), not a wall.");
            return;
        }

        // Raycast to find the opposite horizontal surface
        Ray verticalRay = new Ray(point, rayDirection);
        if (!Physics.Raycast(verticalRay, out RaycastHit oppositeHit, maxVerticalDistance))
        {
            MeasurementManager.Instance.SendStatusMessage(
                $"Height: Could not find opposite surface above/below the {surfaceType}.");
            return;
        }

        float height = Mathf.Abs(oppositeHit.point.y - point.y);

        // Create visualization (vertical line)
        Vector3 bottomPoint = (rayDirection == Vector3.up) ? point : oppositeHit.point;
        Vector3 topPoint = (rayDirection == Vector3.up) ? oppositeHit.point : point;

        GameObject vis = MeasurementRenderer.CreateMeasurementVisual(
            bottomPoint, topPoint, height);

        // Register measurement
        var data = new MeasurementData
        {
            Tool = MeasurementTool.Height,
            PointA = bottomPoint,
            PointB = topPoint,
            Distance = height,
            Visualization = vis,
        };
        MeasurementManager.Instance.RegisterMeasurement(data);

        MeasurementManager.Instance.SendStatusMessage(
            $"Height: {height:F3} m ({surfaceType} to opposite surface). Click again to measure elsewhere.");
    }

    public void Cancel()
    {
        // Single-click tool, nothing to cancel
    }
}
```

- [ ] **Step 2: Test height measurement**

Enter Play mode, select Height tool, click the floor. Verify:
- A vertical line extends from floor to ceiling
- Distance label shows a reasonable room height (e.g., ~2.5-3.5m for an office)
- Clicking a wall shows a helpful error message

---

## Group F: Toolbar and Status Bar UI

### Task 13: Toolbar UI

**Files:**
- Create: `unity/Assets/Scripts/ToolbarUI.cs`

- [ ] **Step 1: Create ToolbarUI.cs**

```csharp
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class ToolbarUI : MonoBehaviour
{
    [Header("Colors")]
    [SerializeField] private Color normalColor = new Color(0.2f, 0.2f, 0.2f, 0.9f);
    [SerializeField] private Color activeColor = new Color(0.1f, 0.5f, 0.9f, 0.9f);
    [SerializeField] private Color hoverColor = new Color(0.35f, 0.35f, 0.35f, 0.9f);

    private Button btnPointToPoint;
    private Button btnWallToWall;
    private Button btnHeight;
    private Button btnClear;
    private TextMeshProUGUI statusText;

    private Button[] toolButtons;
    private MeasurementTool[] toolMapping;

    void Start()
    {
        Canvas canvas = FindObjectOfType<Canvas>();
        if (canvas == null)
            return;

        BuildToolbar(canvas.transform);

        // Subscribe to events
        var mm = MeasurementManager.Instance;
        if (mm != null)
        {
            mm.OnToolChanged += UpdateToolHighlight;
            mm.OnStatusMessage += UpdateStatusText;
            mm.OnMeasurementCompleted += OnMeasurementCompleted;
        }
    }

    private void BuildToolbar(Transform canvasTransform)
    {
        // ---- Top toolbar panel ----
        GameObject toolbarPanel = CreatePanel(canvasTransform, "ToolbarPanel",
            new Vector2(0, 1), new Vector2(1, 1),   // anchor top-stretch
            new Vector2(0, -25), new Vector2(0, -25), // offset: 50px tall at top
            new Vector2(0, 25),
            new Color(0.12f, 0.12f, 0.12f, 0.95f));

        RectTransform toolbarRT = toolbarPanel.GetComponent<RectTransform>();
        toolbarRT.sizeDelta = new Vector2(0, 50);

        // Horizontal layout for buttons
        HorizontalLayoutGroup hlg = toolbarPanel.AddComponent<HorizontalLayoutGroup>();
        hlg.spacing = 8;
        hlg.padding = new RectOffset(10, 10, 5, 5);
        hlg.childAlignment = TextAnchor.MiddleLeft;
        hlg.childForceExpandWidth = false;
        hlg.childForceExpandHeight = true;

        // Tool buttons
        btnPointToPoint = CreateToolButton(toolbarPanel.transform, "Point-to-Point  [1]", 170);
        btnWallToWall = CreateToolButton(toolbarPanel.transform, "Wall-to-Wall  [2]", 160);
        btnHeight = CreateToolButton(toolbarPanel.transform, "Height  [3]", 120);

        // Spacer
        GameObject spacer = new GameObject("Spacer");
        spacer.transform.SetParent(toolbarPanel.transform, false);
        LayoutElement spacerLE = spacer.AddComponent<LayoutElement>();
        spacerLE.flexibleWidth = 1;

        // Clear button
        btnClear = CreateToolButton(toolbarPanel.transform, "Clear All", 110);
        btnClear.GetComponent<Image>().color = new Color(0.5f, 0.15f, 0.15f, 0.9f);

        toolButtons = new Button[] { btnPointToPoint, btnWallToWall, btnHeight };
        toolMapping = new MeasurementTool[]
        {
            MeasurementTool.PointToPoint,
            MeasurementTool.WallToWall,
            MeasurementTool.Height
        };

        // Button click handlers
        btnPointToPoint.onClick.AddListener(() => SelectTool(MeasurementTool.PointToPoint));
        btnWallToWall.onClick.AddListener(() => SelectTool(MeasurementTool.WallToWall));
        btnHeight.onClick.AddListener(() => SelectTool(MeasurementTool.Height));
        btnClear.onClick.AddListener(() => MeasurementManager.Instance?.ClearAllMeasurements());

        // ---- Bottom status bar ----
        GameObject statusPanel = CreatePanel(canvasTransform, "StatusPanel",
            new Vector2(0, 0), new Vector2(1, 0),
            new Vector2(0, 15), new Vector2(0, 15),
            new Vector2(0, 15),
            new Color(0.1f, 0.1f, 0.1f, 0.85f));

        RectTransform statusRT = statusPanel.GetComponent<RectTransform>();
        statusRT.sizeDelta = new Vector2(0, 30);

        GameObject statusTextObj = new GameObject("StatusText");
        statusTextObj.transform.SetParent(statusPanel.transform, false);

        statusText = statusTextObj.AddComponent<TextMeshProUGUI>();
        statusText.text = "Select a measurement tool from the toolbar above.";
        statusText.fontSize = 14;
        statusText.color = Color.white;
        statusText.alignment = TextAlignmentOptions.MidlineLeft;

        RectTransform stRT = statusTextObj.GetComponent<RectTransform>();
        stRT.anchorMin = Vector2.zero;
        stRT.anchorMax = Vector2.one;
        stRT.offsetMin = new Vector2(15, 0);
        stRT.offsetMax = new Vector2(-15, 0);
    }

    void Update()
    {
        // Keyboard shortcuts
        if (Input.GetKeyDown(KeyCode.Alpha1))
            SelectTool(MeasurementTool.PointToPoint);
        if (Input.GetKeyDown(KeyCode.Alpha2))
            SelectTool(MeasurementTool.WallToWall);
        if (Input.GetKeyDown(KeyCode.Alpha3))
            SelectTool(MeasurementTool.Height);
        if (Input.GetKeyDown(KeyCode.Delete) || Input.GetKeyDown(KeyCode.Backspace))
            MeasurementManager.Instance?.ClearAllMeasurements();
    }

    private void SelectTool(MeasurementTool tool)
    {
        // Toggle: clicking the active tool deselects it
        if (MeasurementManager.Instance.ActiveTool == tool)
            MeasurementManager.Instance.SetTool(MeasurementTool.None);
        else
            MeasurementManager.Instance.SetTool(tool);
    }

    private void UpdateToolHighlight(MeasurementTool activeTool)
    {
        for (int i = 0; i < toolButtons.Length; i++)
        {
            Image img = toolButtons[i].GetComponent<Image>();
            img.color = (toolMapping[i] == activeTool) ? activeColor : normalColor;
        }

        if (activeTool == MeasurementTool.None)
            statusText.text = "Select a measurement tool from the toolbar above.";
    }

    private void UpdateStatusText(string message)
    {
        statusText.text = message;
    }

    private void OnMeasurementCompleted(MeasurementData data)
    {
        // Status text is already updated by the tool
    }

    // ---- UI Factory Helpers ----

    private GameObject CreatePanel(Transform parent, string name,
        Vector2 anchorMin, Vector2 anchorMax,
        Vector2 offsetMin, Vector2 offsetMax,
        Vector2 sizeDelta,
        Color bgColor)
    {
        GameObject panel = new GameObject(name);
        panel.transform.SetParent(parent, false);

        Image bg = panel.AddComponent<Image>();
        bg.color = bgColor;

        RectTransform rt = panel.GetComponent<RectTransform>();
        rt.anchorMin = anchorMin;
        rt.anchorMax = anchorMax;
        rt.pivot = new Vector2(0.5f, anchorMin.y > 0.5f ? 1 : 0);
        rt.offsetMin = offsetMin;
        rt.offsetMax = offsetMax;
        rt.sizeDelta = sizeDelta;

        return panel;
    }

    private Button CreateToolButton(Transform parent, string label, float width)
    {
        GameObject btnObj = new GameObject(label.Replace(" ", ""));
        btnObj.transform.SetParent(parent, false);

        Image img = btnObj.AddComponent<Image>();
        img.color = normalColor;

        Button btn = btnObj.AddComponent<Button>();
        ColorBlock cb = btn.colors;
        cb.normalColor = Color.white;
        cb.highlightedColor = new Color(1.3f, 1.3f, 1.3f);
        cb.pressedColor = new Color(0.8f, 0.8f, 0.8f);
        btn.colors = cb;

        LayoutElement le = btnObj.AddComponent<LayoutElement>();
        le.preferredWidth = width;
        le.preferredHeight = 36;

        // Label
        GameObject textObj = new GameObject("Label");
        textObj.transform.SetParent(btnObj.transform, false);

        TextMeshProUGUI tmp = textObj.AddComponent<TextMeshProUGUI>();
        tmp.text = label;
        tmp.fontSize = 14;
        tmp.color = Color.white;
        tmp.alignment = TextAlignmentOptions.Center;
        tmp.fontStyle = FontStyles.Bold;

        RectTransform trt = textObj.GetComponent<RectTransform>();
        trt.anchorMin = Vector2.zero;
        trt.anchorMax = Vector2.one;
        trt.offsetMin = Vector2.zero;
        trt.offsetMax = Vector2.zero;

        return btn;
    }
}
```

- [ ] **Step 2: Attach to scene**

Attach `ToolbarUI` to the UICanvas GameObject.

- [ ] **Step 3: Test toolbar**

Enter Play mode. Verify:
- Top toolbar shows three tool buttons + Clear All
- Clicking a button highlights it blue and shows a status message
- Clicking the active button again deselects it
- Pressing 1/2/3 on keyboard selects the corresponding tool
- Delete/Backspace clears all measurements
- Alt unlocks cursor to click toolbar buttons, Alt again re-locks

---

## Group G: Minimap

### Task 14: Minimap Controller

**Files:**
- Create: `unity/Assets/Scripts/MinimapController.cs`

- [ ] **Step 1: Create MinimapController.cs**

```csharp
using UnityEngine;
using UnityEngine.UI;
using System.IO;

public class MinimapController : MonoBehaviour
{
    [Header("Minimap Size")]
    [SerializeField] private int minimapSize = 220;
    [SerializeField] private int margin = 15;

    [Header("References")]
    [SerializeField] private Transform playerTransform;

    [Header("Density Image Metadata")]
    [Tooltip("Matches metadata.json from generate_density_image.py")]
    private float minX, minZ, maxDim;
    private float offsetX, offsetZ;
    private int imageWidth, imageHeight;

    // Rotation matrix from raw point cloud frame to Manhattan-aligned frame.
    // The density image is rendered in the aligned frame, but the player moves
    // in the raw point cloud frame. generate_density_image.py outputs this
    // rotation_matrix (3x3) alongside min_coords and resolution_mm.
    private Matrix4x4 rawToAlignedRotation = Matrix4x4.identity;

    private Camera minimapCamera;
    private RawImage minimapImage;
    private RectTransform playerDot;
    private RectTransform directionArrow;
    private RenderTexture minimapRT;
    private GameObject minimapRoot;
    private bool minimapActive;

    void Start()
    {
        string minimapPath = AppBootstrap.MinimapPath;

        if (string.IsNullOrEmpty(minimapPath) || !File.Exists(minimapPath))
        {
            Debug.Log("No minimap density image provided. Minimap hidden.");
            return;
        }

        // Load the density image
        byte[] imageData = File.ReadAllBytes(minimapPath);
        Texture2D densityTexture = new Texture2D(2, 2);
        if (!densityTexture.LoadImage(imageData))
        {
            Debug.LogError("Failed to load minimap density image.");
            return;
        }

        // Load metadata (same directory as the density image)
        string metadataPath = Path.Combine(Path.GetDirectoryName(minimapPath), "metadata.json");
        if (!LoadMetadata(metadataPath))
        {
            Debug.LogWarning("No density image metadata found. Using default mapping.");
        }

        BuildMinimap(densityTexture);
        minimapActive = true;
    }

    private bool LoadMetadata(string path)
    {
        if (!File.Exists(path))
            return false;

        string json = File.ReadAllText(path);
        var meta = JsonUtility.FromJson<DensityImageMetadata>(json);

        // The density image maps world XZ coordinates to pixel coordinates.
        // metadata.min_coords are in millimeters; convert to meters.
        minX = meta.min_coords[0] / 1000f;
        minZ = meta.min_coords[1] / 1000f;
        maxDim = meta.max_dim / 1000f;
        offsetX = meta.offset[0] / 1000f;
        offsetZ = meta.offset[1] / 1000f;
        imageWidth = meta.image_width;
        imageHeight = meta.image_height;

        // Load rotation matrix (3x3, row-major flattened) from density image metadata.
        // generate_density_image.py outputs this as "rotation_matrix": [r00, r01, r02, ...]
        // It rotates raw point cloud XZ coordinates into the Manhattan-aligned frame
        // used by the density image.
        if (meta.rotation_matrix != null && meta.rotation_matrix.Length == 9)
        {
            rawToAlignedRotation = Matrix4x4.identity;
            rawToAlignedRotation[0, 0] = meta.rotation_matrix[0];
            rawToAlignedRotation[0, 1] = meta.rotation_matrix[1];
            rawToAlignedRotation[0, 2] = meta.rotation_matrix[2];
            rawToAlignedRotation[1, 0] = meta.rotation_matrix[3];
            rawToAlignedRotation[1, 1] = meta.rotation_matrix[4];
            rawToAlignedRotation[1, 2] = meta.rotation_matrix[5];
            rawToAlignedRotation[2, 0] = meta.rotation_matrix[6];
            rawToAlignedRotation[2, 1] = meta.rotation_matrix[7];
            rawToAlignedRotation[2, 2] = meta.rotation_matrix[8];
            Debug.Log("Loaded density image rotation matrix for minimap coordinate mapping.");
        }

        return true;
    }

    private void BuildMinimap(Texture2D densityTexture)
    {
        Canvas canvas = FindObjectOfType<Canvas>();
        if (canvas == null)
            return;

        // Root panel in bottom-right corner
        minimapRoot = new GameObject("MinimapPanel");
        minimapRoot.transform.SetParent(canvas.transform, false);

        RectTransform rootRT = minimapRoot.AddComponent<RectTransform>();
        rootRT.anchorMin = new Vector2(1, 0);
        rootRT.anchorMax = new Vector2(1, 0);
        rootRT.pivot = new Vector2(1, 0);
        rootRT.anchoredPosition = new Vector2(-margin, margin + 35); // above status bar
        rootRT.sizeDelta = new Vector2(minimapSize, minimapSize);

        // Background
        Image bg = minimapRoot.AddComponent<Image>();
        bg.color = new Color(0, 0, 0, 0.7f);

        // Density image display
        GameObject imgObj = new GameObject("DensityImage");
        imgObj.transform.SetParent(minimapRoot.transform, false);

        minimapImage = imgObj.AddComponent<RawImage>();
        minimapImage.texture = densityTexture;

        RectTransform imgRT = imgObj.GetComponent<RectTransform>();
        imgRT.anchorMin = Vector2.zero;
        imgRT.anchorMax = Vector2.one;
        imgRT.offsetMin = new Vector2(5, 5);
        imgRT.offsetMax = new Vector2(-5, -5);

        // Player position dot
        GameObject dotObj = new GameObject("PlayerDot");
        dotObj.transform.SetParent(minimapRoot.transform, false);

        Image dotImage = dotObj.AddComponent<Image>();
        dotImage.color = Color.red;

        playerDot = dotObj.GetComponent<RectTransform>();
        playerDot.sizeDelta = new Vector2(8, 8);

        // Direction arrow (small triangle above the dot)
        GameObject arrowObj = new GameObject("DirectionArrow");
        arrowObj.transform.SetParent(dotObj.transform, false);

        Image arrowImage = arrowObj.AddComponent<Image>();
        arrowImage.color = Color.red;

        directionArrow = arrowObj.GetComponent<RectTransform>();
        directionArrow.sizeDelta = new Vector2(4, 12);
        directionArrow.anchoredPosition = new Vector2(0, 10);
    }

    void LateUpdate()
    {
        if (!minimapActive || playerTransform == null)
            return;

        UpdatePlayerPosition();
    }

    private void UpdatePlayerPosition()
    {
        // Convert player world position (X, Z) to minimap UV coordinates.
        // The density image uses: pixel = (world_mm - min_coords + offset) / max_dim * image_size
        // In meters: pixel = (world_m * 1000 - min_coords_mm + offset_mm) / max_dim_mm * image_size
        // Simplified with our pre-converted values:

        // Apply rotation from raw point cloud frame to Manhattan-aligned frame.
        // The density image is rendered in the aligned frame, but the player
        // moves in the raw point cloud frame. The rotation_matrix from
        // generate_density_image.py maps raw XZ -> aligned XZ.
        Vector3 rawPos = new Vector3(playerTransform.position.x, 0, playerTransform.position.z);
        Vector3 alignedPos = rawToAlignedRotation.MultiplyPoint3x4(rawPos);

        float worldX = alignedPos.x;
        float worldZ = alignedPos.z;

        // Convert to pixel coordinates in the density image
        float pixelX = (worldX - minX + offsetX) / maxDim * imageWidth;
        float pixelY = (worldZ - minZ + offsetZ) / maxDim * imageHeight;

        // Convert to normalized UV (0-1)
        float u = pixelX / imageWidth;
        float v = pixelY / imageHeight;

        // Map UV to minimap panel local coordinates (with 5px padding)
        float panelInner = minimapSize - 10;
        float localX = -minimapSize / 2f + 5 + u * panelInner;
        float localY = -minimapSize / 2f + 5 + v * panelInner;

        playerDot.anchoredPosition = new Vector2(
            localX + minimapSize / 2f,
            localY + minimapSize / 2f);

        // Rotate direction arrow to match player facing direction
        float yaw = playerTransform.eulerAngles.y;
        directionArrow.localEulerAngles = new Vector3(0, 0, -yaw);
    }

    [System.Serializable]
    private class DensityImageMetadata
    {
        public float[] min_coords;
        public float max_dim;
        public float[] offset;
        public int image_width;
        public int image_height;
        public float[] rotation_matrix; // 3x3 row-major, from generate_density_image.py
    }
}
```

- [ ] **Step 2: Attach to scene**

Attach `MinimapController` to a new empty GameObject named "MinimapController" in SampleScene. Drag the "Player" transform into the `playerTransform` field.

- [ ] **Step 3: Test minimap**

Set `editorMinimapPath` on AppBootstrap to:
```
/home/ruoyu/scan2measure-webframework/data/density_image/tmb_office_one_corridor_dense/tmb_office_one_corridor_dense.png
```

Enter Play mode. Verify:
- Minimap appears in the bottom-right corner
- Density image is visible
- Red dot moves as the player moves
- Arrow rotates with the player's facing direction
- When no minimap path is provided, the minimap is hidden

---

## Group H: Mesh Info Panel

### Task 15: Mesh Info Panel

**Files:**
- Create: `unity/Assets/Scripts/MeshInfoPanel.cs`

- [ ] **Step 1: Create MeshInfoPanel.cs**

```csharp
using UnityEngine;
using UnityEngine.UI;
using System.IO;

public class MeshInfoPanel : MonoBehaviour
{
    private GameObject panelRoot;

    void Start()
    {
        string metadataPath = AppBootstrap.MetadataPath;

        if (string.IsNullOrEmpty(metadataPath) || !File.Exists(metadataPath))
        {
            Debug.Log("No metadata path provided. MeshInfoPanel hidden.");
            return;
        }

        string json = File.ReadAllText(metadataPath);
        var meta = JsonUtility.FromJson<MeshMetadata>(json);

        BuildPanel(meta);
    }

    private void BuildPanel(MeshMetadata meta)
    {
        Canvas canvas = FindObjectOfType<Canvas>();
        if (canvas == null)
            return;

        panelRoot = new GameObject("MeshInfoPanel");
        panelRoot.transform.SetParent(canvas.transform, false);

        // Position at bottom-left, above the status bar
        RectTransform rootRT = panelRoot.AddComponent<RectTransform>();
        rootRT.anchorMin = new Vector2(0, 0);
        rootRT.anchorMax = new Vector2(0, 0);
        rootRT.pivot = new Vector2(0, 0);
        rootRT.anchoredPosition = new Vector2(15, 40); // above 30px status bar + margin

        // Semi-transparent background
        Image bg = panelRoot.AddComponent<Image>();
        bg.color = new Color(0, 0, 0, 0.5f);

        // Format triangle count (e.g. 500000 -> "500K")
        string triString = meta.total_triangles >= 1000
            ? $"{meta.total_triangles / 1000}K"
            : meta.total_triangles.ToString();

        // Info text
        GameObject textObj = new GameObject("InfoText");
        textObj.transform.SetParent(panelRoot.transform, false);

        Text infoText = textObj.AddComponent<Text>();
        infoText.text = $"{meta.point_cloud_name} | {meta.quality_tier} | {triString} tris";
        infoText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        infoText.fontSize = 13;
        infoText.color = new Color(0.85f, 0.85f, 0.85f, 1f);
        infoText.horizontalOverflow = HorizontalWrapMode.Overflow;
        infoText.verticalOverflow = VerticalWrapMode.Overflow;

        RectTransform textRT = textObj.GetComponent<RectTransform>();
        textRT.anchorMin = Vector2.zero;
        textRT.anchorMax = Vector2.one;
        textRT.offsetMin = new Vector2(8, 2);
        textRT.offsetMax = new Vector2(-8, -2);

        // Size the panel to fit the text
        ContentSizeFitter fitter = panelRoot.AddComponent<ContentSizeFitter>();
        fitter.horizontalFit = ContentSizeFitter.FitMode.PreferredSize;
        fitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;

        LayoutElement le = panelRoot.AddComponent<LayoutElement>();
        le.minHeight = 24;

        HorizontalLayoutGroup hlg = panelRoot.AddComponent<HorizontalLayoutGroup>();
        hlg.padding = new RectOffset(8, 8, 2, 2);
        hlg.childAlignment = TextAnchor.MiddleLeft;
    }

    [System.Serializable]
    private class MeshMetadata
    {
        public string quality_tier;
        public string point_cloud_name;
        public int total_triangles;
        public int atlas_resolution;
    }
}
```

- [ ] **Step 2: Attach to scene**

Attach `MeshInfoPanel` to the UICanvas GameObject. It will read `AppBootstrap.MetadataPath` on Start and either display the info overlay or hide itself if no metadata path was provided.

- [ ] **Step 3: Test mesh info panel**

Set `--metadata` on the command line (or add an `editorMetadataPath` fallback for testing) pointing to a metadata JSON file, e.g.:
```json
{"quality_tier": "balanced", "point_cloud_name": "tmb_office_one_corridor_dense", "total_triangles": 500000, "atlas_resolution": 4096}
```

Enter Play mode. Verify:
- A small text overlay appears at the bottom-left: "tmb_office_one_corridor_dense | balanced | 500K tris"
- When no `--metadata` argument is given, the panel does not appear
- Text is readable against both light and dark backgrounds (semi-transparent bg)

---

## Group I: Windows Build

### Task 16: Build Settings

**Files:**
- Configure via Unity Editor

- [ ] **Step 1: Configure build settings**

1. File > Build Settings
2. Platform: PC, Mac & Linux Standalone (should be default)
3. Target Platform: Windows
4. Architecture: x86_64
5. Add "Scenes/SampleScene" to the build
6. Click "Player Settings":
   - Product Name: "VirtualTour"
   - Company Name: "scan2measure"
   - Default Screen Width: 1920
   - Default Screen Height: 1080
   - Fullscreen Mode: Windowed
   - Resizable Window: checked
   - Run In Background: checked

- [ ] **Step 2: Build the .exe**

1. In Build Settings, set the build output folder to `unity/Build/`
2. Click "Build"
3. The output will be `unity/Build/VirtualTour.exe`

- [ ] **Step 3: Test the build**

Run from command line:
```bash
unity/Build/VirtualTour.exe --glb "C:\path\to\tmb_office_one_corridor_dense_noRGB_textured.glb" --minimap "C:\path\to\tmb_office_one_corridor_dense.png"
```

Verify:
- App launches and loads the GLB
- Navigation works (WASD + mouse)
- Measurement tools work (toolbar, point-to-point, wall-to-wall, height)
- Minimap shows density image with player position
- Window title shows "VirtualTour"

---

## Summary

| Group | Tasks | Scripts Created |
|-------|-------|-----------------|
| A: Setup | 1-3 | (project + scene config) |
| B: GLB Loading | 4-5 | `AppBootstrap.cs`, `GLBLoader.cs` |
| C: Navigation | 6 | `FirstPersonController.cs` |
| D: Measurement Core | 7-9 | `MeasurementManager.cs`, `MeasurementRenderer.cs`, `BillboardLabel.cs`, `CrosshairUI.cs` |
| E: Measurement Tools | 10-12 | `PointToPointTool.cs`, `WallToWallTool.cs`, `HeightTool.cs` |
| F: Toolbar UI | 13 | `ToolbarUI.cs` |
| G: Minimap | 14 | `MinimapController.cs` |
| H: Mesh Info Panel | 15 | `MeshInfoPanel.cs` |
| I: Build | 16 | (build config) |

**Total: 16 tasks, ~12 C# files**

**Recommended implementation order:** A (setup) > B (loading) > C (navigation) > D (measurement core) > E (tools) > F (toolbar) > G (minimap) > H (info panel) > I (build). Each group can be tested independently before moving to the next.
