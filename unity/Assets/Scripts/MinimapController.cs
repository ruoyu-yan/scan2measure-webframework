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
