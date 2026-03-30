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
        lr.material = CreateOverlayMaterial(Color.yellow);
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
        renderer.material = CreateOverlayMaterial(Color.red);
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
        // Use TMP overlay shader so text renders on top of geometry.
        // Copy existing material (preserves font atlas) and swap shader.
        Shader overlayShader = Shader.Find("TextMeshPro/Distance Field Overlay");
        if (overlayShader != null)
        {
            Material overlayMat = new Material(tmp.fontMaterial);
            overlayMat.shader = overlayShader;
            tmp.fontMaterial = overlayMat;
        }

        // Billboard: always face the camera
        labelObj.AddComponent<BillboardLabel>();
    }

    private static Material CreateOverlayMaterial(Color color)
    {
        // Custom shader with ZTest Always — built-in shaders don't expose this property
        Shader shader = Shader.Find("Custom/OverlayUnlit")
            ?? Shader.Find("Sprites/Default");
        Material mat = new Material(shader);
        mat.color = color;
        return mat;
    }
}
