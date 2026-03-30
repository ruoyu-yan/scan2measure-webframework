using UnityEngine;
using System;
using System.Collections.Generic;

public interface IMeasurementTool
{
    void Activate();
    void OnPointSelected(Vector3 point, Vector3 normal, int triangleIndex, Collider collider);
    void Cancel();
}

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
