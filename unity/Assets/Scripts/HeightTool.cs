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
