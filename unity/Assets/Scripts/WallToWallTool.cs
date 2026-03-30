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

        // Raycast across the room to find the opposite wall.
        // Normals point inward (into room), so raycast along +wallNormal.
        // Offset origin slightly to avoid self-intersection with the clicked surface.
        Ray oppositeRay = new Ray(point + wallNormal * 0.02f, wallNormal);
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
