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
                Shader shader = Shader.Find("Universal Render Pipeline/Unlit")
                    ?? Shader.Find("Sprites/Default")
                    ?? Shader.Find("Unlit/Color");
                firstMarker.GetComponent<Renderer>().material = new Material(shader);
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
