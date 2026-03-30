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
