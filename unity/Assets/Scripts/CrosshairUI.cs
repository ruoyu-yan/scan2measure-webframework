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
