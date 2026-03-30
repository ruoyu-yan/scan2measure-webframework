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
        Font font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        if (font == null)
            font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        infoText.font = font;
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
