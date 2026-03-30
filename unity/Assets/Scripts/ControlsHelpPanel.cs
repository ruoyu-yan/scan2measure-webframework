using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class ControlsHelpPanel : MonoBehaviour
{
    private TextMeshProUGUI helpText;
    private FirstPersonController fpsController;

    void Start()
    {
        Canvas canvas = FindObjectOfType<Canvas>();
        if (canvas == null) return;

        fpsController = FindObjectOfType<FirstPersonController>();
        if (fpsController == null) return;

        BuildPanel(canvas.transform);
        fpsController.OnFlyModeChanged += OnModeChanged;
        UpdateHelpText(false);
    }

    void OnDestroy()
    {
        if (fpsController != null)
            fpsController.OnFlyModeChanged -= OnModeChanged;
    }

    private void OnModeChanged(bool isFlyMode)
    {
        UpdateHelpText(isFlyMode);
    }

    private void BuildPanel(Transform canvasTransform)
    {
        GameObject panel = new GameObject("ControlsHelpPanel");
        panel.transform.SetParent(canvasTransform, false);

        Image bg = panel.AddComponent<Image>();
        bg.color = new Color(0f, 0f, 0f, 0.5f);

        RectTransform rt = panel.GetComponent<RectTransform>();
        rt.anchorMin = new Vector2(1, 1);
        rt.anchorMax = new Vector2(1, 1);
        rt.pivot = new Vector2(1, 1);
        rt.anchoredPosition = new Vector2(-15, -60);

        ContentSizeFitter fitter = panel.AddComponent<ContentSizeFitter>();
        fitter.horizontalFit = ContentSizeFitter.FitMode.PreferredSize;
        fitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;

        VerticalLayoutGroup vlg = panel.AddComponent<VerticalLayoutGroup>();
        vlg.padding = new RectOffset(10, 10, 6, 6);
        vlg.childAlignment = TextAnchor.UpperLeft;

        GameObject textObj = new GameObject("HelpText");
        textObj.transform.SetParent(panel.transform, false);

        helpText = textObj.AddComponent<TextMeshProUGUI>();
        helpText.fontSize = 12;
        helpText.color = new Color(0.85f, 0.85f, 0.85f, 1f);
        helpText.alignment = TextAlignmentOptions.TopLeft;

        RectTransform textRT = textObj.GetComponent<RectTransform>();
        textRT.anchorMin = Vector2.zero;
        textRT.anchorMax = Vector2.one;
        textRT.offsetMin = Vector2.zero;
        textRT.offsetMax = Vector2.zero;
    }

    private void UpdateHelpText(bool isFlyMode)
    {
        if (isFlyMode)
        {
            helpText.text =
                "<b>FLY MODE</b>\n" +
                "WASD  Move (camera dir)\n" +
                "E / Q  Up / Down\n" +
                "Shift  Sprint\n" +
                "F  Switch to Walk";
        }
        else
        {
            helpText.text =
                "<b>WALK MODE</b>\n" +
                "WASD  Move\n" +
                "Shift  Sprint\n" +
                "F  Switch to Fly";
        }
    }
}
