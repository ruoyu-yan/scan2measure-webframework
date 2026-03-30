using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class ToolbarUI : MonoBehaviour
{
    [Header("Colors")]
    [SerializeField] private Color normalColor = new Color(0.2f, 0.2f, 0.2f, 0.9f);
    [SerializeField] private Color activeColor = new Color(0.1f, 0.5f, 0.9f, 0.9f);
    [SerializeField] private Color hoverColor = new Color(0.35f, 0.35f, 0.35f, 0.9f);

    private Button btnPointToPoint;
    private Button btnWallToWall;
    private Button btnHeight;
    private Button btnClear;
    private TextMeshProUGUI statusText;

    private Button[] toolButtons;
    private MeasurementTool[] toolMapping;

    void Start()
    {
        Canvas canvas = FindObjectOfType<Canvas>();
        if (canvas == null)
            return;

        BuildToolbar(canvas.transform);

        // Subscribe to events
        var mm = MeasurementManager.Instance;
        if (mm != null)
        {
            mm.OnToolChanged += UpdateToolHighlight;
            mm.OnStatusMessage += UpdateStatusText;
            mm.OnMeasurementCompleted += OnMeasurementCompleted;
        }
    }

    private void BuildToolbar(Transform canvasTransform)
    {
        // ---- Top toolbar panel ----
        GameObject toolbarPanel = CreatePanel(canvasTransform, "ToolbarPanel",
            new Vector2(0, 1), new Vector2(1, 1),   // anchor top-stretch
            new Vector2(0, -25), new Vector2(0, -25), // offset: 50px tall at top
            new Vector2(0, 25),
            new Color(0.12f, 0.12f, 0.12f, 0.95f));

        RectTransform toolbarRT = toolbarPanel.GetComponent<RectTransform>();
        toolbarRT.sizeDelta = new Vector2(0, 50);

        // Horizontal layout for buttons
        HorizontalLayoutGroup hlg = toolbarPanel.AddComponent<HorizontalLayoutGroup>();
        hlg.spacing = 8;
        hlg.padding = new RectOffset(10, 10, 5, 5);
        hlg.childAlignment = TextAnchor.MiddleLeft;
        hlg.childForceExpandWidth = false;
        hlg.childForceExpandHeight = true;

        // Tool buttons
        btnPointToPoint = CreateToolButton(toolbarPanel.transform, "Point-to-Point  [1]", 170);
        btnWallToWall = CreateToolButton(toolbarPanel.transform, "Wall-to-Wall  [2]", 160);
        btnHeight = CreateToolButton(toolbarPanel.transform, "Height  [3]", 120);

        // Spacer
        GameObject spacer = new GameObject("Spacer");
        spacer.transform.SetParent(toolbarPanel.transform, false);
        LayoutElement spacerLE = spacer.AddComponent<LayoutElement>();
        spacerLE.flexibleWidth = 1;

        // Clear button
        btnClear = CreateToolButton(toolbarPanel.transform, "Clear All", 110);
        btnClear.GetComponent<Image>().color = new Color(0.5f, 0.15f, 0.15f, 0.9f);

        toolButtons = new Button[] { btnPointToPoint, btnWallToWall, btnHeight };
        toolMapping = new MeasurementTool[]
        {
            MeasurementTool.PointToPoint,
            MeasurementTool.WallToWall,
            MeasurementTool.Height
        };

        // Button click handlers
        btnPointToPoint.onClick.AddListener(() => SelectTool(MeasurementTool.PointToPoint));
        btnWallToWall.onClick.AddListener(() => SelectTool(MeasurementTool.WallToWall));
        btnHeight.onClick.AddListener(() => SelectTool(MeasurementTool.Height));
        btnClear.onClick.AddListener(() => MeasurementManager.Instance?.ClearAllMeasurements());

        // ---- Bottom status bar ----
        GameObject statusPanel = CreatePanel(canvasTransform, "StatusPanel",
            new Vector2(0, 0), new Vector2(1, 0),
            new Vector2(0, 15), new Vector2(0, 15),
            new Vector2(0, 15),
            new Color(0.1f, 0.1f, 0.1f, 0.85f));

        RectTransform statusRT = statusPanel.GetComponent<RectTransform>();
        statusRT.sizeDelta = new Vector2(0, 30);

        GameObject statusTextObj = new GameObject("StatusText");
        statusTextObj.transform.SetParent(statusPanel.transform, false);

        statusText = statusTextObj.AddComponent<TextMeshProUGUI>();
        statusText.text = "Select a measurement tool from the toolbar above.";
        statusText.fontSize = 14;
        statusText.color = Color.white;
        statusText.alignment = TextAlignmentOptions.MidlineLeft;

        RectTransform stRT = statusTextObj.GetComponent<RectTransform>();
        stRT.anchorMin = Vector2.zero;
        stRT.anchorMax = Vector2.one;
        stRT.offsetMin = new Vector2(15, 0);
        stRT.offsetMax = new Vector2(-15, 0);
    }

    void Update()
    {
        // Keyboard shortcuts
        if (Input.GetKeyDown(KeyCode.Alpha1))
            SelectTool(MeasurementTool.PointToPoint);
        if (Input.GetKeyDown(KeyCode.Alpha2))
            SelectTool(MeasurementTool.WallToWall);
        if (Input.GetKeyDown(KeyCode.Alpha3))
            SelectTool(MeasurementTool.Height);
        if (Input.GetKeyDown(KeyCode.Delete) || Input.GetKeyDown(KeyCode.Backspace))
            MeasurementManager.Instance?.ClearAllMeasurements();
    }

    private void SelectTool(MeasurementTool tool)
    {
        // Toggle: clicking the active tool deselects it
        if (MeasurementManager.Instance.ActiveTool == tool)
            MeasurementManager.Instance.SetTool(MeasurementTool.None);
        else
            MeasurementManager.Instance.SetTool(tool);
    }

    private void UpdateToolHighlight(MeasurementTool activeTool)
    {
        for (int i = 0; i < toolButtons.Length; i++)
        {
            Image img = toolButtons[i].GetComponent<Image>();
            img.color = (toolMapping[i] == activeTool) ? activeColor : normalColor;
        }

        if (activeTool == MeasurementTool.None)
            statusText.text = "Select a measurement tool from the toolbar above.";
    }

    private void UpdateStatusText(string message)
    {
        statusText.text = message;
    }

    private void OnMeasurementCompleted(MeasurementData data)
    {
        // Status text is already updated by the tool
    }

    // ---- UI Factory Helpers ----

    private GameObject CreatePanel(Transform parent, string name,
        Vector2 anchorMin, Vector2 anchorMax,
        Vector2 offsetMin, Vector2 offsetMax,
        Vector2 sizeDelta,
        Color bgColor)
    {
        GameObject panel = new GameObject(name);
        panel.transform.SetParent(parent, false);

        Image bg = panel.AddComponent<Image>();
        bg.color = bgColor;

        RectTransform rt = panel.GetComponent<RectTransform>();
        rt.anchorMin = anchorMin;
        rt.anchorMax = anchorMax;
        rt.pivot = new Vector2(0.5f, anchorMin.y > 0.5f ? 1 : 0);
        rt.offsetMin = offsetMin;
        rt.offsetMax = offsetMax;
        rt.sizeDelta = sizeDelta;

        return panel;
    }

    private Button CreateToolButton(Transform parent, string label, float width)
    {
        GameObject btnObj = new GameObject(label.Replace(" ", ""));
        btnObj.transform.SetParent(parent, false);

        Image img = btnObj.AddComponent<Image>();
        img.color = normalColor;

        Button btn = btnObj.AddComponent<Button>();
        ColorBlock cb = btn.colors;
        cb.normalColor = Color.white;
        cb.highlightedColor = new Color(1.3f, 1.3f, 1.3f);
        cb.pressedColor = new Color(0.8f, 0.8f, 0.8f);
        btn.colors = cb;

        LayoutElement le = btnObj.AddComponent<LayoutElement>();
        le.preferredWidth = width;
        le.preferredHeight = 36;

        // Label
        GameObject textObj = new GameObject("Label");
        textObj.transform.SetParent(btnObj.transform, false);

        TextMeshProUGUI tmp = textObj.AddComponent<TextMeshProUGUI>();
        tmp.text = label;
        tmp.fontSize = 14;
        tmp.color = Color.white;
        tmp.alignment = TextAlignmentOptions.Center;
        tmp.fontStyle = FontStyles.Bold;

        RectTransform trt = textObj.GetComponent<RectTransform>();
        trt.anchorMin = Vector2.zero;
        trt.anchorMax = Vector2.one;
        trt.offsetMin = Vector2.zero;
        trt.offsetMax = Vector2.zero;

        return btn;
    }
}
