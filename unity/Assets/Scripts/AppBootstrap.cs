using UnityEngine;
using System;
using System.IO;

public class AppBootstrap : MonoBehaviour
{
    public static string GLBPath { get; private set; }
    public static string MinimapPath { get; private set; }
    public static string MetadataPath { get; private set; }
    public static string CameraPosePath { get; private set; }

    [Header("Fallback paths (Editor testing only)")]
    [SerializeField] private string editorGLBPath = "";
    [SerializeField] private string editorMinimapPath = "";
    [SerializeField] private string editorCameraPosePath = "";

    void Awake()
    {
        ParseCommandLineArgs();

        // In Editor, use inspector fallback paths if no CLI args
        #if UNITY_EDITOR
        if (string.IsNullOrEmpty(GLBPath) && !string.IsNullOrEmpty(editorGLBPath))
            GLBPath = editorGLBPath;
        if (string.IsNullOrEmpty(MinimapPath) && !string.IsNullOrEmpty(editorMinimapPath))
            MinimapPath = editorMinimapPath;
        if (string.IsNullOrEmpty(CameraPosePath) && !string.IsNullOrEmpty(editorCameraPosePath))
            CameraPosePath = editorCameraPosePath;
        #endif

        if (string.IsNullOrEmpty(GLBPath))
        {
            Debug.LogError("No GLB path provided. Use --glb <path> or set editorGLBPath in Inspector.");
            #if !UNITY_EDITOR
            Application.Quit(1);
            #endif
            return;
        }

        if (!File.Exists(GLBPath))
        {
            Debug.LogError($"GLB file not found: {GLBPath}");
            #if !UNITY_EDITOR
            Application.Quit(1);
            #endif
            return;
        }

        Debug.Log($"GLB path: {GLBPath}");
        Debug.Log($"Minimap path: {(string.IsNullOrEmpty(MinimapPath) ? "(none)" : MinimapPath)}");
        Debug.Log($"Metadata path: {(string.IsNullOrEmpty(MetadataPath) ? "(none)" : MetadataPath)}");
        Debug.Log($"Camera pose path: {(string.IsNullOrEmpty(CameraPosePath) ? "(none)" : CameraPosePath)}");
    }

    private void ParseCommandLineArgs()
    {
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--glb" && i + 1 < args.Length)
                GLBPath = args[i + 1];
            else if (args[i] == "--minimap" && i + 1 < args.Length)
                MinimapPath = args[i + 1];
            else if (args[i] == "--metadata" && i + 1 < args.Length)
                MetadataPath = args[i + 1];
            else if (args[i] == "--camera-pose" && i + 1 < args.Length)
                CameraPosePath = args[i + 1];
        }
    }
}
