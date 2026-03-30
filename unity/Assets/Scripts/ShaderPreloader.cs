using UnityEngine;

/// <summary>
/// Holds references to shaders that are loaded at runtime via Shader.Find().
/// Without direct references, Unity strips them from builds.
/// Drag the shaders from the Project window into these fields in the Inspector.
/// </summary>
public class ShaderPreloader : MonoBehaviour
{
    [Header("glTFast Shaders (from Packages > glTFast > Runtime > Shader)")]
    [SerializeField] private Shader gltfPbrMetallicRoughness;
    [SerializeField] private Shader gltfUnlit;

    [Header("Measurement Overlay Shaders (from Assets/Shaders and Assets/TextMesh Pro/Shaders)")]
    [SerializeField] private Shader overlayUnlit;
    [SerializeField] private Shader tmpDistanceFieldOverlay;
}
