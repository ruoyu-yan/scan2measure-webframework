using UnityEngine;
using GLTFast;
using System.IO;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

public class GLBLoader : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private Transform playerTransform;

    private GameObject loadedRoot;

    async void Start()
    {
        string glbPath = AppBootstrap.GLBPath;
        if (string.IsNullOrEmpty(glbPath))
            return;

        await LoadGLB(glbPath);
    }

    private async Task LoadGLB(string path)
    {
        Debug.Log($"Loading GLB: {path}");

        var gltf = new GltfImport();
        byte[] data = File.ReadAllBytes(path);
        bool success = await gltf.LoadGltfBinary(data);

        if (!success)
        {
            Debug.LogError("Failed to load GLB file.");
            return;
        }

        loadedRoot = new GameObject("LoadedMesh");
        await gltf.InstantiateMainSceneAsync(loadedRoot.transform);
        loadedRoot.transform.rotation = Quaternion.Euler(-90f, 0f, 0f);
        await Task.Yield(); // wait one frame for transforms to update

        // Fix triangle winding — glTFast X-negation flips winding order,
        // which inverts normals (causes backface culling holes + physics fall-through).
        // Reversing each triangle's index order restores correct winding.
        foreach (var mf in loadedRoot.GetComponentsInChildren<MeshFilter>())
        {
            Mesh mesh = mf.sharedMesh;
            for (int sub = 0; sub < mesh.subMeshCount; sub++)
            {
                int[] tris = mesh.GetTriangles(sub);
                for (int i = 0; i < tris.Length; i += 3)
                {
                    int tmp = tris[i];
                    tris[i] = tris[i + 2];
                    tris[i + 2] = tmp;
                }
                mesh.SetTriangles(tris, sub);
            }
            mesh.RecalculateNormals();
        }

        // Generate mesh colliders for all child meshes (after winding fix)
        int colliderCount = 0;
        foreach (var mf in loadedRoot.GetComponentsInChildren<MeshFilter>())
        {
            var mc = mf.gameObject.AddComponent<MeshCollider>();
            mc.sharedMesh = mf.sharedMesh;
            colliderCount++;
        }
        Debug.Log($"GLB loaded: {colliderCount} mesh collider(s) generated.");

        // Wait a frame for colliders to finish cooking before positioning
        await Task.Yield();

        // Position the player
        PositionPlayer();
    }

    private float[] ParseTranslation(string json)
    {
        // Manual regex extraction — JsonUtility can choke on nested arrays like rotation:[[...]]
        var match = Regex.Match(json, @"""translation""\s*:\s*\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]");
        if (!match.Success)
            return null;
        return new float[] {
            float.Parse(match.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture),
            float.Parse(match.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture),
            float.Parse(match.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture)
        };
    }

    private void PositionPlayer()
    {
        if (loadedRoot == null || playerTransform == null)
            return;

        // Compute combined bounds of all renderers
        Bounds combinedBounds = new Bounds();
        bool first = true;
        foreach (var renderer in loadedRoot.GetComponentsInChildren<Renderer>())
        {
            if (first)
            {
                combinedBounds = renderer.bounds;
                first = false;
            }
            else
            {
                combinedBounds.Encapsulate(renderer.bounds);
            }
        }

        if (first)
            return; // no renderers found

        float floorY = combinedBounds.min.y;
        Debug.Log($"Mesh bounds: min={combinedBounds.min}, max={combinedBounds.max}");

        // Try to spawn at camera pose from camera_pose.json
        string posePath = AppBootstrap.CameraPosePath;
        if (!string.IsNullOrEmpty(posePath) && File.Exists(posePath))
        {
            string json = File.ReadAllText(posePath);
            float[] t = ParseTranslation(json);
            if (t != null)
            {
                Debug.Log($"Parsed camera_pose.json translation: ({t[0]:F3}, {t[1]:F3}, {t[2]:F3})");

                // Point cloud coords: X,Y horizontal, Z up
                // glTFast negates X for right-to-left hand conversion
                // After mesh rotation Euler(-90,0,0): unity = (-pc_x, pc_z, -pc_y)
                float spawnX = -t[0];
                float spawnZ = -t[1];
                playerTransform.position = new Vector3(spawnX, floorY + 1.6f, spawnZ);

                Debug.Log($"Player spawned at camera pose: {playerTransform.position}");
                return;
            }
            Debug.LogWarning("camera_pose.json found but translation could not be parsed. Falling back to mesh center.");
        }

        // Fallback: center of mesh
        Vector3 center = combinedBounds.center;
        playerTransform.position = new Vector3(center.x, floorY + 1.6f, center.z);
        Debug.Log($"Player placed at mesh center: {playerTransform.position}");
    }
}
