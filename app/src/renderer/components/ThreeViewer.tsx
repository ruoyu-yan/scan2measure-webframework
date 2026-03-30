import { useRef, useEffect, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

interface ThreeViewerProps {
  objPath?: string;
  plyPath?: string;
  glbPath?: string;
}

export default function ThreeViewer({ objPath, plyPath, glbPath }: ThreeViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [effectivePlyPath, setEffectivePlyPath] = useState<string | undefined>(plyPath);

  // Downsample PLY for preview before loading
  useEffect(() => {
    if (!plyPath) {
      setEffectivePlyPath(undefined);
      return;
    }
    const downsampledPath = plyPath.replace(/\.ply$/, "_preview.ply");
    window.electronAPI
      .downsamplePly(plyPath, downsampledPath)
      .then((result: unknown) => {
        const typed = result as { ok?: boolean; outputPath?: string };
        setEffectivePlyPath(typed.ok && typed.outputPath ? typed.outputPath : plyPath);
      })
      .catch(() => {
        // Fall back to original PLY if downsampling fails
        setEffectivePlyPath(plyPath);
      });
  }, [plyPath]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(5, 5, 5);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambient);
    const directional = new THREE.DirectionalLight(0xffffff, 0.8);
    directional.position.set(10, 10, 10);
    scene.add(directional);

    // Grid helper
    const grid = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
    scene.add(grid);

    // Load model based on provided path (use downsampled PLY if available)
    const filePath = glbPath || objPath || effectivePlyPath;
    if (filePath) {
      setLoading(true);
      setError(null);
      const fileUrl = `file://${filePath}`;

      if (glbPath) {
        new GLTFLoader().load(
          fileUrl,
          (gltf) => {
            scene.add(gltf.scene);
            fitCameraToObject(camera, controls, gltf.scene);
            setLoading(false);
          },
          undefined,
          (err) => { setError(`Failed to load GLB: ${err}`); setLoading(false); }
        );
      } else if (objPath) {
        new OBJLoader().load(
          fileUrl,
          (obj) => {
            // Wireframe material for line detection output
            obj.traverse((child) => {
              if (child instanceof THREE.Mesh) {
                child.material = new THREE.MeshBasicMaterial({
                  color: 0x00b4d8,
                  wireframe: true,
                });
              }
            });
            scene.add(obj);
            fitCameraToObject(camera, controls, obj);
            setLoading(false);
          },
          undefined,
          (err) => { setError(`Failed to load OBJ: ${err}`); setLoading(false); }
        );
      } else if (effectivePlyPath) {
        new PLYLoader().load(
          fileUrl,
          (geometry) => {
            if (geometry.hasAttribute("color")) {
              const material = new THREE.PointsMaterial({ size: 0.01, vertexColors: true });
              scene.add(new THREE.Points(geometry, material));
            } else {
              const material = new THREE.PointsMaterial({ size: 0.01, color: 0x00b4d8 });
              scene.add(new THREE.Points(geometry, material));
            }
            fitCameraToObject(camera, controls, new THREE.Mesh(geometry));
            setLoading(false);
          },
          undefined,
          (err) => { setError(`Failed to load PLY: ${err}`); setLoading(false); }
        );
      }
    }

    // Animation loop
    let animId: number;
    function animate() {
      animId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    // Handle resize
    const handleResize = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    // Cleanup: dispose scene objects to prevent GPU memory leaks
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", handleResize);
      scene.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.geometry.dispose();
          if (Array.isArray(obj.material)) {
            obj.material.forEach((m) => m.dispose());
          } else {
            obj.material.dispose();
          }
        } else if (obj instanceof THREE.Points) {
          obj.geometry.dispose();
          if (obj.material instanceof THREE.Material) obj.material.dispose();
        }
      });
      controls.dispose();
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [objPath, effectivePlyPath, glbPath]);

  return (
    <div ref={containerRef} className="three-viewer" style={{ width: "100%", height: "100%" }}>
      {loading && (
        <div className="three-viewer__overlay">Loading 3D model...</div>
      )}
      {error && (
        <div className="three-viewer__overlay three-viewer__overlay--error">{error}</div>
      )}
    </div>
  );
}

/** Fit camera to center on a loaded object. */
function fitCameraToObject(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  object: THREE.Object3D
) {
  const box = new THREE.Box3().setFromObject(object);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();

  controls.target.copy(center);
  camera.position.copy(center);
  camera.position.x += size * 0.8;
  camera.position.y += size * 0.5;
  camera.position.z += size * 0.8;
  camera.near = size / 100;
  camera.far = size * 100;
  camera.updateProjectionMatrix();
  controls.update();
}
