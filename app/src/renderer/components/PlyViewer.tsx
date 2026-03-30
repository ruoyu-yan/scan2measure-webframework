import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

interface PlyViewerProps {
  plyPath: string;
}

export default function PlyViewer({ plyPath }: PlyViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pointCount, setPointCount] = useState(0);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let disposed = false;
    let animId = 0;
    let renderer: THREE.WebGLRenderer | null = null;
    let controls: OrbitControls | null = null;
    let resizeObserver: ResizeObserver | null = null;
    const disposables: { dispose: () => void }[] = [];

    async function init() {
      if (!container || disposed) return;

      setLoading(true);
      setError(null);

      // Load PLY via custom local-file:// protocol (supports streaming large files)
      const loader = new PLYLoader();
      let geometry: THREE.BufferGeometry;
      try {
        geometry = await new Promise<THREE.BufferGeometry>((resolve, reject) => {
          const url = `local-file://${plyPath}`;
          loader.load(url, resolve, undefined, reject);
        });
      } catch (err) {
        if (!disposed) setError(`Failed to load PLY: ${(err as Error).message}`);
        setLoading(false);
        return;
      }

      if (disposed) {
        geometry.dispose();
        return;
      }

      disposables.push(geometry);

      const numPoints = geometry.attributes.position.count;
      setPointCount(numPoints);

      // Downsample for rendering if too many points (>2M)
      const MAX_RENDER_POINTS = 2_000_000;
      let renderGeometry = geometry;
      if (numPoints > MAX_RENDER_POINTS) {
        renderGeometry = downsampleGeometry(geometry, MAX_RENDER_POINTS);
        disposables.push(renderGeometry);
      }

      // Three.js setup
      const width = container.clientWidth;
      const height = container.clientHeight;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x1a1a2e);

      const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 10000);

      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      container.appendChild(renderer.domElement);

      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.1;

      // Create point cloud material
      const hasColors = !!renderGeometry.attributes.color;
      const material = new THREE.PointsMaterial({
        size: 0.02,
        vertexColors: hasColors,
        sizeAttenuation: true,
      });
      if (!hasColors) {
        material.color.set(0x88aaff);
      }
      disposables.push(material);

      const points = new THREE.Points(renderGeometry, material);
      scene.add(points);

      // Auto-fit camera
      renderGeometry.computeBoundingBox();
      const box = renderGeometry.boundingBox!;
      const center = new THREE.Vector3();
      box.getCenter(center);
      const size = new THREE.Vector3();
      box.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z);
      const fitDistance = maxDim * 1.2;

      controls.target.copy(center);
      camera.position.set(
        center.x,
        center.y + fitDistance * 0.8,
        center.z + fitDistance * 0.6,
      );
      camera.near = maxDim / 1000;
      camera.far = maxDim * 100;
      camera.updateProjectionMatrix();
      controls.update();

      setLoading(false);

      const localRenderer = renderer;
      const localControls = controls;
      function animate() {
        if (disposed) return;
        animId = requestAnimationFrame(animate);
        localControls.update();
        localRenderer.render(scene, camera);
      }
      animate();

      resizeObserver = new ResizeObserver(() => {
        if (disposed || !container || !localRenderer) return;
        const w = container.clientWidth;
        const h = container.clientHeight;
        if (w === 0 || h === 0) return;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        localRenderer.setSize(w, h);
      });
      resizeObserver.observe(container);
    }

    init();

    return () => {
      disposed = true;
      cancelAnimationFrame(animId);
      resizeObserver?.disconnect();
      for (const d of disposables) d.dispose();
      controls?.dispose();
      if (renderer) {
        renderer.dispose();
        if (container.contains(renderer.domElement)) {
          container.removeChild(renderer.domElement);
        }
      }
    };
  }, [plyPath]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "100%",
        position: "relative",
        overflow: "hidden",
        background: "#1a1a2e",
      }}
    >
      {loading && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#8888aa",
            fontSize: 14,
            fontFamily: "monospace",
            zIndex: 10,
            pointerEvents: "none",
          }}
        >
          <div style={{ textAlign: "center" }}>
            <div className="running-spinner" style={{ margin: "0 auto 12px" }} />
            Loading point cloud...
          </div>
        </div>
      )}

      {error && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#ff6b6b",
            fontSize: 14,
            fontFamily: "monospace",
            padding: 24,
            textAlign: "center",
            zIndex: 10,
            pointerEvents: "none",
          }}
        >
          {error}
        </div>
      )}

      {/* Point count badge */}
      {!loading && !error && pointCount > 0 && (
        <div
          style={{
            position: "absolute",
            bottom: 36,
            left: 12,
            fontSize: 11,
            fontFamily: "monospace",
            color: "#aaaacc",
            zIndex: 5,
            pointerEvents: "none",
          }}
        >
          {(pointCount / 1_000_000).toFixed(1)}M points
        </div>
      )}

      {!loading && !error && (
        <div
          style={{
            position: "absolute",
            bottom: 10,
            left: "50%",
            transform: "translateX(-50%)",
            fontSize: 11,
            fontFamily: "monospace",
            color: "#555577",
            whiteSpace: "nowrap",
            zIndex: 5,
            pointerEvents: "none",
          }}
        >
          Drag to rotate &bull; Scroll to zoom &bull; Right-click to pan
        </div>
      )}
    </div>
  );
}

/** Uniformly downsample a geometry to roughly targetCount points. */
function downsampleGeometry(src: THREE.BufferGeometry, targetCount: number): THREE.BufferGeometry {
  const srcPos = src.attributes.position;
  const srcCol = src.attributes.color;
  const total = srcPos.count;
  const stride = Math.max(1, Math.floor(total / targetCount));
  const kept = Math.ceil(total / stride);

  const positions = new Float32Array(kept * 3);
  const colors = srcCol ? new Float32Array(kept * 3) : null;

  for (let i = 0, out = 0; i < total && out < kept; i += stride, out++) {
    positions[out * 3] = srcPos.getX(i);
    positions[out * 3 + 1] = srcPos.getY(i);
    positions[out * 3 + 2] = srcPos.getZ(i);
    if (colors && srcCol) {
      colors[out * 3] = srcCol.getX(i);
      colors[out * 3 + 1] = srcCol.getY(i);
      colors[out * 3 + 2] = srcCol.getZ(i);
    }
  }

  const dst = new THREE.BufferGeometry();
  dst.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  if (colors) {
    dst.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  }
  return dst;
}
