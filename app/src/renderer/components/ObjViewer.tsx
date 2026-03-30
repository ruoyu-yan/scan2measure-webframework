import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface ObjViewerProps {
  objPath: string;
}

/**
 * Parse an OBJ file containing colored line segments from 3DLineDetection.
 * Format: v x y z r g b / l v1 v2
 */
function parseObjLines(text: string) {
  const vertices: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];
  for (const line of text.split('\n')) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === 'v' && parts.length >= 7) {
      vertices.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
      colors.push(parseFloat(parts[4]), parseFloat(parts[5]), parseFloat(parts[6]));
    } else if (parts[0] === 'l' && parts.length >= 3) {
      indices.push(parseInt(parts[1]) - 1, parseInt(parts[2]) - 1); // OBJ is 1-indexed
    }
  }
  return { vertices, colors, indices };
}

export default function ObjViewer({ objPath }: ObjViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

      // ---- Load OBJ via IPC ----
      setLoading(true);
      setError(null);

      let dataUri: string | null = null;
      try {
        dataUri = await window.electronAPI.readImage(objPath);
      } catch {
        setError('Failed to read OBJ file.');
        setLoading(false);
        return;
      }

      if (disposed) return;

      if (!dataUri) {
        setError(`File not found: ${objPath}`);
        setLoading(false);
        return;
      }

      // Decode base64 data URI to text
      let objText: string;
      try {
        const base64 = dataUri.split(',')[1];
        objText = atob(base64);
      } catch {
        setError('Failed to decode OBJ file data.');
        setLoading(false);
        return;
      }

      // Parse OBJ
      const parsed = parseObjLines(objText);
      if (parsed.vertices.length === 0 || parsed.indices.length === 0) {
        setError('OBJ file contains no line geometry.');
        setLoading(false);
        return;
      }

      if (disposed) return;

      // ---- Three.js setup ----
      const width = container.clientWidth;
      const height = container.clientHeight;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x050510);

      const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 10000);

      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(width, height);
      renderer.setPixelRatio(window.devicePixelRatio);
      container.appendChild(renderer.domElement);

      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.1;

      // Build geometry
      const geometry = new THREE.BufferGeometry();
      const posAttr = new THREE.Float32BufferAttribute(parsed.vertices, 3);
      const colAttr = new THREE.Float32BufferAttribute(parsed.colors, 3);
      geometry.setAttribute('position', posAttr);
      geometry.setAttribute('color', colAttr);
      geometry.setIndex(parsed.indices);
      disposables.push(geometry);

      const material = new THREE.LineBasicMaterial({ vertexColors: true });
      disposables.push(material);

      const lineSegments = new THREE.LineSegments(geometry, material);
      scene.add(lineSegments);

      // Auto-fit camera to bounding box
      geometry.computeBoundingBox();
      const box = geometry.boundingBox!;
      const center = new THREE.Vector3();
      box.getCenter(center);
      const size = new THREE.Vector3();
      box.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z);
      const fitDistance = maxDim * 1.5;

      controls.target.copy(center);
      camera.position.set(
        center.x + fitDistance * 0.6,
        center.y + fitDistance * 0.4,
        center.z + fitDistance * 0.6,
      );
      camera.near = maxDim / 1000;
      camera.far = maxDim * 100;
      camera.updateProjectionMatrix();
      controls.update();

      setLoading(false);

      // Animation loop
      const localRenderer = renderer;
      const localControls = controls;
      function animate() {
        if (disposed) return;
        animId = requestAnimationFrame(animate);
        localControls.update();
        localRenderer.render(scene, camera);
      }
      animate();

      // Responsive resize
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

    // Cleanup
    return () => {
      disposed = true;
      cancelAnimationFrame(animId);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
      for (const d of disposables) {
        d.dispose();
      }
      if (controls) {
        controls.dispose();
      }
      if (renderer) {
        renderer.dispose();
        if (container && container.contains(renderer.domElement)) {
          container.removeChild(renderer.domElement);
        }
      }
    };
  }, [objPath]);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        overflow: 'hidden',
        background: '#050510',
      }}
    >
      {/* Loading overlay */}
      {loading && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#8888aa',
            fontSize: 14,
            fontFamily: 'monospace',
            zIndex: 10,
            pointerEvents: 'none',
          }}
        >
          Loading OBJ...
        </div>
      )}

      {/* Error overlay */}
      {error && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#ff6b6b',
            fontSize: 14,
            fontFamily: 'monospace',
            padding: 24,
            textAlign: 'center',
            zIndex: 10,
            pointerEvents: 'none',
          }}
        >
          {error}
        </div>
      )}

      {/* Legend (bottom-left) */}
      {!loading && !error && (
        <div
          style={{
            position: 'absolute',
            bottom: 36,
            left: 12,
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
            fontSize: 11,
            fontFamily: 'monospace',
            color: '#aaaacc',
            zIndex: 5,
            pointerEvents: 'none',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span
              style={{
                display: 'inline-block',
                width: 14,
                height: 8,
                backgroundColor: '#ff4444',
                borderRadius: 1,
              }}
            />
            Vertical (Z)
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span
              style={{
                display: 'inline-block',
                width: 14,
                height: 8,
                backgroundColor: '#44ff44',
                borderRadius: 1,
              }}
            />
            Wall-X
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span
              style={{
                display: 'inline-block',
                width: 14,
                height: 8,
                backgroundColor: '#4488ff',
                borderRadius: 1,
              }}
            />
            Wall-Y
          </div>
        </div>
      )}

      {/* Hint text (bottom-center) */}
      {!loading && !error && (
        <div
          style={{
            position: 'absolute',
            bottom: 10,
            left: '50%',
            transform: 'translateX(-50%)',
            fontSize: 11,
            fontFamily: 'monospace',
            color: '#555577',
            whiteSpace: 'nowrap',
            zIndex: 5,
            pointerEvents: 'none',
          }}
        >
          Drag to rotate &bull; Scroll to zoom &bull; Right-click to pan
        </div>
      )}
    </div>
  );
}
