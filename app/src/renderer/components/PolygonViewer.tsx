import { useEffect, useRef, useState } from "react";

interface Room {
  label: string;
  vertices_pixel: number[][];
}

interface PolygonsData {
  rooms: Room[];
}

interface PolygonViewerProps {
  densityImagePath: string;
  polygonsJsonPath: string;
}

const ROOM_COLORS = [
  "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
  "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
];

export default function PolygonViewer({ densityImagePath, polygonsJsonPath }: PolygonViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function render() {
      try {
        // Load density image
        const imgDataUri = await window.electronAPI.readImage(densityImagePath);
        if (cancelled || !imgDataUri) return;

        // Load polygons JSON (read as base64, decode)
        const jsonDataUri = await window.electronAPI.readImage(polygonsJsonPath);
        if (cancelled || !jsonDataUri) return;

        // Decode base64 JSON
        const base64 = jsonDataUri.split(",")[1];
        const jsonStr = atob(base64);
        const polygonsData: PolygonsData = JSON.parse(jsonStr);

        // Load image
        const img = new Image();
        img.src = imgDataUri;
        await new Promise<void>((resolve, reject) => {
          img.onload = () => resolve();
          img.onerror = () => reject(new Error("Failed to load density image"));
        });
        if (cancelled) return;

        const canvas = canvasRef.current;
        const container = containerRef.current;
        if (!canvas || !container) return;

        // Fit canvas to container while preserving aspect ratio
        const containerW = container.clientWidth;
        const containerH = container.clientHeight;
        const scale = Math.min(containerW / img.width, containerH / img.height, 1);
        const drawW = Math.floor(img.width * scale);
        const drawH = Math.floor(img.height * scale);

        canvas.width = drawW;
        canvas.height = drawH;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        // Draw density image
        ctx.drawImage(img, 0, 0, drawW, drawH);

        // Draw polygon outlines
        const sx = drawW / img.width;
        const sy = drawH / img.height;

        for (let i = 0; i < polygonsData.rooms.length; i++) {
          const room = polygonsData.rooms[i];
          const color = ROOM_COLORS[i % ROOM_COLORS.length];
          const verts = room.vertices_pixel;
          if (!verts || verts.length < 3) continue;

          // Fill with semi-transparent color
          ctx.beginPath();
          ctx.moveTo(verts[0][0] * sx, verts[0][1] * sy);
          for (let j = 1; j < verts.length; j++) {
            ctx.lineTo(verts[j][0] * sx, verts[j][1] * sy);
          }
          ctx.closePath();
          ctx.fillStyle = color + "30"; // ~19% opacity
          ctx.fill();

          // Stroke outline
          ctx.beginPath();
          ctx.moveTo(verts[0][0] * sx, verts[0][1] * sy);
          for (let j = 1; j < verts.length; j++) {
            ctx.lineTo(verts[j][0] * sx, verts[j][1] * sy);
          }
          ctx.closePath();
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.stroke();

          // Label
          const cx = verts.reduce((s, v) => s + v[0], 0) / verts.length * sx;
          const cy = verts.reduce((s, v) => s + v[1], 0) / verts.length * sy;
          ctx.font = "bold 14px sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillStyle = "#000";
          ctx.fillText(room.label, cx + 1, cy + 1);
          ctx.fillStyle = color;
          ctx.fillText(room.label, cx, cy);
        }
      } catch (err) {
        if (!cancelled) setError((err as Error).message);
      }
    }

    render();
    return () => { cancelled = true; };
  }, [densityImagePath, polygonsJsonPath]);

  if (error) {
    return <div style={{ padding: 24, color: "#f87171" }}>Error: {error}</div>;
  }

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        overflow: "hidden",
      }}
    >
      <canvas ref={canvasRef} style={{ maxWidth: "100%", maxHeight: "100%" }} />
    </div>
  );
}
