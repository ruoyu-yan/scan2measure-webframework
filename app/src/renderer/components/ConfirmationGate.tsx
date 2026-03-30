import { useRef, useEffect, useState } from "react";

/** A single match entry from demo6_alignment.json */
interface AlignmentMatch {
  pano_name: string;
  room_label: string;
  camera_position: [number, number];
  angle_deg: number;
  scale: number;
  transformed: number[][];  // polygon vertices [[x,y], ...]
}

/** Top-level alignment JSON structure */
interface AlignmentData {
  matches: AlignmentMatch[];
  scale: number;
}

/** Internal state per pano: original data + drag offset */
interface PanoEntry {
  match: AlignmentMatch;
  offsetX: number;
  offsetY: number;
}

interface ConfirmationGateProps {
  densityImagePath?: string;
  cameraPositions?: Array<{ name: string; x: number; y: number }>;
  alignmentJsonPath?: string;
  panoramaDir?: string;
  onConfirm?: () => void;
  onCorrect?: (correctedAlignment: unknown) => void;
}

export default function ConfirmationGate({
  densityImagePath,
  alignmentJsonPath,
  panoramaDir,
  onConfirm,
  onCorrect,
}: ConfirmationGateProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [entries, setEntries] = useState<PanoEntry[]>([]);
  const [alignmentData, setAlignmentData] = useState<AlignmentData | null>(null);
  const [dragging, setDragging] = useState<number | null>(null);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [selectedPano, setSelectedPano] = useState<number | null>(null);
  const [thumbnailSrc, setThumbnailSrc] = useState<string | null>(null);
  const [imgSize, setImgSize] = useState<{ w: number; h: number }>({ w: 800, h: 600 });

  // Load alignment JSON
  useEffect(() => {
    if (!alignmentJsonPath) return;
    fetch(`file://${alignmentJsonPath}`)
      .then((r) => r.json())
      .then((data: AlignmentData) => {
        setAlignmentData(data);
        setEntries(
          data.matches.map((m) => ({ match: m, offsetX: 0, offsetY: 0 }))
        );
      })
      .catch(() => {
        // Failed to load alignment JSON
      });
  }, [alignmentJsonPath]);

  // Load density image to get natural size
  useEffect(() => {
    if (!densityImagePath) return;
    const img = new Image();
    img.src = `file://${densityImagePath}`;
    img.onload = () => setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
  }, [densityImagePath]);

  // Load pano thumbnail on click
  useEffect(() => {
    if (selectedPano === null || !panoramaDir) {
      setThumbnailSrc(null);
      return;
    }
    const panoName = entries[selectedPano]?.match.pano_name;
    if (!panoName) return;
    // Try common extensions
    for (const ext of [".jpg", ".jpeg", ".png"]) {
      const tryPath = `${panoramaDir}/${panoName}${ext}`;
      // In Electron renderer, use file:// protocol
      setThumbnailSrc(`file://${tryPath}`);
      return;
    }
  }, [selectedPano, entries, panoramaDir]);

  // Colors for polygon outlines (cycle through palette)
  const COLORS = ["#00b4d8", "#ff6b6b", "#51cf66", "#ffd43b", "#cc5de8", "#ff922b"];

  // SVG coordinate helpers
  const getSVGCoords = (e: React.MouseEvent<SVGSVGElement>): { x: number; y: number } => {
    const svg = svgRef.current!;
    const rect = svg.getBoundingClientRect();
    const scaleX = imgSize.w / rect.width;
    const scaleY = imgSize.h / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  // Build polygon points string with offset applied
  const polygonPoints = (entry: PanoEntry): string =>
    entry.match.transformed
      .map(([x, y]) => `${x + entry.offsetX},${y + entry.offsetY}`)
      .join(" ");

  // Compute centroid for label placement
  const centroid = (entry: PanoEntry): { cx: number; cy: number } => {
    const verts = entry.match.transformed;
    const n = verts.length || 1;
    const cx = verts.reduce((s, [x]) => s + x, 0) / n + entry.offsetX;
    const cy = verts.reduce((s, [, y]) => s + y, 0) / n + entry.offsetY;
    return { cx, cy };
  };

  // Check if click is inside a polygon (ray casting)
  const pointInPolygon = (px: number, py: number, entry: PanoEntry): boolean => {
    const verts = entry.match.transformed.map(([x, y]) => [
      x + entry.offsetX,
      y + entry.offsetY,
    ]);
    let inside = false;
    for (let i = 0, j = verts.length - 1; i < verts.length; j = i++) {
      const xi = verts[i][0], yi = verts[i][1];
      const xj = verts[j][0], yj = verts[j][1];
      if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
        inside = !inside;
      }
    }
    return inside;
  };

  const handleMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    const { x, y } = getSVGCoords(e);
    // Check polygons in reverse order (topmost first)
    for (let i = entries.length - 1; i >= 0; i--) {
      if (pointInPolygon(x, y, entries[i])) {
        setDragging(i);
        setDragStart({ x, y });
        return;
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (dragging === null || !dragStart) return;
    const { x, y } = getSVGCoords(e);
    const dx = x - dragStart.x;
    const dy = y - dragStart.y;
    setEntries((prev) => {
      const updated = [...prev];
      updated[dragging] = {
        ...updated[dragging],
        offsetX: updated[dragging].offsetX + dx,
        offsetY: updated[dragging].offsetY + dy,
      };
      return updated;
    });
    setDragStart({ x, y });
  };

  const handleMouseUp = () => {
    setDragging(null);
    setDragStart(null);
  };

  const handlePolygonClick = (idx: number) => {
    if (dragging !== null) return; // Don't toggle on drag end
    setSelectedPano((prev) => (prev === idx ? null : idx));
  };

  const handleCorrect = () => {
    if (!onCorrect || !alignmentData) return;
    // Preserve ALL original fields, only update camera_position based on drag offset
    const corrected: AlignmentData = {
      matches: entries.map((entry) => ({
        pano_name: entry.match.pano_name,
        room_label: entry.match.room_label,
        camera_position: [
          entry.match.camera_position[0] + entry.offsetX,
          entry.match.camera_position[1] + entry.offsetY,
        ] as [number, number],
        angle_deg: entry.match.angle_deg,
        scale: entry.match.scale,
        transformed: entry.match.transformed,
      })),
      scale: alignmentData.scale,
    };
    onCorrect(corrected);
  };

  return (
    <div className="confirmation-gate">
      <div className="confirmation-gate__canvas">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${imgSize.w} ${imgSize.h}`}
          preserveAspectRatio="xMidYMid meet"
          style={{ width: "100%", maxHeight: "100%", cursor: dragging !== null ? "grabbing" : "default" }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          {/* Density image background */}
          {densityImagePath && (
            <image
              href={`file://${densityImagePath}`}
              x={0} y={0}
              width={imgSize.w} height={imgSize.h}
            />
          )}

          {/* Polygon outlines for each pano */}
          {entries.map((entry, idx) => {
            const color = COLORS[idx % COLORS.length];
            const { cx, cy } = centroid(entry);
            return (
              <g key={entry.match.pano_name} onClick={() => handlePolygonClick(idx)}>
                <polygon
                  points={polygonPoints(entry)}
                  fill={selectedPano === idx ? `${color}33` : `${color}1a`}
                  stroke={color}
                  strokeWidth={2}
                  style={{ cursor: "grab" }}
                />
                {/* Camera position marker */}
                <circle
                  cx={entry.match.camera_position[0] + entry.offsetX}
                  cy={entry.match.camera_position[1] + entry.offsetY}
                  r={5}
                  fill={color}
                />
                {/* Pano name label at centroid */}
                <text
                  x={cx} y={cy}
                  fill="#fff"
                  fontSize={12}
                  fontWeight="bold"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  style={{ pointerEvents: "none", textShadow: "1px 1px 2px #000" }}
                >
                  {entry.match.pano_name}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Pano thumbnail overlay on click */}
        {selectedPano !== null && thumbnailSrc && (
          <div className="confirmation-gate__thumbnail">
            <img
              src={thumbnailSrc}
              alt={entries[selectedPano]?.match.pano_name}
              style={{ maxWidth: 320, maxHeight: 180, borderRadius: 6, border: "2px solid var(--accent)" }}
              onError={() => setThumbnailSrc(null)}
            />
            <p style={{ color: "var(--text-secondary)", fontSize: "0.8rem", marginTop: 4 }}>
              {entries[selectedPano]?.match.pano_name}
            </p>
          </div>
        )}
      </div>
      <div className="confirmation-gate__actions">
        <button className="btn btn--primary" onClick={onConfirm}>
          Confirm &amp; Continue
        </button>
        <button className="btn btn--secondary" onClick={handleCorrect}>
          Re-run with Corrections
        </button>
      </div>
    </div>
  );
}
