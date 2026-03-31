import { useState, useEffect, useRef, useCallback } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ConfirmMatchingProps {
  densityImagePath: string;
  alignmentJsonPath: string;
  polygonsJsonPath?: string;  // path to <pcName>_polygons.json with room polygon data
  panoThumbnails: Record<string, string>; // panoName -> absolute file path
  onConfirm: () => void;
  onCorrect?: (correctedAlignment: unknown) => void;
}

export type { ConfirmMatchingProps };

/** One match entry from demo6_alignment.json. */
interface AlignmentMatch {
  pano_name: string;
  room_idx: number;
  room_label?: string;
  score?: number;
  rotation_deg?: number;
  scale?: number;
  camera_position: [number, number];
  room_polygon?: number[][]; // optional: [[x,y], ...]
  transformed?: number[][];  // optional: pano polygon after alignment
}

/** Room entry (if present in alignment JSON). */
interface RoomEntry {
  polygon: number[][]; // [[x,y], ...]
  label: string;
}

/** Room entry from the separate polygons JSON file (<pcName>_polygons.json). */
interface PolygonsJsonRoom {
  label: string;
  vertices_world_meters: number[][]; // [[x,y], ...]
  mask_index?: number;
  [key: string]: unknown;
}

interface PolygonsJsonData {
  rooms: PolygonsJsonRoom[];
  [key: string]: unknown;
}

/** Top-level alignment JSON structure (flexible). */
interface AlignmentData {
  matches: AlignmentMatch[];
  rooms?: RoomEntry[];
  metadata?: {
    scale?: number;
    image_size?: [number, number];
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

/** Internal marker state. */
interface MarkerState {
  match: AlignmentMatch;
  /** Position in image-pixel coords. */
  cx: number;
  cy: number;
  /** Current assigned room label. */
  roomLabel: string;
  /** Index into COLORS palette. */
  colorIndex: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const COLORS = [
  "#ff6", "#6ff", "#f6f", "#6f6", "#f96", "#69f", "#f66", "#6f9",
];

/** High-resolution internal canvas size (replaces the old 256x256 density image). */
const CANVAS_RES = 1024;

const MARKER_RADIUS = 24;
const MARKER_RADIUS_SELECTED = 30;
const RING_WIDTH = 6;
const RING_WIDTH_SELECTED = 8;
const HIT_RADIUS = 48;

/** Distinct pastel/muted fill colors for room polygons (background). */
const ROOM_FILLS = [
  "rgba(100,149,237,0.30)",  // cornflower blue
  "rgba(144,238,144,0.30)",  // light green
  "rgba(255,182,193,0.30)",  // light pink
  "rgba(255,218,125,0.30)",  // warm yellow
  "rgba(186,152,255,0.30)",  // lavender
  "rgba(127,219,199,0.30)",  // teal
  "rgba(255,160,122,0.30)",  // light salmon
  "rgba(176,196,222,0.30)",  // light steel blue
];

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/** Ray-casting point-in-polygon test. */
function pointInPolygon(px: number, py: number, polygon: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1];
    const xj = polygon[j][0], yj = polygon[j][1];
    if (
      yi > py !== yj > py &&
      px < ((xj - xi) * (py - yi)) / (yj - yi) + xi
    ) {
      inside = !inside;
    }
  }
  return inside;
}

/** Strip common prefixes for shorter display names. */
function shortName(name: string): string {
  return name.replace(/^TMB_/i, "");
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ConfirmMatching(props: ConfirmMatchingProps) {
  const {
    // densityImagePath is accepted for backward compatibility but no longer used;
    // room polygons are rendered programmatically at high resolution instead.
    densityImagePath: _densityImagePath,
    alignmentJsonPath,
    polygonsJsonPath,
    panoThumbnails,
    onConfirm,
    onCorrect,
  } = props;
  void _densityImagePath; // suppress unused-variable warning

  // Refs
  const wrapperRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Data loading state
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [alignment, setAlignment] = useState<AlignmentData | null>(null);
  const [polygonsData, setPolygonsData] = useState<PolygonsJsonData | null>(null);

  // Marker state
  const [markers, setMarkers] = useState<MarkerState[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [draggingIdx, setDraggingIdx] = useState<number | null>(null);
  const [dragOffset, setDragOffset] = useState<{ dx: number; dy: number }>({
    dx: 0,
    dy: 0,
  });
  // Track whether a real drag happened (vs. just a mousedown+up click)
  const didDragRef = useRef(false);

  // Popup thumbnail
  const [thumbnailSrc, setThumbnailSrc] = useState<string | null>(null);

  // Canvas sizing (CSS pixels, fitted to container)
  const [canvasSize, setCanvasSize] = useState<{ w: number; h: number }>({
    w: 800,
    h: 600,
  });
  // Internal canvas resolution (high-res, replaces old density-image size)
  const [imgNatural] = useState<{ w: number; h: number }>({
    w: CANVAS_RES,
    h: CANVAS_RES,
  });

  // Room polygons in image-pixel space (for hit-testing & drawing)
  const [roomPolygonsCanvas, setRoomPolygonsCanvas] = useState<
    { label: string; polygon: number[][] }[]
  >([]);

  // -----------------------------------------------------------------------
  // Coordinate transform refs: world meters <-> image pixels.
  //
  // The alignment camera_positions are in world-meter space. We compute a
  // bounding box over all known world coordinates and map them uniformly
  // into the density image pixel space.
  // -----------------------------------------------------------------------
  const worldToImgRef = useRef<(wx: number, wy: number) => { x: number; y: number }>(
    (wx, wy) => ({ x: wx, y: wy })
  );
  const imgToWorldRef = useRef<(ix: number, iy: number) => { x: number; y: number }>(
    (ix, iy) => ({ x: ix, y: iy })
  );

  // -----------------------------------------------------------------------
  // Load alignment JSON via readImage IPC (returns base64 of any file)
  // -----------------------------------------------------------------------
  useEffect(() => {
    if (!alignmentJsonPath) return;
    let cancelled = false;

    (async () => {
      try {
        const b64: string = await (window as any).electronAPI.readImage(
          alignmentJsonPath
        );
        if (cancelled) return;
        const commaIdx = b64.indexOf(",");
        const jsonText = atob(b64.substring(commaIdx + 1));
        const data: AlignmentData = JSON.parse(jsonText);
        if (cancelled) return;
        setAlignment(data);
      } catch {
        if (!cancelled) setError("Failed to load alignment JSON");
      }
    })();

    return () => { cancelled = true; };
  }, [alignmentJsonPath]);

  // -----------------------------------------------------------------------
  // Load separate polygons JSON (room boundaries in world-meter coords)
  // -----------------------------------------------------------------------
  useEffect(() => {
    if (!polygonsJsonPath) return;
    let cancelled = false;

    (async () => {
      try {
        const b64: string = await (window as any).electronAPI.readImage(
          polygonsJsonPath
        );
        if (cancelled) return;
        const commaIdx = b64.indexOf(",");
        const jsonText = atob(b64.substring(commaIdx + 1));
        const data: PolygonsJsonData = JSON.parse(jsonText);
        if (cancelled) return;
        setPolygonsData(data);
      } catch {
        // Polygons JSON is optional; don't set error state
      }
    })();

    return () => { cancelled = true; };
  }, [polygonsJsonPath]);

  // -----------------------------------------------------------------------
  // Once alignment is loaded, build coordinate transforms and initialise
  // markers.  Also re-runs when polygonsData arrives so room boundaries
  // are included in the bounding box and drawn on the canvas.
  // -----------------------------------------------------------------------
  useEffect(() => {
    if (!alignment) return;
    setLoading(false);

    const imgW = CANVAS_RES;
    const imgH = CANVAS_RES;

    // Gather all world-space points to determine bounding box
    const worldPts: number[][] = [];
    for (const m of alignment.matches) {
      if (m.camera_position) worldPts.push(m.camera_position);
      if (m.room_polygon) worldPts.push(...m.room_polygon);
      if (m.transformed) worldPts.push(...m.transformed);
    }
    if (alignment.rooms) {
      for (const r of alignment.rooms) {
        if (r.polygon) worldPts.push(...r.polygon);
      }
    }
    // Include room polygon vertices from the separate polygons JSON
    if (polygonsData?.rooms) {
      for (const r of polygonsData.rooms) {
        if (r.vertices_world_meters) worldPts.push(...r.vertices_world_meters);
      }
    }

    if (worldPts.length === 0) {
      setError("No coordinate data found in alignment JSON");
      return;
    }

    // World bounding box
    let wMinX = Infinity, wMaxX = -Infinity;
    let wMinY = Infinity, wMaxY = -Infinity;
    for (const [wx, wy] of worldPts) {
      if (wx < wMinX) wMinX = wx;
      if (wx > wMaxX) wMaxX = wx;
      if (wy < wMinY) wMinY = wy;
      if (wy > wMaxY) wMaxY = wy;
    }

    // Add 10% padding
    const rangeX = wMaxX - wMinX || 1;
    const rangeY = wMaxY - wMinY || 1;
    const padX = rangeX * 0.1;
    const padY = rangeY * 0.1;
    wMinX -= padX;  wMaxX += padX;
    wMinY -= padY;  wMaxY += padY;
    const wRangeX = wMaxX - wMinX;
    const wRangeY = wMaxY - wMinY;

    // Map into image pixel space with margin for marker labels
    const margin = 0.05;
    const drawW = imgW * (1 - 2 * margin);
    const drawH = imgH * (1 - 2 * margin);
    const offX = imgW * margin;
    const offY = imgH * margin;

    // Uniform scale preserving aspect ratio
    const scaleU = Math.min(drawW / wRangeX, drawH / wRangeY);
    const cxOff = offX + (drawW - wRangeX * scaleU) / 2;
    const cyOff = offY + (drawH - wRangeY * scaleU) / 2;

    const worldToImg = (wx: number, wy: number) => ({
      x: cxOff + (wx - wMinX) * scaleU,
      y: cyOff + (wy - wMinY) * scaleU,
    });
    const imgToWorld = (ix: number, iy: number) => ({
      x: wMinX + (ix - cxOff) / scaleU,
      y: wMinY + (iy - cyOff) / scaleU,
    });

    worldToImgRef.current = worldToImg;
    imgToWorldRef.current = imgToWorld;

    // Build room polygons in image-pixel space.
    // Priority: 1) separate polygons JSON (vertices_world_meters)
    //           2) alignment.rooms (embedded in alignment JSON)
    //           3) per-match room_polygon fields
    const roomPolys: { label: string; polygon: number[][] }[] = [];

    // Source 1: separate polygons JSON (<pcName>_polygons.json)
    if (polygonsData?.rooms) {
      for (const r of polygonsData.rooms) {
        if (!r.vertices_world_meters || r.vertices_world_meters.length < 3) continue;
        roomPolys.push({
          label: r.label,
          polygon: r.vertices_world_meters.map(([wx, wy]) => {
            const p = worldToImg(wx, wy);
            return [p.x, p.y];
          }),
        });
      }
    }
    // Source 2: alignment.rooms (if embedded in alignment JSON)
    if (roomPolys.length === 0 && alignment.rooms) {
      for (const r of alignment.rooms) {
        if (!r.polygon || r.polygon.length < 3) continue;
        roomPolys.push({
          label: r.label,
          polygon: r.polygon.map(([wx, wy]) => {
            const p = worldToImg(wx, wy);
            return [p.x, p.y];
          }),
        });
      }
    }
    // Source 3: per-match room_polygon fields
    if (roomPolys.length === 0) {
      const seen = new Set<number>();
      for (const m of alignment.matches) {
        if (m.room_polygon && m.room_polygon.length >= 3 && !seen.has(m.room_idx)) {
          seen.add(m.room_idx);
          roomPolys.push({
            label: m.room_label || `Room ${m.room_idx}`,
            polygon: m.room_polygon.map(([wx, wy]) => {
              const p = worldToImg(wx, wy);
              return [p.x, p.y];
            }),
          });
        }
      }
    }
    setRoomPolygonsCanvas(roomPolys);

    // Build initial markers
    const newMarkers: MarkerState[] = alignment.matches.map((m, i) => {
      const pos = worldToImg(m.camera_position[0], m.camera_position[1]);
      return {
        match: m,
        cx: pos.x,
        cy: pos.y,
        roomLabel: m.room_label || `Room ${m.room_idx}`,
        colorIndex: i % COLORS.length,
      };
    });
    setMarkers(newMarkers);
    setSelectedIdx(null);
    setDraggingIdx(null);
  }, [alignment, polygonsData]);

  // -----------------------------------------------------------------------
  // Resize observer: keep canvas fitted to container with image aspect ratio
  // -----------------------------------------------------------------------
  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;

    const updateSize = () => {
      const rect = wrapper.getBoundingClientRect();
      const cW = rect.width;
      const cH = rect.height;
      if (cW <= 0 || cH <= 0) return;

      const imgAspect = imgNatural.w / imgNatural.h;
      const cAspect = cW / cH;

      let w: number, h: number;
      if (cAspect > imgAspect) {
        h = cH;
        w = h * imgAspect;
      } else {
        w = cW;
        h = w / imgAspect;
      }
      setCanvasSize({ w: Math.floor(w), h: Math.floor(h) });
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(wrapper);
    return () => observer.disconnect();
  }, [imgNatural]);

  // -----------------------------------------------------------------------
  // Room hit-testing in image-pixel space
  // -----------------------------------------------------------------------
  const findRoomAtImgCoords = useCallback(
    (ix: number, iy: number): string => {
      for (const room of roomPolygonsCanvas) {
        if (pointInPolygon(ix, iy, room.polygon)) {
          return room.label;
        }
      }
      return "(unassigned)";
    },
    [roomPolygonsCanvas]
  );

  // -----------------------------------------------------------------------
  // Draw everything onto the canvas
  // -----------------------------------------------------------------------
  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Internal resolution: high-res for crisp rendering
    canvas.width = imgNatural.w;
    canvas.height = imgNatural.h;

    // Dark background
    ctx.fillStyle = "#0a0a1a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // ----- Room polygons as filled background shapes (replaces density image) -----
    for (let ri = 0; ri < roomPolygonsCanvas.length; ri++) {
      const room = roomPolygonsCanvas[ri];
      if (room.polygon.length < 3) continue;

      // Filled room shape with distinct pastel color
      ctx.beginPath();
      ctx.moveTo(room.polygon[0][0], room.polygon[0][1]);
      for (let i = 1; i < room.polygon.length; i++) {
        ctx.lineTo(room.polygon[i][0], room.polygon[i][1]);
      }
      ctx.closePath();
      ctx.fillStyle = ROOM_FILLS[ri % ROOM_FILLS.length];
      ctx.fill();

      // Crisp outline
      ctx.strokeStyle = "rgba(255,255,255,0.45)";
      ctx.lineWidth = 3;
      ctx.stroke();

      // Room label at centroid
      const centX = room.polygon.reduce((s, p) => s + p[0], 0) / room.polygon.length;
      const centY = room.polygon.reduce((s, p) => s + p[1], 0) / room.polygon.length;
      ctx.font = "bold 24px sans-serif";
      ctx.fillStyle = "rgba(255,255,255,0.45)";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(room.label, centX, centY);
    }

    // Pano transformed polygons (dashed outlines, if available)
    for (const marker of markers) {
      if (!marker.match.transformed || marker.match.transformed.length < 3) continue;
      const color = COLORS[marker.colorIndex];
      const poly = marker.match.transformed.map(([wx, wy]) =>
        worldToImgRef.current(wx, wy)
      );
      ctx.beginPath();
      ctx.moveTo(poly[0].x, poly[0].y);
      for (let i = 1; i < poly.length; i++) {
        ctx.lineTo(poly[i].x, poly[i].y);
      }
      ctx.closePath();
      ctx.fillStyle = color + "1a";
      ctx.fill();
      ctx.strokeStyle = color + "88";
      ctx.lineWidth = 3;
      ctx.setLineDash([12, 8]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Markers
    for (let i = 0; i < markers.length; i++) {
      const m = markers[i];
      const color = COLORS[m.colorIndex];
      const isSel = i === selectedIdx;
      const isDrag = i === draggingIdx;
      const r = isSel ? MARKER_RADIUS_SELECTED : MARKER_RADIUS;
      const ringW = isSel ? RING_WIDTH_SELECTED : RING_WIDTH;

      ctx.save();

      // Glow
      ctx.shadowColor = color;
      ctx.shadowBlur = isSel ? 48 : 30;

      // Filled circle
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(m.cx, m.cy, r, 0, Math.PI * 2);
      ctx.fill();

      ctx.shadowBlur = 0;

      // White ring
      ctx.strokeStyle = isSel ? "#fff" : "rgba(255,255,255,0.7)";
      ctx.lineWidth = ringW;
      if (isDrag) ctx.setLineDash([12, 8]);
      ctx.beginPath();
      ctx.arc(m.cx, m.cy, r + ringW + 3, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      // Dashed highlight circle around selected (when not dragging)
      if (isSel && !isDrag) {
        ctx.strokeStyle = color + "88";
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 8]);
        ctx.beginPath();
        ctx.arc(m.cx, m.cy, r + 24, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Short label above
      ctx.font = "bold 28px sans-serif";
      ctx.fillStyle = "#fff";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.shadowColor = "#000";
      ctx.shadowBlur = 8;
      ctx.fillText(shortName(m.match.pano_name), m.cx, m.cy - r - 18);

      // Room assignment below
      ctx.font = "22px sans-serif";
      ctx.fillStyle = "#aaa";
      ctx.textBaseline = "top";
      ctx.shadowBlur = 0;
      ctx.fillText(m.roomLabel, m.cx, m.cy + r + 24);

      ctx.restore();
    }

    // Instruction text at bottom
    ctx.font = "28px sans-serif";
    ctx.fillStyle = "#555";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(
      "Click marker to inspect  \u2022  Drag to reassign room",
      imgNatural.w / 2,
      imgNatural.h - 12
    );
  }, [imgNatural, roomPolygonsCanvas, markers, selectedIdx, draggingIdx]);

  // Redraw whenever relevant state changes
  useEffect(() => { redraw(); }, [redraw]);

  // -----------------------------------------------------------------------
  // Load pano thumbnail for the selected marker
  // -----------------------------------------------------------------------
  useEffect(() => {
    if (selectedIdx === null) { setThumbnailSrc(null); return; }
    const marker = markers[selectedIdx];
    if (!marker) { setThumbnailSrc(null); return; }
    const thumbPath = panoThumbnails[marker.match.pano_name];
    if (!thumbPath) { setThumbnailSrc(null); return; }

    let cancelled = false;
    (async () => {
      try {
        const b64: string = await (window as any).electronAPI.readImage(thumbPath);
        if (!cancelled) setThumbnailSrc(b64);
      } catch {
        if (!cancelled) setThumbnailSrc(null);
      }
    })();
    return () => { cancelled = true; };
  }, [selectedIdx, markers, panoThumbnails]);

  // -----------------------------------------------------------------------
  // Mouse helpers
  // -----------------------------------------------------------------------

  /** Convert browser clientX/clientY to image-pixel coordinates. */
  const clientToImgCoords = useCallback(
    (clientX: number, clientY: number): { ix: number; iy: number } => {
      const canvas = canvasRef.current;
      if (!canvas) return { ix: 0, iy: 0 };
      const rect = canvas.getBoundingClientRect();
      return {
        ix: ((clientX - rect.left) / rect.width) * imgNatural.w,
        iy: ((clientY - rect.top) / rect.height) * imgNatural.h,
      };
    },
    [imgNatural]
  );

  /** Hit-test markers (reverse order = topmost first). */
  const hitTestMarker = useCallback(
    (ix: number, iy: number): number | null => {
      for (let i = markers.length - 1; i >= 0; i--) {
        const dx = ix - markers[i].cx;
        const dy = iy - markers[i].cy;
        if (dx * dx + dy * dy < HIT_RADIUS * HIT_RADIUS) return i;
      }
      return null;
    },
    [markers]
  );

  // -----------------------------------------------------------------------
  // Mouse event handlers
  // -----------------------------------------------------------------------

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      const { ix, iy } = clientToImgCoords(e.clientX, e.clientY);
      const hitIdx = hitTestMarker(ix, iy);
      if (hitIdx !== null) {
        setDraggingIdx(hitIdx);
        setSelectedIdx(hitIdx);
        didDragRef.current = false;
        setDragOffset({
          dx: ix - markers[hitIdx].cx,
          dy: iy - markers[hitIdx].cy,
        });
        e.preventDefault();
      }
    },
    [clientToImgCoords, hitTestMarker, markers]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (draggingIdx === null) {
        // Hover cursor
        const { ix, iy } = clientToImgCoords(e.clientX, e.clientY);
        const canvas = canvasRef.current;
        if (canvas) {
          canvas.style.cursor = hitTestMarker(ix, iy) !== null ? "grab" : "default";
        }
        return;
      }

      // Dragging
      didDragRef.current = true;
      const { ix, iy } = clientToImgCoords(e.clientX, e.clientY);
      const newCx = ix - dragOffset.dx;
      const newCy = iy - dragOffset.dy;
      const newRoom = findRoomAtImgCoords(newCx, newCy);

      setMarkers((prev) => {
        const updated = [...prev];
        updated[draggingIdx] = {
          ...updated[draggingIdx],
          cx: newCx,
          cy: newCy,
          roomLabel: newRoom,
        };
        return updated;
      });
    },
    [draggingIdx, dragOffset, clientToImgCoords, hitTestMarker, findRoomAtImgCoords]
  );

  const handleMouseUp = useCallback(() => {
    if (draggingIdx !== null) {
      // Finalise room assignment at drop position
      setMarkers((prev) => {
        const updated = [...prev];
        const m = updated[draggingIdx];
        updated[draggingIdx] = {
          ...m,
          roomLabel: findRoomAtImgCoords(m.cx, m.cy),
        };
        return updated;
      });
      setDraggingIdx(null);
    }
  }, [draggingIdx, findRoomAtImgCoords]);

  const handleMouseLeave = useCallback(() => {
    if (draggingIdx !== null) {
      setDraggingIdx(null);
    }
  }, [draggingIdx]);

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent) => {
      // If a real drag occurred, don't toggle selection on mouseup/click
      if (didDragRef.current) return;

      const { ix, iy } = clientToImgCoords(e.clientX, e.clientY);
      const hitIdx = hitTestMarker(ix, iy);
      if (hitIdx !== null) {
        setSelectedIdx((prev) => (prev === hitIdx ? null : hitIdx));
      } else {
        setSelectedIdx(null);
      }
    },
    [clientToImgCoords, hitTestMarker]
  );

  // -----------------------------------------------------------------------
  // Popup positioning (HTML div, not drawn on canvas)
  // -----------------------------------------------------------------------
  const computePopupStyle = useCallback((): React.CSSProperties => {
    if (selectedIdx === null) return { display: "none" };
    const marker = markers[selectedIdx];
    if (!marker) return { display: "none" };

    const canvas = canvasRef.current;
    const wrapper = wrapperRef.current;
    if (!canvas || !wrapper) return { display: "none" };

    const canvasRect = canvas.getBoundingClientRect();
    const wrapperRect = wrapper.getBoundingClientRect();

    // Marker screen position relative to wrapper
    const markerSX = (marker.cx / imgNatural.w) * canvasRect.width;
    const markerSY = (marker.cy / imgNatural.h) * canvasRect.height;
    const canvasOX = canvasRect.left - wrapperRect.left;
    const canvasOY = canvasRect.top - wrapperRect.top;

    const popupW = 380;
    let left = canvasOX + markerSX + 20;
    if (left + popupW > wrapperRect.width) {
      left = canvasOX + markerSX - popupW - 20;
    }
    // Clamp top so popup stays within canvas bounds
    const popupEstH = 280;
    let top = canvasOY + markerSY - 60;
    if (top + popupEstH > canvasOY + canvasRect.height) {
      top = canvasOY + canvasRect.height - popupEstH - 8;
    }
    top = Math.max(4, top);

    return {
      position: "absolute",
      left: `${left}px`,
      top: `${top}px`,
      maxWidth: `${popupW}px`,
      zIndex: 10,
      background: "rgba(10,10,30,0.95)",
      border: `1px solid ${COLORS[marker.colorIndex]}`,
      borderRadius: "8px",
      padding: "12px 16px",
      boxShadow: "0 4px 20px rgba(0,0,0,0.6)",
      pointerEvents: "none" as const,
    };
  }, [selectedIdx, markers, imgNatural]);

  // -----------------------------------------------------------------------
  // Confirm / Correct
  // -----------------------------------------------------------------------
  const handleConfirm = useCallback(() => { onConfirm(); }, [onConfirm]);

  const handleCorrect = useCallback(() => {
    if (!onCorrect || !alignment) return;
    const correctedMatches = markers.map((m) => {
      const worldPos = imgToWorldRef.current(m.cx, m.cy);
      return {
        ...m.match,
        camera_position: [worldPos.x, worldPos.y] as [number, number],
        room_label: m.roomLabel,
      };
    });
    onCorrect({ ...alignment, matches: correctedMatches });
  }, [onCorrect, alignment, markers]);

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  if (error) {
    return (
      <div style={containerStyle}>
        <div style={errorMsgStyle}>{error}</div>
      </div>
    );
  }

  if (loading) {
    return (
      <div style={containerStyle}>
        <div style={loadingMsgStyle}>Loading alignment data...</div>
      </div>
    );
  }

  const selMarker = selectedIdx !== null ? markers[selectedIdx] : null;

  return (
    <div style={containerStyle}>
      {/* Canvas wrapper */}
      <div ref={wrapperRef} style={canvasWrapperStyle}>
        <canvas
          ref={canvasRef}
          style={{
            width: canvasSize.w,
            height: canvasSize.h,
            borderRadius: 4,
            cursor: draggingIdx !== null ? "grabbing" : "default",
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          onClick={handleCanvasClick}
        />

        {/* Popup */}
        {selMarker && (
          <div style={computePopupStyle()}>
            <div style={{
              fontWeight: 600,
              color: COLORS[selMarker.colorIndex],
              marginBottom: 4,
              fontSize: 14,
            }}>
              {selMarker.match.pano_name}
            </div>
            <div style={{ fontSize: 12, color: "#aaa", marginBottom: 6 }}>
              Assigned to:{" "}
              <strong style={{ color: "#fff" }}>{selMarker.roomLabel}</strong>
            </div>
            {selMarker.match.score != null && (
              <div style={{ fontSize: 11, color: "#777", marginBottom: 6 }}>
                Score: {selMarker.match.score.toFixed(3)}
              </div>
            )}
            {thumbnailSrc ? (
              <img
                src={thumbnailSrc}
                alt={selMarker.match.pano_name}
                style={{
                  width: "100%",
                  height: 160,
                  objectFit: "cover",
                  borderRadius: 4,
                }}
              />
            ) : (
              <div style={{
                height: 80,
                background: "#1a2744",
                borderRadius: 4,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#555",
                fontSize: 11,
              }}>
                No preview
              </div>
            )}
          </div>
        )}
      </div>

      {/* Action buttons */}
      <div style={actionsBarStyle}>
        <button style={btnPrimaryStyle} onClick={handleConfirm}>
          Confirm &amp; Continue
        </button>
        {onCorrect && (
          <button style={btnSecondaryStyle} onClick={handleCorrect}>
            Re-run with Corrections
          </button>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Inline styles (matching app dark theme)
// ---------------------------------------------------------------------------

const containerStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  height: "100%",
  width: "100%",
};

const canvasWrapperStyle: React.CSSProperties = {
  flex: 1,
  position: "relative",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  overflow: "hidden",
  minHeight: 0,
};

const actionsBarStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "center",
  gap: 16,
  padding: "12px 16px",
  borderTop: "1px solid #0f3460",
  flexShrink: 0,
};

const btnPrimaryStyle: React.CSSProperties = {
  padding: "8px 20px",
  border: "none",
  borderRadius: 6,
  fontSize: 13,
  fontWeight: 500,
  cursor: "pointer",
  background: "#e94560",
  color: "#fff",
};

const btnSecondaryStyle: React.CSSProperties = {
  padding: "8px 20px",
  borderRadius: 6,
  fontSize: 13,
  fontWeight: 500,
  cursor: "pointer",
  background: "#1a2744",
  color: "#ccc",
  border: "1px solid #333",
};

const loadingMsgStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  height: "100%",
  color: "#a0a0b0",
  fontSize: "0.95rem",
};

const errorMsgStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  height: "100%",
  color: "#f44336",
  fontSize: "0.95rem",
};
