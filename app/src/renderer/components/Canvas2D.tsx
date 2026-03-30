import { useRef, useEffect } from "react";

interface Canvas2DProps {
  imagePaths?: string[];
  densityImagePath?: string;
  polygonsJsonPath?: string;
}

export default function Canvas2D({ imagePaths, densityImagePath, polygonsJsonPath: _polygonsJsonPath }: Canvas2DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Load density image as the base layer
    const imgSrc = densityImagePath || (imagePaths && imagePaths[0]);
    if (!imgSrc) {
      ctx.fillStyle = "#1a1a2e";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#a0a0b0";
      ctx.font = "16px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for stage output...", canvas.width / 2, canvas.height / 2);
      return;
    }

    const img = new Image();
    // file:// protocol for local images in Electron
    img.src = `file://${imgSrc}`;
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);
    };
    img.onerror = () => {
      ctx.fillStyle = "#1a1a2e";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#f44336";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(`Failed to load: ${imgSrc}`, canvas.width / 2, canvas.height / 2);
    };
  }, [imagePaths, densityImagePath]);

  return (
    <div className="canvas2d-container">
      <canvas ref={canvasRef} width={800} height={600} />
    </div>
  );
}
