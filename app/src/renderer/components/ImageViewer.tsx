import { useState, useEffect } from "react";

interface ImageViewerProps {
  imagePath: string;
}

export default function ImageViewer({ imagePath }: ImageViewerProps) {
  const [dataUri, setDataUri] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setDataUri(null);

    window.electronAPI
      .readImage(imagePath)
      .then((uri) => {
        if (cancelled) return;
        if (uri) {
          setDataUri(uri);
        } else {
          setError(`File not found: ${imagePath}`);
        }
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(`Failed to load image: ${(err as Error).message}`);
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [imagePath]);

  if (loading) {
    return (
      <div style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100%",
        color: "var(--text-secondary)",
      }}>
        <span>Loading image...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100%",
        color: "var(--error)",
      }}>
        <span>{error}</span>
      </div>
    );
  }

  return (
    <div style={{
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      height: "100%",
    }}>
      {dataUri && (
        <img
          src={dataUri}
          alt={imagePath.split(/[/\\]/).pop() || ""}
          style={{
            maxWidth: "100%",
            maxHeight: "100%",
            borderRadius: 4,
            objectFit: "contain",
          }}
        />
      )}
    </div>
  );
}
