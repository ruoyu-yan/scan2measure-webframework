import { useState, useEffect, useCallback } from "react";

interface ImageGalleryProps {
  images: Array<{ path: string; label: string }>;
}

interface LoadedImage {
  path: string;
  label: string;
  dataUri: string | null;
  error: boolean;
}

export default function ImageGallery({ images }: ImageGalleryProps) {
  const [loaded, setLoaded] = useState<LoadedImage[]>([]);
  const [expanded, setExpanded] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadAll() {
      const results: LoadedImage[] = [];
      for (const img of images) {
        try {
          const uri = await window.electronAPI.readImage(img.path);
          if (cancelled) return;
          results.push({
            path: img.path,
            label: img.label,
            dataUri: uri || null,
            error: !uri,
          });
        } catch {
          if (cancelled) return;
          results.push({
            path: img.path,
            label: img.label,
            dataUri: null,
            error: true,
          });
        }
      }
      if (!cancelled) setLoaded(results);
    }

    loadAll();
    return () => {
      cancelled = true;
    };
  }, [images]);

  const handleClose = useCallback(() => setExpanded(null), []);

  if (images.length === 0) {
    return (
      <div style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100%",
        color: "var(--text-secondary)",
      }}>
        No images to display
      </div>
    );
  }

  // Show expanded view if an image is selected
  if (expanded !== null && loaded[expanded]?.dataUri) {
    return (
      <div style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
      }}>
        <div style={{
          padding: "8px 0",
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}>
          <button
            className="btn btn--secondary"
            onClick={handleClose}
            style={{ fontSize: "0.8rem", padding: "4px 12px" }}
          >
            Back to gallery
          </button>
          <span style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
            {loaded[expanded].label}
          </span>
        </div>
        <div style={{
          flex: 1,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          overflow: "hidden",
        }}>
          <img
            src={loaded[expanded].dataUri!}
            alt={loaded[expanded].label}
            style={{
              maxWidth: "100%",
              maxHeight: "100%",
              borderRadius: 4,
              objectFit: "contain",
            }}
          />
        </div>
      </div>
    );
  }

  // Grid view
  return (
    <div style={{
      display: "flex",
      flexWrap: "wrap",
      gap: 12,
      alignContent: "flex-start",
    }}>
      {loaded.map((item, idx) => (
        <div
          key={item.path}
          onClick={() => !item.error && setExpanded(idx)}
          style={{
            width: 220,
            background: "var(--bg-card)",
            borderRadius: 8,
            overflow: "hidden",
            cursor: item.error ? "default" : "pointer",
            border: "1px solid transparent",
            transition: "border-color 0.15s",
          }}
          onMouseEnter={(e) => {
            if (!item.error) (e.currentTarget.style.borderColor = "var(--accent)");
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = "transparent";
          }}
        >
          <div style={{
            width: "100%",
            height: 160,
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            background: "var(--bg-primary)",
          }}>
            {item.dataUri ? (
              <img
                src={item.dataUri}
                alt={item.label}
                style={{
                  maxWidth: "100%",
                  maxHeight: "100%",
                  objectFit: "contain",
                }}
              />
            ) : item.error ? (
              <span style={{ color: "var(--error)", fontSize: "0.75rem" }}>
                Failed to load
              </span>
            ) : (
              <span style={{ color: "var(--text-secondary)", fontSize: "0.75rem" }}>
                Loading...
              </span>
            )}
          </div>
          <div style={{
            padding: "6px 10px",
            fontSize: "0.8rem",
            color: "var(--text-secondary)",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}>
            {item.label}
          </div>
        </div>
      ))}
      {loaded.length < images.length && (
        <div style={{
          display: "flex",
          alignItems: "center",
          color: "var(--text-secondary)",
          fontSize: "0.85rem",
          padding: "0 16px",
        }}>
          Loading images...
        </div>
      )}
    </div>
  );
}
