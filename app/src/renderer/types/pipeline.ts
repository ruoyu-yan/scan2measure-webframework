export type StageStatus =
  | "pending"
  | "active"
  | "complete"
  | "error"
  | "confirmation";

export type QualityTier = "preview" | "balanced" | "high";

export interface StageProgress {
  current: number;
  total: number;
  message: string;
}

export interface StageDefinition {
  index: number;
  id: string;
  name: string;
  description: string;
  scriptPath: string; // relative to project root
  condaEnv: "scan_env" | "sam3";
  viewType: "2d" | "3d" | "progress" | "confirmation";
  /** For stages that run per-panorama, set true */
  perPano: boolean;
}

export interface PipelineState {
  stages: StageStatus[];
  currentStage: number;
  progress: StageProgress | null;
  logLines: string[];
  stderrTail: string;
  elapsedMs: number;
  qualityTier: QualityTier;
}
