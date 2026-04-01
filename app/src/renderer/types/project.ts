export type ProjectType = "full_pipeline" | "tour_only";
export type ProjectStatus = "pending" | "in_progress" | "completed" | "error";

export interface ProjectInputs {
  pointCloud?: string;
  panoramas?: string[];
  glbFile?: string;
}

export interface ProjectOutputs {
  densityImage?: string;
  coloredPly?: string;
  meshGlb?: string;
  meshMetadata?: string;
}

export interface Project {
  id: string;
  name: string;
  created: string;
  type: ProjectType;
  status: ProjectStatus;
  inputs: ProjectInputs;
  outputs: ProjectOutputs;
  lastCompletedStage: number;
  qualityTier: string;
}

export interface ProjectStore {
  projects: Project[];
}
