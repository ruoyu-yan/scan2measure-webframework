import fs from "node:fs";
import path from "node:path";
import { v4 as uuidv4 } from "uuid";

export interface ProjectRecord {
  id: string;
  name: string;
  created: string;
  type: "full_pipeline" | "tour_only";
  status: "pending" | "in_progress" | "completed" | "error";
  inputs: {
    pointCloud?: string;
    panoramas?: string[];
    glbFile?: string;
  };
  outputs: {
    densityImage?: string;
    coloredPly?: string;
    meshGlb?: string;
    meshMetadata?: string;
  };
  lastCompletedStage: number;
  qualityTier: string;
}

interface StoreData {
  projects: ProjectRecord[];
}

export class ProjectStore {
  private filePath: string;
  private data: StoreData;

  constructor(projectRoot: string) {
    this.filePath = path.join(projectRoot, "data", "projects", "projects.json");
    this.data = this.load();
  }

  private load(): StoreData {
    try {
      const raw = fs.readFileSync(this.filePath, "utf-8");
      return JSON.parse(raw) as StoreData;
    } catch {
      return { projects: [] };
    }
  }

  private save(): void {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    fs.writeFileSync(this.filePath, JSON.stringify(this.data, null, 2));
  }

  list(): ProjectRecord[] {
    return this.data.projects;
  }

  get(id: string): ProjectRecord | undefined {
    return this.data.projects.find((p) => p.id === id);
  }

  create(partial: Omit<ProjectRecord, "id" | "created" | "status" | "outputs" | "lastCompletedStage">): ProjectRecord {
    const project: ProjectRecord = {
      ...partial,
      id: uuidv4(),
      created: new Date().toISOString(),
      status: "pending",
      outputs: {},
      lastCompletedStage: -1,
    };
    this.data.projects.unshift(project);
    this.save();
    return project;
  }

  update(id: string, patch: Partial<ProjectRecord>): ProjectRecord | null {
    const idx = this.data.projects.findIndex((p) => p.id === id);
    if (idx === -1) return null;
    this.data.projects[idx] = { ...this.data.projects[idx], ...patch };
    this.save();
    return this.data.projects[idx];
  }

  delete(id: string): boolean {
    const before = this.data.projects.length;
    this.data.projects = this.data.projects.filter((p) => p.id !== id);
    if (this.data.projects.length < before) {
      this.save();
      return true;
    }
    return false;
  }

  /** Create the per-project output directory and return its absolute path. */
  ensureProjectDir(projectRoot: string, projectId: string): string {
    const dir = path.join(projectRoot, "data", "projects", projectId);
    fs.mkdirSync(dir, { recursive: true });
    return dir;
  }

  /** Check that all input/output file paths still exist on disk. */
  validatePaths(project: ProjectRecord): string[] {
    const missing: string[] = [];
    const check = (p: string | undefined) => {
      if (p && !fs.existsSync(p)) missing.push(p);
    };
    check(project.inputs.pointCloud);
    project.inputs.panoramas?.forEach(check);
    check(project.inputs.glbFile);
    check(project.outputs.densityImage);
    check(project.outputs.coloredPly);
    check(project.outputs.meshGlb);
    return missing;
  }
}
