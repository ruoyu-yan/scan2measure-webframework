export interface StageConfig {
  index: number;
  id: string;
  name: string;
  description: string;
  scriptPaths: string[];
  condaEnv: "scan_env" | "sam3";
  viewType: "2d" | "3d" | "progress" | "confirmation";
  perPano: boolean;
}

/**
 * Full pipeline stage definitions.
 * Script paths are relative to the project root (scan2measure-webframework/).
 * Stages 5a and 5b run sequentially within a single "Line Detection" stage group.
 */
export const FULL_PIPELINE_STAGES: StageConfig[] = [
  {
    index: 0,
    id: "density_image",
    name: "Density Image",
    description: "Generating 2D density projection from point cloud",
    scriptPaths: ["src/preprocessing/generate_density_image.py"],
    condaEnv: "scan_env",
    viewType: "2d",
    perPano: false,
  },
  {
    index: 1,
    id: "sam3_segmentation",
    name: "Room Segmentation",
    description: "SAM3 room boundary detection on density image",
    scriptPaths: ["src/experiments/SAM3_room_segmentation.py"],
    condaEnv: "sam3",
    viewType: "2d",
    perPano: false,
  },
  {
    index: 2,
    id: "sam3_polygons",
    name: "Mask to Polygons",
    description: "Converting SAM3 masks to world-meter polygons",
    scriptPaths: ["src/floorplan/SAM3_mask_to_polygons.py"],
    condaEnv: "scan_env",
    viewType: "2d",
    perPano: false,
  },
  {
    index: 3,
    id: "pano_footprints",
    name: "Pano Footprints",
    description: "Extracting room polygons from panoramic images",
    scriptPaths: ["src/experiments/SAM3_pano_footprint_extraction.py"],
    condaEnv: "sam3",
    viewType: "2d",
    perPano: true,
  },
  {
    index: 4,
    id: "polygon_matching",
    name: "Polygon Matching",
    description: "Fitting pano footprints into density-image room slots",
    scriptPaths: ["src/floorplan/align_polygons_demo6.py"],
    condaEnv: "scan_env",
    viewType: "2d",
    perPano: false,
  },
  {
    index: 5,
    id: "line_detection_3d",
    name: "3D Line Detection",
    description: "Detecting 3D line segments and clustering by principal direction",
    scriptPaths: [
      "src/geometry_3d/point_cloud_geometry_baker_V4.py",
      "src/geometry_3d/cluster_3d_lines.py",
    ],
    condaEnv: "scan_env",
    viewType: "3d",
    perPano: false,
  },
  {
    index: 6,
    id: "line_detection_2d",
    name: "2D Feature Extraction",
    description: "Extracting sphere-based line features from panoramas",
    scriptPaths: ["src/features_2d/image_feature_extractionV2.py"],
    condaEnv: "scan_env",
    viewType: "2d",
    perPano: true,
  },
  {
    index: 7,
    id: "pose_estimation",
    name: "Pose Estimation",
    description: "Multi-room camera localization with Voronoi filtering",
    scriptPaths: ["src/pose_estimation/multiroom_pose_estimation.py"],
    condaEnv: "scan_env",
    viewType: "2d",
    perPano: false,
  },
  {
    index: 8,
    id: "confirmation",
    name: "Confirm Poses",
    description: "Verify camera positions before colorization",
    scriptPaths: [],
    condaEnv: "scan_env",
    viewType: "confirmation",
    perPano: false,
  },
  {
    index: 9,
    id: "colorization",
    name: "Colorization",
    description: "Coloring point cloud from panoramic images",
    scriptPaths: ["src/colorization/colorize_point_cloud.py"],
    condaEnv: "scan_env",
    viewType: "progress",
    perPano: false,
  },
  {
    index: 10,
    id: "meshing",
    name: "Meshing",
    description: "Generating UV-textured GLB mesh",
    scriptPaths: ["src/meshing/mesh_reconstruction.py"],
    condaEnv: "scan_env",
    viewType: "progress",
    perPano: false,
  },
  {
    index: 11,
    id: "done",
    name: "Complete",
    description: "Pipeline complete -- preview mesh or launch virtual tour",
    scriptPaths: [],
    condaEnv: "scan_env",
    viewType: "3d",
    perPano: false,
  },
];

/** Stages for Mesh Only entry point (colored PLY in, GLB out). */
export const MESH_ONLY_STAGES: StageConfig[] = [
  {
    index: 0,
    id: "meshing",
    name: "Meshing",
    description: "Generating UV-textured GLB mesh from colored point cloud",
    scriptPaths: ["src/meshing/mesh_reconstruction.py"],
    condaEnv: "scan_env",
    viewType: "progress",
    perPano: false,
  },
  {
    index: 1,
    id: "done",
    name: "Complete",
    description: "Mesh ready -- preview or launch virtual tour",
    scriptPaths: [],
    condaEnv: "scan_env",
    viewType: "3d",
    perPano: false,
  },
];

/** Per-project output subdirectory names, indexed by stage id. */
export const STAGE_OUTPUT_DIRS: Record<string, string> = {
  density_image: "density_image",
  sam3_segmentation: "sam3_segmentation",
  sam3_polygons: "sam3_polygons",
  pano_footprints: "sam3_footprints",
  polygon_matching: "alignment",
  line_detection_3d: "line_detection",
  line_detection_2d: "feature_extraction",
  pose_estimation: "pose_estimation",
  colorization: "textured_point_cloud",
  meshing: "mesh",
};

/** Default projects data directory (relative to repo root). */
export const PROJECTS_DATA_DIR = "data/projects";

/** Default projects.json location (relative to repo root). */
export const PROJECTS_JSON = "data/projects/projects.json";
