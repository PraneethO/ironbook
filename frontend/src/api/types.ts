/**
 * Strongly-typed models matching CONTRACT.md §1. These mirror the backend's
 * JSON shapes exactly so the rest of the app never invents fields.
 */

export type ProjectStatus =
  | 'draft'
  | 'uploading'
  | 'queued'
  | 'processing'
  | 'ready'
  | 'failed';

export interface Project {
  id: string;
  name: string;
  status: ProjectStatus;
  created_at: string;
  updated_at: string;
  photo_count: number;
  thumbnail_url: string | null;
  has_asset: boolean;
}

export type JobStatus = 'queued' | 'processing' | 'ready' | 'failed';
export type StageStatus = 'pending' | 'active' | 'done' | 'failed';

/** Ordered stage keys, per CONTRACT.md STAGES table. */
export type StageKey =
  | 'preprocessing'
  | 'pose_estimation'
  | 'structure'
  | 'optimization'
  | 'compression'
  | 'viewer_asset';

export interface JobStage {
  key: StageKey | string;
  label: string;
  status: StageStatus;
  progress: number;
}

export interface JobLog {
  ts: string;
  level: 'info' | 'warn' | 'error';
  stage: string;
  message: string;
}

export interface Job {
  project_id: string;
  status: JobStatus;
  progress: number;
  current_stage: string | null;
  stages: JobStage[];
  logs: JobLog[];
  error: string | null;
}

export interface RejectedFile {
  filename: string;
  reason: string;
}

export interface ValidationReport {
  accepted: number;
  rejected: RejectedFile[];
  photo_count: number;
  coverage_score: number;
  quality_score: number;
  warnings: string[];
  ready_to_reconstruct: boolean;
}

export interface UploadedImage {
  filename: string;
  thumbnail_url: string;
  width: number;
  height: number;
  sharpness: number;
}

export type ReconstructionBackend =
  | 'brush'
  | 'msplat'
  | 'gaussian_3dgs'
  | 'depth'
  | 'fallback'
  | 'colmap_gsplat';

export interface Health {
  status: string;
  reconstruction_backend: ReconstructionBackend;
}

export interface AssetInfo {
  splat_count: number;
  bytes: number;
  bounds: { min: [number, number, number]; max: [number, number, number] };
  format: string;
}

export interface ShareLink {
  url: string;
}

// --- Reasoning navigation agent (mirrors backend models.py additions) -----

export type AgentActionType =
  | 'move'
  | 'rotate'
  | 'zoom'
  | 'fly_to'
  | 'look_at'
  | 'highlight'
  | 'clear_highlight'
  | 'reset_view'
  | 'set_splat_scale'
  | 'set_background'
  | 'set_brightness';

export type AgentDirection =
  | 'forward'
  | 'backward'
  | 'left'
  | 'right'
  | 'up'
  | 'down'
  | 'clockwise'
  | 'counterclockwise'
  | 'in'
  | 'out';

export interface AgentAction {
  type: AgentActionType;
  direction?: AgentDirection;
  amount?: number;
  target_2d?: [number, number]; // [nx, ny] in [0,1], top-left origin
  label?: string;
}

export interface AgentActResponse {
  answer: string;
  actions: AgentAction[];
}

export interface CameraSnapshot {
  mode: string;
  fov: number;
  eye: [number, number, number];
  target: [number, number, number];
  bounds: {
    min: [number, number, number];
    max: [number, number, number];
  };
}

export interface AgentTurn {
  role: 'user' | 'assistant';
  text: string;
}
