export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'partial';
export type StageStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'skipped';

export interface VideoRecord {
  video_id: string;
  file_name: string;
  sha256: string;
  status: 'uploaded' | 'validated' | 'rejected';
  mime_type: string;
  file_size_bytes: number;
  duration_us: number;
  width: number;
  height: number;
  fps_avg: number;
  created_at: string;
  authorization_attested: boolean;
  adult_subject_attested: boolean;
  file_path?: string;
}

export interface StageInfo {
  stage_id?: string;
  name: string;
  status: StageStatus;
  progress: number;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  metrics?: Record<string, any>;
}

export interface AnalysisJob {
  analysis_id: string;
  video_id: string;
  status: JobStatus;
  progress: number;
  requested_modules: string[];
  config_overrides: Record<string, any>;
  stages: StageInfo[];
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error?: {
    code: string;
    message: string;
    details?: Record<string, any>;
  };
  annotations_history?: {
    revision: number;
    timestamp: string;
    operations: AnnotationOp[];
  }[];
}

export interface AnnotationOp {
  op: 'add' | 'replace' | 'remove';
  path: string;
  value?: any;
}

export interface SchemaValidationError {
  instancePath: string;
  schemaPath: string;
  keyword: string;
  params: Record<string, any>;
  message?: string;
}

export interface ValidationReport {
  valid: boolean;
  schema_version: string;
  validated_at: string;
  errors: SchemaValidationError[];
  warnings: string[];
  summary: {
    passed_rules: number;
    failed_rules: number;
    invariant_checks: { name: string; status: 'pass' | 'fail' | 'warn'; detail: string }[];
  };
}

export interface ArtifactInfo {
  id: string;
  name: string;
  kind: 'keyframe' | 'overlay' | 'plot' | 'report' | 'blueprint' | 'bundle';
  uri: string;
  mime_type: string;
  size_bytes: number;
  sha256: string;
}

export interface BlueprintManifest {
  schema_version: string;
  blueprint_id: string;
  created_at: string;
  source_video: {
    file_name: string;
    mime_type: string;
    container: string;
    file_size_bytes: number;
    sha256: string;
    duration_us: number;
    width: number;
    height: number;
    display_aspect_ratio: string;
    pixel_aspect_ratio: string;
    rotation_deg: number;
    video_codec: string;
    pixel_format: string;
    bit_depth: number;
    color_primaries: string;
    color_transfer: string;
    color_space: string;
    fps_avg: number;
    fps_nominal: number;
    variable_frame_rate: boolean;
    source_frame_count: number;
    has_audio: boolean;
    audio_codec: string | null;
    audio_sample_rate_hz: number | null;
    audio_channels: number | null;
    metadata_stripped: boolean;
  };
  timebase: {
    normalized_to_cfr: boolean;
    fps_num: number;
    fps_den: number;
    frame_count: number;
    frame_duration_us: number;
    start_pts_us: number;
    end_pts_us: number;
    source_pts_map_ref: string | null;
  };
  processing: {
    job_id: string;
    status: JobStatus;
    requested_modules: string[];
    started_at: string;
    completed_at: string;
    pipeline_version: string;
    config_hash: string;
    execution_mode: string;
    hardware: {
      gpu: string;
      vram_gb: number;
    };
    stages: StageInfo[];
  };
  shots: any[];
  characters: any[];
  camera: {
    per_shot: any[];
    quality: {
      score: number;
      coverage: number;
      warnings: string[];
      errors: string[];
    };
  };
  environment: {
    background_mask_ref: string | null;
    depth_ref: string | null;
    luminance_ref: string | null;
    exposure_change_ref: string | null;
    white_balance_proxy_ref: string | null;
    blur_ref: string | null;
    occluder_tracks: any[];
    quality: {
      score: number;
      coverage: number;
      warnings: string[];
      errors: string[];
    };
  };
  semantics: any | null;
  quality: {
    overall_score: number;
    module_scores: Record<string, number>;
    warnings: string[];
    errors: string[];
    low_confidence_intervals: any[];
    reproducible: boolean;
  };
  artifacts: {
    manifest_uri: string;
    bundle_uri: string;
    overlays: any[];
    reports: any[];
  };
  provenance: {
    tools: any[];
    system: {
      os: string;
      python: string;
      node?: string;
    };
  };
  extensions: Record<string, any>;
}
