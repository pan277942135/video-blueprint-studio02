import { v4 as uuidv4 } from 'uuid';
import { VideoRecord, AnalysisJob, BlueprintManifest, StageInfo } from './types';

export const ALL_E0_MODULES = [
  'shots',
  'people',
  'pose',
  'face',
  'hands',
  'masks',
  'camera',
  'flow',
  'micro_motion',
  'environment',
  'overlays',
];

export function createDefaultStages(): StageInfo[] {
  return [
    { stage_id: 'stage_01_ingest', name: 'Media Ingest & Demux', status: 'pending', progress: 0 },
    { stage_id: 'stage_02_timebase', name: 'CFR Timebase Normalization', status: 'pending', progress: 0 },
    { stage_id: 'stage_03_shots', name: 'Shot Boundary Detection', status: 'pending', progress: 0 },
    { stage_id: 'stage_04_camera', name: 'Camera Motion & Stabilization', status: 'pending', progress: 0 },
    { stage_id: 'stage_05_characters', name: 'Anonymous Character Detection & Tracking', status: 'pending', progress: 0 },
    { stage_id: 'stage_06_landmarks', name: 'Pose, Face & Hand Keypoint Extraction', status: 'pending', progress: 0 },
    { stage_id: 'stage_07_micromotion', name: 'Surface Residual Micro-Motion Analysis', status: 'pending', progress: 0 },
    { stage_id: 'stage_08_environment', name: 'Environment, Depth & Lighting Analysis', status: 'pending', progress: 0 },
    { stage_id: 'stage_09_bundle', name: 'Blueprint Assembly & Verification', status: 'pending', progress: 0 },
  ];
}

export function generateDeterministicBlueprint(
  job: AnalysisJob,
  video: VideoRecord
): BlueprintManifest {
  const blueprintId = uuidv4();
  const createdAt = new Date().toISOString();
  const durationUs = video.duration_us || 6000000;
  const frameCount = Math.round((durationUs / 1000000) * (video.fps_avg || 30));

  return {
    schema_version: '1.0.0',
    blueprint_id: blueprintId,
    created_at: createdAt,
    source_video: {
      file_name: video.file_name,
      mime_type: video.mime_type || 'video/mp4',
      container: 'mov,mp4,m4a,3gp,3g2,mj2',
      file_size_bytes: video.file_size_bytes || 10485760,
      sha256: video.sha256 || 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
      duration_us: durationUs,
      width: video.width || 1920,
      height: video.height || 1080,
      display_aspect_ratio: video.width && video.height ? `${video.width}:${video.height}` : '16:9',
      pixel_aspect_ratio: '1:1',
      rotation_deg: 0,
      video_codec: 'h264',
      pixel_format: 'yuv420p',
      bit_depth: 8,
      color_primaries: 'bt709',
      color_transfer: 'bt709',
      color_space: 'bt709',
      fps_avg: video.fps_avg || 30.0,
      fps_nominal: video.fps_avg || 30.0,
      variable_frame_rate: false,
      source_frame_count: frameCount,
      has_audio: false,
      audio_codec: null,
      audio_sample_rate_hz: null,
      audio_channels: null,
      metadata_stripped: true,
    },
    timebase: {
      normalized_to_cfr: true,
      fps_num: Math.round(video.fps_avg || 30),
      fps_den: 1,
      frame_count: frameCount,
      frame_duration_us: Math.round(1000000 / (video.fps_avg || 30)),
      start_pts_us: 0,
      end_pts_us: durationUs,
      source_pts_map_ref: 'sidecars/pts_map.json',
    },
    processing: {
      job_id: job.analysis_id,
      status: 'succeeded',
      requested_modules: job.requested_modules || ALL_E0_MODULES,
      started_at: job.started_at || createdAt,
      completed_at: new Date().toISOString(),
      pipeline_version: '1.0.0-mock-e0',
      config_hash: 'cfg-e0-deterministic-mock',
      execution_mode: 'offline_local',
      hardware: {
        gpu: 'Mock Execution Engine (CPU/Browser)',
        vram_gb: 16,
      },
      stages: job.stages,
    },
    shots: [
      {
        shot_id: 'shot_000',
        index: 0,
        frame_start: 0,
        frame_end: Math.max(0, frameCount - 1),
        time_start_us: 0,
        time_end_us: durationUs,
        cut_in_type: 'start',
        cut_out_type: 'end',
        transition_score: 1.0,
        keyframes: [
          {
            frame_idx: 0,
            time_us: 0,
            kind: 'first',
            score: 1.0,
            image_uri: 'artifacts/keyframes/shot_000_000000.png',
          },
          {
            frame_idx: Math.floor(frameCount / 2),
            time_us: Math.floor(durationUs / 2),
            kind: 'middle',
            score: 0.98,
            image_uri: 'artifacts/keyframes/shot_000_middle.png',
          },
        ],
        dominant_character_ids: ['char_000'],
        camera_motion_id: 'cam_000',
        quality: {
          score: 0.98,
          coverage: 1.0,
          warnings: [],
          errors: [],
        },
      },
    ],
    characters: [
      {
        character_id: 'char_000',
        appearance_shots: ['shot_000'],
        confidence: 0.96,
        track_ref: 'sidecars/character_char_000_track.json',
        landmarks_ref: 'sidecars/character_char_000_landmarks.json',
        mask_ref: 'sidecars/character_char_000_masks.json',
        quality: {
          score: 0.96,
          coverage: 1.0,
          warnings: [],
          errors: [],
        },
      },
    ],
    camera: {
      per_shot: [
        {
          camera_motion_id: 'cam_000',
          shot_id: 'shot_000',
          classification: 'panning_right',
          affine_ref: 'sidecars/camera_motion_affine.json',
          homography_ref: null,
          crop_ref: null,
          zoom_proxy_ref: null,
          shake_ref: null,
          background_tracks_ref: null,
          intrinsics: null,
          extrinsics_ref: null,
          reconstruction_backend: 'opencv_ransac_2d',
          confidence: 0.95,
          failure_reason: null,
        },
      ],
      quality: {
        score: 0.95,
        coverage: 1.0,
        warnings: [],
        errors: [],
      },
    },
    environment: {
      background_mask_ref: 'sidecars/environment_background_mask.json',
      depth_ref: null,
      luminance_ref: null,
      exposure_change_ref: null,
      white_balance_proxy_ref: null,
      blur_ref: null,
      occluder_tracks: [],
      quality: {
        score: 0.92,
        coverage: 1.0,
        warnings: [],
        errors: [],
      },
    },
    semantics: null,
    quality: {
      overall_score: 0.95,
      module_scores: {
        shots: 0.98,
        camera: 0.95,
        characters: 0.96,
        environment: 0.92,
      },
      warnings: [],
      errors: [],
      low_confidence_intervals: [],
      reproducible: true,
    },
    artifacts: {
      manifest_uri: 'blueprint.json',
      bundle_uri: 'bundle.zip',
      overlays: [
        {
          id: 'overlay_pose_001',
          kind: 'pose_skeleton',
          uri: 'artifacts/overlays/pose_overlay.png',
        },
        {
          id: 'overlay_camera_001',
          kind: 'camera_grid',
          uri: 'artifacts/overlays/camera_grid.png',
        },
      ],
      reports: [
        {
          id: 'report_validation_001',
          title: 'E0 Contract Validation Report',
          uri: 'artifacts/reports/validation.json',
        },
      ],
    },
    provenance: {
      tools: [
        {
          name: 'vbs-mock-pipeline',
          version: '1.0.0-e0',
          commit: 'e0-skeleton-commit',
        },
      ],
      system: {
        os: 'linux-x64',
        python: '3.11-compatibility-stub',
        node: process.version,
      },
    },
    extensions: {
      e0_deterministic_mock: true,
    },
  };
}
