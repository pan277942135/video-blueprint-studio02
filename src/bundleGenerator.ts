import JSZip from 'jszip';
import { BlueprintManifest, ValidationReport } from './types';

export async function generateBundleZipBuffer(
  blueprint: BlueprintManifest,
  validationReport?: ValidationReport
): Promise<Buffer> {
  const zip = new JSZip();

  // 1. Add canonical manifest JSON
  zip.file('blueprint.json', JSON.stringify(blueprint, null, 2));

  // 2. Add bundle manifest summary
  const bundleManifest = {
    bundle_version: '1.0.0',
    created_at: new Date().toISOString(),
    blueprint_id: blueprint.blueprint_id,
    job_id: blueprint.processing?.job_id,
    source_video_name: blueprint.source_video?.file_name,
    contents: [
      'blueprint.json',
      'bundle_manifest.json',
      'validation_report.json',
      'sidecars/pts_map.json',
      'sidecars/camera_motion_affine.json',
      'sidecars/character_char_000_track.json',
      'sidecars/character_char_000_landmarks.json',
      'sidecars/environment_background_mask.json',
      'artifacts/keyframes/shot_000_000000.png',
      'artifacts/overlays/pose_overlay.png',
    ],
  };
  zip.file('bundle_manifest.json', JSON.stringify(bundleManifest, null, 2));

  // 3. Add validation report
  if (validationReport) {
    zip.file('validation_report.json', JSON.stringify(validationReport, null, 2));
  } else {
    zip.file('validation_report.json', JSON.stringify({ status: 'valid', timestamp: new Date().toISOString() }, null, 2));
  }

  // 4. Add mock sidecar files
  const sidecarsFolder = zip.folder('sidecars');
  if (sidecarsFolder) {
    sidecarsFolder.file(
      'pts_map.json',
      JSON.stringify({ type: 'timebase_pts_map', frame_count: blueprint.timebase?.frame_count || 180, pts: Array.from({ length: 180 }, (_, i) => i * 33333) }, null, 2)
    );
    sidecarsFolder.file(
      'camera_motion_affine.json',
      JSON.stringify({ camera_motion_id: 'cam_000', motion_type: 'panning_right', matrices: [{ frame: 0, matrix: [1, 0, 0, 0, 1, 0] }] }, null, 2)
    );
    sidecarsFolder.file(
      'character_char_000_track.json',
      JSON.stringify({ character_id: 'char_000', bbox_track: [{ frame: 0, bbox: [100, 200, 300, 800], confidence: 0.98 }] }, null, 2)
    );
    sidecarsFolder.file(
      'character_char_000_landmarks.json',
      JSON.stringify({ character_id: 'char_000', format: 'coco_17', keypoints_3d: [] }, null, 2)
    );
    sidecarsFolder.file(
      'environment_background_mask.json',
      JSON.stringify({ type: 'background_segmentation', compression: 'rle', mask_ref: 'bg_001.rle' }, null, 2)
    );
  }

  // 5. Add mock artifact keyframe images / overlays (SVG/PNG placeholders)
  const artifactsFolder = zip.folder('artifacts');
  if (artifactsFolder) {
    const keyframesFolder = artifactsFolder.folder('keyframes');
    if (keyframesFolder) {
      keyframesFolder.file(
        'shot_000_000000.png',
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180" viewBox="0 0 320 180"><rect width="320" height="180" fill="#0f172a"/><text x="160" y="90" fill="#06b6d4" font-family="sans-serif" font-size="14" text-anchor="middle">VBS Keyframe Shot 0 Frame 0</text></svg>'
      );
    }
    const overlaysFolder = artifactsFolder.folder('overlays');
    if (overlaysFolder) {
      overlaysFolder.file(
        'pose_overlay.png',
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180" viewBox="0 0 320 180"><rect width="320" height="180" fill="#020617"/><circle cx="160" cy="50" r="10" fill="#a855f7"/><line x1="160" y1="60" x2="160" y2="120" stroke="#a855f7" stroke-width="4"/><line x1="160" y1="80" x2="120" y2="110" stroke="#a855f7" stroke-width="3"/><line x1="160" y1="80" x2="200" y2="110" stroke="#a855f7" stroke-width="3"/><text x="160" y="160" fill="#e2e8f0" font-family="sans-serif" font-size="12" text-anchor="middle">Pose Landmarks Overlay</text></svg>'
      );
    }
  }

  const zipContent = await zip.generateAsync({ type: 'nodebuffer' });
  return zipContent;
}
