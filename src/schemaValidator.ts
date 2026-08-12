import Ajv2020 from 'ajv/dist/2020.js';
import addFormats from 'ajv-formats';
import fs from 'fs';
import path from 'path';
import { ValidationReport, BlueprintManifest } from './types';

let ajvValidator: any = null;

function getValidator() {
  if (ajvValidator) return ajvValidator;

  const ajv = new Ajv2020({
    allErrors: true,
    strict: false,
    validateFormats: true,
  });
  addFormats(ajv);

  const schemaPath = path.join(process.cwd(), 'contracts', 'video_blueprint.schema.json');
  if (fs.existsSync(schemaPath)) {
    const schemaContent = JSON.parse(fs.readFileSync(schemaPath, 'utf-8'));
    ajvValidator = ajv.compile(schemaContent);
  } else {
    throw new Error(`Schema file not found at ${schemaPath}`);
  }

  return ajvValidator;
}

export function validateBlueprintObject(blueprint: any): ValidationReport {
  const validator = getValidator();
  const valid = validator(blueprint);

  const errors = (validator.errors || []).map((err: any) => ({
    instancePath: err.instancePath || '',
    schemaPath: err.schemaPath || '',
    keyword: err.keyword || '',
    params: err.params || {},
    message: err.message || '',
  }));

  // Perform engineering invariant checks
  const invariantChecks: { name: string; status: 'pass' | 'fail' | 'warn'; detail: string }[] = [];

  // Invariant 1: Manifest + Sidecar check (no giant embedded arrays in manifest root)
  const isManifestSidecar = !blueprint.dense_motion_vectors && !blueprint.raw_frames;
  invariantChecks.push({
    name: 'Manifest + Sidecar Separation',
    status: isManifestSidecar ? 'pass' : 'fail',
    detail: isManifestSidecar
      ? 'Manifest contains references only, keeping payload lightweight.'
      : 'Violation: Found embedded dense array in manifest root.',
  });

  // Invariant 2: Character Anonymous Track Check
  const chars = blueprint.characters || [];
  const hasIdentityEmbeddings = chars.some((c: any) => c.biometric_embedding || c.identity_name);
  invariantChecks.push({
    name: 'Anonymous Character Tracking (No Biometrics)',
    status: !hasIdentityEmbeddings ? 'pass' : 'fail',
    detail: !hasIdentityEmbeddings
      ? 'Characters are strictly anonymous track IDs.'
      : 'Violation: Biometric identity embeddings or identity names detected!',
  });

  // Invariant 3: Timebase CFR normalization check
  const timebase = blueprint.timebase || {};
  const isTimebaseValid = timebase.normalized_to_cfr === true && timebase.fps_num > 0 && timebase.fps_den > 0;
  invariantChecks.push({
    name: 'CFR Timebase Normalization',
    status: isTimebaseValid ? 'pass' : 'warn',
    detail: isTimebaseValid
      ? `Timebase normalized to CFR at ${timebase.fps_num / timebase.fps_den} FPS.`
      : 'Timebase normalization check incomplete.',
  });

  // Invariant 4: Quality & Provenance Traceability
  const hasProvenance = !!blueprint.provenance && !!blueprint.quality;
  invariantChecks.push({
    name: 'Provenance and Quality Auditability',
    status: hasProvenance ? 'pass' : 'fail',
    detail: hasProvenance
      ? 'Quality scores and system provenance fully present.'
      : 'Missing provenance or quality record.',
  });

  const passedCount = invariantChecks.filter((c) => c.status === 'pass').length;
  const failedCount = invariantChecks.filter((c) => c.status === 'fail').length;

  return {
    valid: valid && failedCount === 0,
    schema_version: blueprint.schema_version || '1.0.0',
    validated_at: new Date().toISOString(),
    errors,
    warnings: [],
    summary: {
      passed_rules: passedCount,
      failed_rules: errors.length + failedCount,
      invariant_checks: invariantChecks,
    },
  };
}
