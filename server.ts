import express from 'express';
import cors from 'cors';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { execFileSync } from 'child_process';
import multer from 'multer';
import { v4 as uuidv4 } from 'uuid';
import { createServer as createViteServer } from 'vite';
import { VideoRecord, AnalysisJob, AnnotationOp } from './src/types.js';
import { createDefaultStages, generateDeterministicBlueprint, ALL_E0_MODULES } from './src/mockPipeline.js';
import { validateBlueprintObject } from './src/schemaValidator.js';

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());

const videosStore = new Map<string, VideoRecord>();
const jobsStore = new Map<string, AnalysisJob>();
const blueprintsStore = new Map<string, any>();
const bundlePathsStore = new Map<string, string>();
const validationReportsStore = new Map<string, any>();
const sseClients = new Map<string, express.Response[]>();

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 100 * 1024 * 1024 },
});

// E0 demo fixture is retained only for explicit sample/demo inspection.
// Real uploaded analyses must never fall back to this fixture or its metadata.
const sampleVideoId = '00000000-0000-4000-8000-000000000010';
const sampleJobId = '00000000-0000-4000-8000-000000000002';

const sampleVideo: VideoRecord = {
  video_id: sampleVideoId,
  file_name: 'sample_motion_source.mp4',
  sha256: '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08',
  status: 'validated',
  mime_type: 'video/mp4',
  file_size_bytes: 10485760,
  duration_us: 6000000,
  width: 1080,
  height: 1920,
  fps_avg: 30.0,
  created_at: new Date(Date.now() - 3600000).toISOString(),
  authorization_attested: true,
  adult_subject_attested: true,
};
videosStore.set(sampleVideoId, sampleVideo);

const sampleJob: AnalysisJob = {
  analysis_id: sampleJobId,
  video_id: sampleVideoId,
  status: 'succeeded',
  progress: 1.0,
  requested_modules: ALL_E0_MODULES,
  config_overrides: {},
  stages: createDefaultStages().map((s) => ({ ...s, status: 'succeeded', progress: 1.0 })),
  created_at: new Date(Date.now() - 3000000).toISOString(),
  started_at: new Date(Date.now() - 2900000).toISOString(),
  completed_at: new Date(Date.now() - 2800000).toISOString(),
};
jobsStore.set(sampleJobId, sampleJob);
blueprintsStore.set(sampleJobId, generateDeterministicBlueprint(sampleJob, sampleVideo));

function notifySseClients(analysisId: string, jobData: AnalysisJob) {
  const clients = sseClients.get(analysisId) || [];
  clients.forEach((res) => {
    res.write(`data: ${JSON.stringify(jobData)}\n\n`);
  });
}

function failJob(job: AnalysisJob, code: string, message: string) {
  job.status = 'failed';
  job.progress = 0;
  job.completed_at = new Date().toISOString();
  job.error = { code, message };
  job.stages = job.stages.map((stage) => ({
    ...stage,
    status: 'failed',
    progress: 0,
    completed_at: new Date().toISOString(),
    error_message: message,
  }));
  notifySseClients(job.analysis_id, job);
}

// E1 real-media runtime. This path is deliberately fail-closed: no E0/demo
// blueprint or bundle is generated if probing, normalization, validation, or
// export fails.
function runRealAnalysisPipelineAsync(jobId: string) {
  const job = jobsStore.get(jobId);
  if (!job) return;

  const video = videosStore.get(job.video_id);
  if (!video || !video.file_path || !fs.existsSync(video.file_path)) {
    failJob(job, 'REAL_MEDIA_REQUIRED', 'Persisted uploaded video file is required for real-media analysis.');
    return;
  }

  job.status = 'running';
  job.progress = 0.05;
  job.started_at = new Date().toISOString();
  job.error = undefined;
  job.stages = createDefaultStages();
  if (job.stages.length > 0) {
    job.stages[0].status = 'running';
    job.stages[0].started_at = new Date().toISOString();
  }
  notifySseClients(jobId, job);

  try {
    const runOutput = execFileSync('python3', [
      'scripts/run_pipeline_job.py',
      jobId,
      video.file_path,
      video.file_name,
      video.sha256,
    ], { encoding: 'utf-8' });

    const runRes = JSON.parse(runOutput);
    if (
      runRes.status !== 'succeeded' ||
      runRes.valid !== true ||
      !runRes.blueprint_path ||
      !fs.existsSync(runRes.blueprint_path) ||
      !runRes.bundle_path ||
      !fs.existsSync(runRes.bundle_path) ||
      !runRes.validation_report_path ||
      !fs.existsSync(runRes.validation_report_path)
    ) {
      const detail = runRes.error || (Array.isArray(runRes.errors) ? runRes.errors.join('; ') : 'Real-media pipeline did not return complete validated outputs.');
      failJob(job, 'REAL_MEDIA_PIPELINE_FAILED', detail);
      return;
    }

    const bpData = JSON.parse(fs.readFileSync(runRes.blueprint_path, 'utf-8'));
    const reportData = JSON.parse(fs.readFileSync(runRes.validation_report_path, 'utf-8'));
    if (reportData.valid !== true) {
      failJob(job, 'BUNDLE_VALIDATION_FAILED', 'Real-media blueprint validation failed.');
      return;
    }

    blueprintsStore.set(jobId, bpData);
    bundlePathsStore.set(jobId, runRes.bundle_path);
    validationReportsStore.set(jobId, reportData);

    job.status = 'succeeded';
    job.progress = 1.0;
    job.completed_at = new Date().toISOString();
    job.stages = createDefaultStages().map((stage) => ({
      ...stage,
      status: 'succeeded',
      progress: 1.0,
      started_at: job.started_at,
      completed_at: job.completed_at,
    }));
    notifySseClients(jobId, job);
  } catch (err: any) {
    console.error('Real-media Python pipeline execution error:', err);
    const stderr = typeof err?.stderr === 'string' ? err.stderr.trim() : '';
    const stdout = typeof err?.stdout === 'string' ? err.stdout.trim() : '';
    failJob(job, 'REAL_MEDIA_PIPELINE_FAILED', stderr || stdout || err?.message || 'Unknown real-media pipeline error');
  }
}

// --- API ENDPOINTS ---

app.post('/api/v1/videos', upload.single('file'), (req: express.Request, res: express.Response) => {
  const authAttested = req.body.authorization_attested === 'true' || req.body.authorization_attested === true;
  const adultAttested = req.body.adult_subject_attested === 'true' || req.body.adult_subject_attested === true;

  if (!authAttested || !adultAttested) {
    return res.status(400).json({
      code: 'ATTESTATION_REQUIRED',
      message: 'Both authorization_attested and adult_subject_attested must be true.',
    });
  }

  const file = req.file;
  if (!file) {
    return res.status(400).json({
      code: 'VIDEO_FILE_REQUIRED',
      message: 'A real uploaded video file is required.',
    });
  }

  const fileName = file.originalname || 'source_video.mp4';
  const videoId = uuidv4();
  const tempFilePath = path.join(os.tmpdir(), `vbs_upload_${videoId}_${fileName}`);
  fs.writeFileSync(tempFilePath, file.buffer);

  let probedData: any;
  try {
    const probeOutput = execFileSync('python3', ['scripts/probe_video.py', tempFilePath], { encoding: 'utf-8' });
    probedData = JSON.parse(probeOutput);
  } catch (err: any) {
    console.error('Python video probe execution error:', err);
    try { fs.unlinkSync(tempFilePath); } catch { /* best effort cleanup */ }
    return res.status(422).json({
      code: 'MEDIA_PROBE_FAILED',
      message: typeof err?.stderr === 'string' && err.stderr.trim() ? err.stderr.trim() : 'Uploaded video could not be probed.',
    });
  }

  const requiredProbeFields = ['sha256', 'file_size_bytes', 'duration_us', 'width', 'height', 'fps_avg'];
  const missingProbeFields = requiredProbeFields.filter((field) => probedData?.[field] === undefined || probedData?.[field] === null);
  if (missingProbeFields.length > 0) {
    try { fs.unlinkSync(tempFilePath); } catch { /* best effort cleanup */ }
    return res.status(422).json({
      code: 'MEDIA_PROBE_INCOMPLETE',
      message: `Probe output missing required fields: ${missingProbeFields.join(', ')}`,
    });
  }

  const record: VideoRecord = {
    video_id: videoId,
    file_name: fileName,
    sha256: probedData.sha256,
    status: 'validated',
    mime_type: probedData.mime_type || file.mimetype || 'video/mp4',
    file_size_bytes: probedData.file_size_bytes,
    duration_us: probedData.duration_us,
    width: probedData.width,
    height: probedData.height,
    fps_avg: probedData.fps_avg,
    created_at: new Date().toISOString(),
    authorization_attested: true,
    adult_subject_attested: true,
    file_path: tempFilePath,
  };

  videosStore.set(videoId, record);
  return res.status(201).json(record);
});

app.get('/api/v1/videos', (req, res) => {
  res.json(Array.from(videosStore.values()));
});

app.post('/api/v1/analyses', (req, res) => {
  const { video_id, modules, config_overrides } = req.body;
  if (!video_id || !videosStore.has(video_id)) {
    return res.status(400).json({
      code: 'INVALID_VIDEO_ID',
      message: 'Specified video_id does not exist.',
    });
  }

  const video = videosStore.get(video_id)!;
  if (!video.file_path || !fs.existsSync(video.file_path)) {
    return res.status(400).json({
      code: 'REAL_MEDIA_REQUIRED',
      message: 'Analysis requires a persisted video uploaded through /api/v1/videos.',
    });
  }

  const analysisId = uuidv4();
  const job: AnalysisJob = {
    analysis_id: analysisId,
    video_id,
    status: 'queued',
    progress: 0,
    requested_modules: modules || ALL_E0_MODULES,
    config_overrides: config_overrides || {},
    stages: createDefaultStages(),
    created_at: new Date().toISOString(),
  };

  jobsStore.set(analysisId, job);
  setTimeout(() => runRealAnalysisPipelineAsync(analysisId), 50);

  return res.status(202).json(job);
});

app.get('/api/v1/analyses', (req, res) => {
  res.json(Array.from(jobsStore.values()));
});

app.get('/api/v1/analyses/:analysis_id', (req, res) => {
  const job = jobsStore.get(req.params.analysis_id);
  if (!job) {
    return res.status(404).json({ code: 'NOT_FOUND', message: 'Analysis job not found.' });
  }
  return res.json(job);
});

app.post('/api/v1/analyses/:analysis_id/cancel', (req, res) => {
  const job = jobsStore.get(req.params.analysis_id);
  if (!job) {
    return res.status(404).json({ code: 'NOT_FOUND', message: 'Analysis job not found.' });
  }

  job.status = 'cancelled';
  notifySseClients(req.params.analysis_id, job);
  return res.status(202).json({ status: 'cancellation_requested', analysis_id: job.analysis_id });
});

app.post('/api/v1/analyses/:analysis_id/retry', (req, res) => {
  const job = jobsStore.get(req.params.analysis_id);
  if (!job) {
    return res.status(404).json({ code: 'NOT_FOUND', message: 'Analysis job not found.' });
  }

  const video = videosStore.get(job.video_id);
  if (!video || !video.file_path || !fs.existsSync(video.file_path)) {
    return res.status(400).json({
      code: 'REAL_MEDIA_REQUIRED',
      message: 'Retry requires the original persisted uploaded video.',
    });
  }

  job.status = 'queued';
  job.progress = 0;
  job.stages = createDefaultStages();
  job.error = undefined;
  notifySseClients(req.params.analysis_id, job);

  setTimeout(() => runRealAnalysisPipelineAsync(job.analysis_id), 50);
  return res.status(202).json({ status: 'retry_queued', analysis_id: job.analysis_id });
});

app.get('/api/v1/analyses/:analysis_id/blueprint', (req, res) => {
  const jobId = req.params.analysis_id;
  const job = jobsStore.get(jobId);
  if (!job) {
    return res.status(404).json({ code: 'NOT_FOUND', message: 'Analysis job not found.' });
  }
  if (job.status === 'failed') {
    return res.status(409).json({ code: job.error?.code || 'ANALYSIS_FAILED', message: job.error?.message || 'Analysis failed.' });
  }

  const bp = blueprintsStore.get(jobId);
  if (!bp) {
    return res.status(404).json({ code: 'NOT_FOUND', message: 'Blueprint not yet generated for this job.' });
  }

  return res.json(bp);
});

app.get('/api/v1/analyses/:analysis_id/bundle', (req, res) => {
  const jobId = req.params.analysis_id;
  const job = jobsStore.get(jobId);
  if (!job) {
    return res.status(404).json({ code: 'NOT_FOUND', message: 'Analysis job not found.' });
  }
  if (job.status === 'failed') {
    return res.status(409).json({ code: job.error?.code || 'ANALYSIS_FAILED', message: job.error?.message || 'Analysis failed.' });
  }

  const bundlePath = bundlePathsStore.get(jobId);
  if (!bundlePath || !fs.existsSync(bundlePath)) {
    return res.status(404).json({ code: 'NOT_FOUND', message: 'Validated real-media bundle not yet generated for this job.' });
  }

  return res.download(bundlePath, `bundle_${jobId.slice(0, 8)}.zip`);
});

app.get('/api/v1/analyses/:analysis_id/artifacts', (req, res) => {
  const jobId = req.params.analysis_id;
  const bp = blueprintsStore.get(jobId);

  return res.json({
    analysis_id: jobId,
    artifacts: bp
      ? bp.artifacts
      : {
          manifest_uri: 'blueprint.json',
          bundle_uri: 'bundle.zip',
          overlays: [],
          reports: [],
        },
  });
});

app.patch('/api/v1/analyses/:analysis_id/annotations', (req, res) => {
  const jobId = req.params.analysis_id;
  const job = jobsStore.get(jobId);
  const operations: AnnotationOp[] = req.body.operations;

  if (!job) {
    return res.status(404).json({ code: 'NOT_FOUND', message: 'Job not found.' });
  }

  if (!Array.isArray(operations)) {
    return res.status(400).json({ code: 'BAD_REQUEST', message: 'Operations array required.' });
  }

  if (!job.annotations_history) {
    job.annotations_history = [];
  }

  const revision = job.annotations_history.length + 1;
  job.annotations_history.push({
    revision,
    timestamp: new Date().toISOString(),
    operations,
  });

  return res.json({
    analysis_id: jobId,
    revision,
    applied_operations: operations.length,
    status: 'revision_created',
  });
});

app.post('/api/v1/analyses/:analysis_id/validate', (req, res) => {
  const jobId = req.params.analysis_id;
  const job = jobsStore.get(jobId);
  if (!job) {
    return res.status(404).json({ code: 'NOT_FOUND', message: 'Analysis job not found.' });
  }
  if (job.status === 'failed') {
    return res.status(409).json({ code: job.error?.code || 'ANALYSIS_FAILED', message: job.error?.message || 'Analysis failed.' });
  }

  const existingReport = validationReportsStore.get(jobId);
  if (existingReport) {
    return res.json(existingReport);
  }

  const bp = blueprintsStore.get(jobId);
  if (!bp) {
    return res.status(404).json({ code: 'NOT_FOUND', message: 'Blueprint not found.' });
  }

  const report = validateBlueprintObject(bp);
  return res.json(report);
});

app.get('/api/v1/analyses/:analysis_id/events', (req, res) => {
  const jobId = req.params.analysis_id;

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  if (!sseClients.has(jobId)) {
    sseClients.set(jobId, []);
  }
  sseClients.get(jobId)!.push(res);

  const job = jobsStore.get(jobId);
  if (job) {
    res.write(`data: ${JSON.stringify(job)}\n\n`);
  }

  req.on('close', () => {
    const list = sseClients.get(jobId) || [];
    sseClients.set(
      jobId,
      list.filter((client) => client !== res)
    );
  });
});

app.post('/api/v1/test/run-contract-tests', (req, res) => {
  try {
    const examplePath = path.join(process.cwd(), 'contracts', 'example_blueprint.json');
    if (!fs.existsSync(examplePath)) {
      return res.status(500).json({ success: false, error: 'example_blueprint.json missing' });
    }
    const exampleContent = JSON.parse(fs.readFileSync(examplePath, 'utf-8'));
    const report = validateBlueprintObject(exampleContent);
    return res.json({
      success: report.valid,
      report,
      message: report.valid ? 'example_blueprint.json validates perfectly against schema' : 'Validation failed',
    });
  } catch (err: any) {
    return res.status(500).json({ success: false, error: err.message });
  }
});

app.use('/api', (req: express.Request, res: express.Response) => {
  return res.status(404).json({
    code: 'NOT_FOUND',
    message: `API route ${req.method} ${req.originalUrl} not found.`,
  });
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    // Express 5/path-to-regexp no longer accepts the legacy `*` route pattern.
    // Use an unpathed middleware fallback so SPA navigation remains compatible
    // without intercepting unmatched /api requests handled above.
    app.use((req, res, next) => {
      if (req.method !== 'GET') {
        return next();
      }
      return res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Video Blueprint Studio M1 server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Server failed to start:', err);
});