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
import { generateBundleZipBuffer } from './src/bundleGenerator.js';

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());

// In-memory databases for E0 mock service
const videosStore = new Map<string, VideoRecord>();
const jobsStore = new Map<string, AnalysisJob>();
const blueprintsStore = new Map<string, any>();
const bundlePathsStore = new Map<string, string>();
const validationReportsStore = new Map<string, any>();
const sseClients = new Map<string, express.Response[]>();

// Configure multer for video upload
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 100 * 1024 * 1024 }, // 100MB limit for E0
});

// Seed default sample video & job for instant testing out of the box
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

// Helper: Notify SSE Clients
function notifySseClients(analysisId: string, jobData: AnalysisJob) {
  const clients = sseClients.get(analysisId) || [];
  clients.forEach((res) => {
    res.write(`data: ${JSON.stringify(jobData)}\n\n`);
  });
}

// Background simulator to advance job progress step-by-step
function runMockAnalysisPipelineAsync(jobId: string) {
  const job = jobsStore.get(jobId);
  if (!job) return;

  job.status = 'running';
  job.started_at = new Date().toISOString();
  notifySseClients(jobId, job);

  let currentStageIndex = 0;
  const interval = setInterval(() => {
    const currentJob = jobsStore.get(jobId);
    if (!currentJob || currentJob.status === 'cancelled' || currentJob.status === 'failed') {
      clearInterval(interval);
      return;
    }

    if (currentStageIndex < currentJob.stages.length) {
      const stage = currentJob.stages[currentStageIndex];
      stage.status = 'running';
      stage.progress = 1.0;
      stage.started_at = new Date().toISOString();
      stage.completed_at = new Date().toISOString();
      stage.status = 'succeeded';

      currentStageIndex++;
      currentJob.progress = Number((currentStageIndex / currentJob.stages.length).toFixed(2));
      notifySseClients(jobId, currentJob);
    } else {
      clearInterval(interval);
      currentJob.status = 'succeeded';
      currentJob.progress = 1.0;
      currentJob.completed_at = new Date().toISOString();

      const video = videosStore.get(currentJob.video_id);
      if (video) {
        try {
          const runOutput = execFileSync('python3', [
            'scripts/run_pipeline_job.py',
            jobId,
            video.file_path || '',
            video.file_name,
            video.sha256
          ], { encoding: 'utf-8' });

          const runRes = JSON.parse(runOutput);
          if (runRes.status === 'succeeded' && runRes.blueprint_path && fs.existsSync(runRes.blueprint_path)) {
            const bpData = JSON.parse(fs.readFileSync(runRes.blueprint_path, 'utf-8'));
            blueprintsStore.set(jobId, bpData);
            if (runRes.bundle_path) {
              bundlePathsStore.set(jobId, runRes.bundle_path);
            }
            if (runRes.validation_report_path && fs.existsSync(runRes.validation_report_path)) {
              validationReportsStore.set(jobId, JSON.parse(fs.readFileSync(runRes.validation_report_path, 'utf-8')));
            }
          } else {
            const bp = generateDeterministicBlueprint(currentJob, video);
            blueprintsStore.set(jobId, bp);
          }
        } catch (err) {
          console.error('Python pipeline execution error:', err);
          const bp = generateDeterministicBlueprint(currentJob, video);
          blueprintsStore.set(jobId, bp);
        }
      }

      notifySseClients(jobId, currentJob);
    }
  }, 600);
}

// --- API ENDPOINTS ---

// POST /api/v1/videos
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
  const fileName = file ? file.originalname : req.body.file_name || 'source_video.mp4';
  const videoId = uuidv4();
  let tempFilePath = '';

  if (file) {
    tempFilePath = path.join(os.tmpdir(), `vbs_upload_${videoId}_${fileName}`);
    fs.writeFileSync(tempFilePath, file.buffer);
  }

  let probedData: any = {};
  if (tempFilePath && fs.existsSync(tempFilePath)) {
    try {
      const probeOutput = execFileSync('python3', ['scripts/probe_video.py', tempFilePath], { encoding: 'utf-8' });
      probedData = JSON.parse(probeOutput);
    } catch (err) {
      console.error('Python video probe execution error:', err);
    }
  }

  const record: VideoRecord = {
    video_id: videoId,
    file_name: fileName,
    sha256: probedData.sha256 || (file ? 'mock_sha256_' + uuidv4().slice(0, 8) : 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
    status: 'validated',
    mime_type: probedData.mime_type || (file ? file.mimetype : 'video/mp4'),
    file_size_bytes: probedData.file_size_bytes || (file ? file.size : 15420000),
    duration_us: probedData.duration_us || 10000000,
    width: probedData.width || 1920,
    height: probedData.height || 1080,
    fps_avg: probedData.fps_avg || 30.0,
    created_at: new Date().toISOString(),
    authorization_attested: true,
    adult_subject_attested: true,
    file_path: tempFilePath,
  };

  videosStore.set(videoId, record);
  return res.status(201).json(record);
});

// GET /api/v1/videos
app.get('/api/v1/videos', (req, res) => {
  res.json(Array.from(videosStore.values()));
});

// POST /api/v1/analyses
app.post('/api/v1/analyses', (req, res) => {
  const { video_id, modules, config_overrides } = req.body;
  if (!video_id || !videosStore.has(video_id)) {
    return res.status(400).json({
      code: 'INVALID_VIDEO_ID',
      message: 'Specified video_id does not exist.',
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

  // Trigger deterministic async execution
  setTimeout(() => runMockAnalysisPipelineAsync(analysisId), 300);

  return res.status(202).json(job);
});

// GET /api/v1/analyses
app.get('/api/v1/analyses', (req, res) => {
  res.json(Array.from(jobsStore.values()));
});

// GET /api/v1/analyses/:analysis_id
app.get('/api/v1/analyses/:analysis_id', (req, res) => {
  const job = jobsStore.get(req.params.analysis_id);
  if (!job) {
    return res.status(404).json({ code: 'NOT_FOUND', message: 'Analysis job not found.' });
  }
  return res.json(job);
});

// POST /api/v1/analyses/:analysis_id/cancel
app.post('/api/v1/analyses/:analysis_id/cancel', (req, res) => {
  const job = jobsStore.get(req.params.analysis_id);
  if (!job) {
    return res.status(404).json({ code: 'NOT_FOUND', message: 'Analysis job not found.' });
  }

  job.status = 'cancelled';
  notifySseClients(req.params.analysis_id, job);
  return res.status(202).json({ status: 'cancellation_requested', analysis_id: job.analysis_id });
});

// POST /api/v1/analyses/:analysis_id/retry
app.post('/api/v1/analyses/:analysis_id/retry', (req, res) => {
  const job = jobsStore.get(req.params.analysis_id);
  if (!job) {
    return res.status(404).json({ code: 'NOT_FOUND', message: 'Analysis job not found.' });
  }

  job.status = 'queued';
  job.progress = 0;
  job.stages = createDefaultStages();
  job.error = undefined;
  notifySseClients(req.params.analysis_id, job);

  setTimeout(() => runMockAnalysisPipelineAsync(job.analysis_id), 300);
  return res.status(202).json({ status: 'retry_queued', analysis_id: job.analysis_id });
});

// GET /api/v1/analyses/:analysis_id/blueprint
app.get('/api/v1/analyses/:analysis_id/blueprint', (req, res) => {
  const jobId = req.params.analysis_id;
  const bp = blueprintsStore.get(jobId);

  if (!bp) {
    const job = jobsStore.get(jobId);
    if (job) {
      const video = videosStore.get(job.video_id);
      if (video) {
        const generatedBp = generateDeterministicBlueprint(job, video);
        blueprintsStore.set(jobId, generatedBp);
        return res.json(generatedBp);
      }
    }
    return res.status(404).json({ code: 'NOT_FOUND', message: 'Blueprint not yet generated for this job.' });
  }

  return res.json(bp);
});

// GET /api/v1/analyses/:analysis_id/bundle
app.get('/api/v1/analyses/:analysis_id/bundle', async (req, res) => {
  const jobId = req.params.analysis_id;

  const bundlePath = bundlePathsStore.get(jobId);
  if (bundlePath && fs.existsSync(bundlePath)) {
    return res.download(bundlePath, `bundle_${jobId.slice(0, 8)}.zip`);
  }

  let bp = blueprintsStore.get(jobId);

  if (!bp) {
    const job = jobsStore.get(jobId);
    if (job) {
      const video = videosStore.get(job.video_id);
      if (video) {
        bp = generateDeterministicBlueprint(job, video);
        blueprintsStore.set(jobId, bp);
      }
    }
  }

  if (!bp) {
    return res.status(404).json({ code: 'NOT_FOUND', message: 'Blueprint not found to generate bundle.' });
  }

  const report = validateBlueprintObject(bp);
  const zipBuffer = await generateBundleZipBuffer(bp, report);

  res.setHeader('Content-Type', 'application/zip');
  res.setHeader('Content-Disposition', `attachment; filename="bundle_${jobId.slice(0, 8)}.zip"`);
  return res.send(zipBuffer);
});

// GET /api/v1/analyses/:analysis_id/artifacts
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

// PATCH /api/v1/analyses/:analysis_id/annotations
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

// POST /api/v1/analyses/:analysis_id/validate
app.post('/api/v1/analyses/:analysis_id/validate', (req, res) => {
  const jobId = req.params.analysis_id;

  const existingReport = validationReportsStore.get(jobId);
  if (existingReport) {
    return res.json(existingReport);
  }

  let bp = blueprintsStore.get(jobId);

  if (!bp) {
    const job = jobsStore.get(jobId);
    if (job) {
      const video = videosStore.get(job.video_id);
      if (video) {
        bp = generateDeterministicBlueprint(job, video);
        blueprintsStore.set(jobId, bp);
      }
    }
  }

  if (!bp) {
    return res.status(404).json({ code: 'NOT_FOUND', message: 'Blueprint not found.' });
  }

  const report = validateBlueprintObject(bp);
  return res.json(report);
});

// GET /api/v1/analyses/:analysis_id/events
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

// Helper POST /api/v1/test/run-contract-tests
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

// JSON 404 handler for unmatched API routes
app.use('/api', (req: express.Request, res: express.Response) => {
  return res.status(404).json({
    code: 'NOT_FOUND',
    message: `API route ${req.method} ${req.originalUrl} not found.`,
  });
});

// Mount Vite middleware in development or static serve in production
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
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Video Blueprint Studio M1 server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Server failed to start:', err);
});
