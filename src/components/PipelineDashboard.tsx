import React, { useState } from 'react';
import { VideoRecord, AnalysisJob } from '../types';
import { Upload, Play, CheckCircle2, AlertCircle, RefreshCw, XCircle, FileCode, Download, ShieldCheck, Cpu } from 'lucide-react';
import { ALL_E0_MODULES } from '../mockPipeline';

interface PipelineDashboardProps {
  videos: VideoRecord[];
  jobs: AnalysisJob[];
  selectedJobId: string | null;
  setSelectedJobId: (id: string) => void;
  onUploadVideo: (file: File | null, fileName: string, authAttested: boolean, adultAttested: boolean) => Promise<void>;
  onCreateAnalysis: (videoId: string, selectedModules: string[]) => Promise<void>;
  onCancelJob: (jobId: string) => Promise<void>;
  onRetryJob: (jobId: string) => Promise<void>;
  onViewBlueprint: (jobId: string) => void;
  onDownloadBundle: (jobId: string) => void;
}

export const PipelineDashboard: React.FC<PipelineDashboardProps> = ({
  videos,
  jobs,
  selectedJobId,
  setSelectedJobId,
  onUploadVideo,
  onCreateAnalysis,
  onCancelJob,
  onRetryJob,
  onViewBlueprint,
  onDownloadBundle,
}) => {
  // Upload State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [customFileName, setCustomFileName] = useState('');
  const [authAttested, setAuthAttested] = useState(true);
  const [adultAttested, setAdultAttested] = useState(true);
  const [isUploading, setIsUploading] = useState(false);

  // Job Creation State
  const [targetVideoId, setTargetVideoId] = useState<string>(videos[0]?.video_id || '');
  const [selectedModules, setSelectedModules] = useState<string[]>(ALL_E0_MODULES);
  const [isStartingJob, setIsStartingJob] = useState(false);

  const activeJob = jobs.find((j) => j.analysis_id === selectedJobId) || jobs[0];

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!authAttested || !adultAttested) {
      alert('上传前必须勾选并签署两项合规声明。');
      return;
    }
    setIsUploading(true);
    try {
      await onUploadVideo(selectedFile, customFileName || 'source_video.mp4', authAttested, adultAttested);
      setSelectedFile(null);
      setCustomFileName('');
    } catch (err: any) {
      alert(`上传失败: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleCreateAnalysisSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const vid = targetVideoId || videos[0]?.video_id;
    if (!vid) {
      alert('请先选择或上传视频。');
      return;
    }
    setIsStartingJob(true);
    try {
      await onCreateAnalysis(vid, selectedModules);
    } catch (err: any) {
      alert(`创建分析任务失败: ${err.message}`);
    } finally {
      setIsStartingJob(false);
    }
  };

  const toggleModule = (mod: string) => {
    if (selectedModules.includes(mod)) {
      setSelectedModules(selectedModules.filter((m) => m !== mod));
    } else {
      setSelectedModules([...selectedModules, mod]);
    }
  };

  return (
    <div className="space-[#12] space-y-8">
      {/* Top Banner Notice */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-start space-x-3">
        <Cpu className="w-5 h-5 text-cyan-400 mt-0.5 shrink-0" />
        <div className="text-sm">
          <span className="font-semibold text-slate-200">Epic E0 确定性分析目标流程：</span>{' '}
          <span className="text-slate-400">
            带签署证明上传视频 &rarr; 排队分析任务 &rarr; 执行确定性 Mock 流水线 &rarr; 生成合规{' '}
            <code className="text-cyan-400 font-mono text-xs">blueprint.json</code> &rarr; 打包可验证 <code className="text-cyan-400 font-mono text-xs">bundle.zip</code>。
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Upload & Job Creation */}
        <div className="lg:col-span-5 space-y-6">
          {/* Card 1: Upload Source Video */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
            <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
              <Upload className="w-5 h-5 text-cyan-400" />
              <h2 className="font-semibold text-slate-100">1. 上传源视频</h2>
            </div>

            <form onSubmit={handleUploadSubmit} className="space-y-4 text-sm">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">选择视频文件或模拟数据流</label>
                <input
                  type="file"
                  accept="video/*"
                  onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                  className="w-full text-xs text-slate-400 file:mr-3 file:py-2 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700 cursor-pointer"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">自定义文件名（可选）</label>
                <input
                  type="text"
                  placeholder="source_motion.mp4"
                  value={customFileName}
                  onChange={(e) => setCustomFileName(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-slate-200 focus:outline-none focus:border-cyan-500 text-xs"
                />
              </div>

              {/* Legal / Policy Attestations Required by Contract */}
              <div className="p-3 bg-slate-950/80 border border-slate-800/80 rounded-lg space-y-2 text-xs">
                <div className="flex items-center space-x-2 text-slate-300 font-medium">
                  <ShieldCheck className="w-4 h-4 text-cyan-400" />
                  <span>OpenAPI 必选合规声明</span>
                </div>
                <label className="flex items-center space-x-2 cursor-pointer text-slate-400">
                  <input
                    type="checkbox"
                    checked={authAttested}
                    onChange={(e) => setAuthAttested(e.target.checked)}
                    className="rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0"
                  />
                  <span>
                    <strong className="text-slate-300">authorization_attested:</strong> 已获得源视频授权
                  </span>
                </label>
                <label className="flex items-center space-x-2 cursor-pointer text-slate-400">
                  <input
                    type="checkbox"
                    checked={adultAttested}
                    onChange={(e) => setAdultAttested(e.target.checked)}
                    className="rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0"
                  />
                  <span>
                    <strong className="text-slate-300">adult_subject_attested:</strong> 视频主体均为成年人
                  </span>
                </label>
              </div>

              <button
                type="submit"
                disabled={isUploading}
                className="w-full py-2.5 px-4 bg-cyan-500 hover:bg-cyan-400 disabled:bg-slate-800 text-slate-950 font-semibold rounded-lg transition text-xs flex items-center justify-center space-x-2"
              >
                {isUploading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                <span>{isUploading ? '正在上传视频...' : '上传视频记录'}</span>
              </button>
            </form>
          </div>

          {/* Card 2: Create Analysis Job */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
            <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
              <Play className="w-5 h-5 text-cyan-400" />
              <h2 className="font-semibold text-slate-100">2. 创建分析任务</h2>
            </div>

            <form onSubmit={handleCreateAnalysisSubmit} className="space-y-4 text-sm">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">目标视频记录</label>
                <select
                  value={targetVideoId || (videos[0]?.video_id ?? '')}
                  onChange={(e) => setTargetVideoId(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-slate-200 focus:outline-none focus:border-cyan-500 text-xs font-mono"
                >
                  {videos.map((v) => (
                    <option key={v.video_id} value={v.video_id}>
                      {v.file_name} ({v.video_id.slice(0, 8)}...)
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-2">选择分析模块 ({selectedModules.length})</label>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {ALL_E0_MODULES.map((mod) => {
                    const isSelected = selectedModules.includes(mod);
                    return (
                      <button
                        type="button"
                        key={mod}
                        onClick={() => toggleModule(mod)}
                        className={`px-2.5 py-1.5 rounded-md text-left transition border font-mono ${
                          isSelected
                            ? 'bg-cyan-950/60 text-cyan-300 border-cyan-800/60'
                            : 'bg-slate-950 text-slate-500 border-slate-800 hover:text-slate-300'
                        }`}
                      >
                        {isSelected ? '✓ ' : '+ '}
                        {mod}
                      </button>
                    );
                  })}
                </div>
              </div>

              <button
                type="submit"
                disabled={isStartingJob || videos.length === 0}
                className="w-full py-2.5 px-4 bg-emerald-500 hover:bg-emerald-400 disabled:bg-slate-800 text-slate-950 font-semibold rounded-lg transition text-xs flex items-center justify-center space-x-2"
              >
                {isStartingJob ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
                <span>{isStartingJob ? '正在排队分析...' : '启动确定性 Mock 分析'}</span>
              </button>
            </form>
          </div>
        </div>

        {/* Right Column: Active Job Stages Tracker & Jobs Table */}
        <div className="lg:col-span-7 space-y-6">
          {/* Active Job Real-Time Tracker */}
          {activeJob ? (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-5">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div>
                  <div className="flex items-center space-x-3">
                    <h2 className="font-semibold text-slate-100 text-base">任务追踪</h2>
                    <span
                      className={`px-2.5 py-0.5 text-xs font-semibold rounded-full border ${
                        activeJob.status === 'succeeded'
                          ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                          : activeJob.status === 'running'
                          ? 'bg-cyan-950 text-cyan-400 border-cyan-800 animate-pulse'
                          : activeJob.status === 'cancelled'
                          ? 'bg-amber-950 text-amber-400 border-amber-800'
                          : 'bg-slate-800 text-slate-300 border-slate-700'
                      }`}
                    >
                      {activeJob.status === 'succeeded'
                        ? '已完成'
                        : activeJob.status === 'running'
                        ? '运行中'
                        : activeJob.status === 'cancelled'
                        ? '已取消'
                        : activeJob.status === 'failed'
                        ? '已失败'
                        : activeJob.status.toUpperCase()}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 font-mono mt-0.5">任务 ID: {activeJob.analysis_id}</p>
                </div>

                <div className="flex items-center space-x-2">
                  {activeJob.status === 'running' && (
                    <button
                      onClick={() => onCancelJob(activeJob.analysis_id)}
                      className="px-3 py-1.5 bg-rose-950/80 hover:bg-rose-900 text-rose-300 border border-rose-800 rounded-lg text-xs font-medium flex items-center space-x-1"
                    >
                      <XCircle className="w-3.5 h-3.5" />
                      <span>取消</span>
                    </button>
                  )}

                  {(activeJob.status === 'failed' || activeJob.status === 'cancelled' || activeJob.status === 'succeeded') && (
                    <button
                      onClick={() => onRetryJob(activeJob.analysis_id)}
                      className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-xs font-medium flex items-center space-x-1"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                      <span>重试阶段</span>
                    </button>
                  )}

                  <button
                    onClick={() => onViewBlueprint(activeJob.analysis_id)}
                    className="px-3 py-1.5 bg-cyan-950/80 hover:bg-cyan-900 text-cyan-300 border border-cyan-800 rounded-lg text-xs font-medium flex items-center space-x-1"
                  >
                    <FileCode className="w-3.5 h-3.5" />
                    <span>蓝图</span>
                  </button>

                  <button
                    onClick={() => onDownloadBundle(activeJob.analysis_id)}
                    className="px-3 py-1.5 bg-emerald-950/80 hover:bg-emerald-900 text-emerald-300 border border-emerald-800 rounded-lg text-xs font-medium flex items-center space-x-1"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>下载 ZIP 包</span>
                  </button>
                </div>
              </div>

              {/* Progress Bar */}
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>总流水线进度</span>
                  <span className="font-mono text-cyan-400">{Math.round(activeJob.progress * 100)}%</span>
                </div>
                <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className="bg-gradient-to-r from-cyan-500 to-emerald-500 h-full transition-all duration-300"
                    style={{ width: `${Math.round(activeJob.progress * 100)}%` }}
                  />
                </div>
              </div>

              {/* Stage Stepper List */}
              <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
                {activeJob.stages.map((stage, idx) => {
                  const isDone = stage.status === 'succeeded';
                  const isRunning = stage.status === 'running';

                  return (
                    <div
                      key={stage.stage_id}
                      className={`p-3 rounded-lg border flex items-center justify-between text-xs transition ${
                        isDone
                          ? 'bg-slate-950/60 border-slate-800 text-slate-300'
                          : isRunning
                          ? 'bg-cyan-950/40 border-cyan-800/80 text-cyan-200'
                          : 'bg-slate-950/20 border-slate-900 text-slate-500'
                      }`}
                    >
                      <div className="flex items-center space-x-3">
                        <span className="font-mono text-slate-500 w-6">0{idx + 1}</span>
                        {isDone ? (
                          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                        ) : isRunning ? (
                          <RefreshCw className="w-4 h-4 text-cyan-400 animate-spin shrink-0" />
                        ) : (
                          <AlertCircle className="w-4 h-4 text-slate-600 shrink-0" />
                        )}
                        <div>
                          <p className="font-medium">{stage.name}</p>
                          <p className="text-[10px] text-slate-500 font-mono">{stage.stage_id}</p>
                        </div>
                      </div>

                      <span
                        className={`font-mono text-[11px] px-2 py-0.5 rounded ${
                          isDone
                            ? 'bg-emerald-950/80 text-emerald-400'
                            : isRunning
                            ? 'bg-cyan-950/80 text-cyan-400'
                            : 'bg-slate-900 text-slate-600'
                        }`}
                      >
                        {stage.status === 'succeeded'
                          ? '已完成'
                          : stage.status === 'running'
                          ? '进行中'
                          : stage.status === 'pending'
                          ? '等待中'
                          : stage.status === 'failed'
                          ? '已失败'
                          : stage.status}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 text-center text-slate-500 text-sm">
              暂无分析任务。请先上传视频并创建任务。
            </div>
          )}

          {/* Historical Jobs List */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
            <h3 className="font-semibold text-slate-200 text-sm">所有分析任务 ({jobs.length})</h3>
            <div className="space-y-2">
              {jobs.map((j) => (
                <div
                  key={j.analysis_id}
                  onClick={() => setSelectedJobId(j.analysis_id)}
                  className={`p-3 rounded-lg border cursor-pointer flex items-center justify-between text-xs transition ${
                    j.analysis_id === activeJob?.analysis_id
                      ? 'bg-slate-800/80 border-cyan-500/50 text-slate-100'
                      : 'bg-slate-950 border-slate-800/80 text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                  }`}
                >
                  <div>
                    <div className="font-mono font-medium text-slate-200">{j.analysis_id.slice(0, 18)}...</div>
                    <div className="text-[11px] text-slate-500 mt-0.5">
                      创建时间：{new Date(j.created_at).toLocaleTimeString()} &bull; 模块数：{j.requested_modules.length}
                    </div>
                  </div>

                  <div className="flex items-center space-x-3">
                    <span className="font-mono text-cyan-400">{Math.round(j.progress * 100)}%</span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                        j.status === 'succeeded' ? 'bg-emerald-950 text-emerald-400' : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {j.status === 'succeeded'
                        ? '成功'
                        : j.status === 'running'
                        ? '运行中'
                        : j.status === 'cancelled'
                        ? '已取消'
                        : j.status === 'failed'
                        ? '失败'
                        : j.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
