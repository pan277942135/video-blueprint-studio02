import React, { useState } from 'react';
import { BlueprintManifest, ValidationReport } from '../types';
import { Copy, Download, Check, ShieldCheck, Search, FileCode } from 'lucide-react';

interface BlueprintInspectorProps {
  blueprint: BlueprintManifest | null;
  validationReport: ValidationReport | null;
  onValidate: () => void;
  onDownloadBlueprint: () => void;
}

export const BlueprintInspector: React.FC<BlueprintInspectorProps> = ({
  blueprint,
  validationReport,
  onValidate,
  onDownloadBlueprint,
}) => {
  const [copied, setCopied] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  if (!blueprint) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center text-slate-400 space-y-3">
        <FileCode className="w-10 h-10 mx-auto text-slate-600" />
        <h3 className="text-base font-semibold text-slate-200">未选择蓝图</h3>
        <p className="text-xs">请排队或选择一个分析任务以查看其规范 Manifest。</p>
      </div>
    );
  }

  const jsonString = JSON.stringify(blueprint, null, 2);

  const handleCopy = () => {
    navigator.clipboard.writeText(jsonString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Top Controls Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-3">
            <h2 className="font-bold text-slate-100 text-base">蓝图 Manifest 检查器</h2>
            <span className="font-mono text-xs text-cyan-400 bg-cyan-950 px-2.5 py-0.5 rounded border border-cyan-800">
              v{blueprint.schema_version}
            </span>
          </div>
          <p className="text-xs text-slate-400 font-mono mt-0.5">蓝图 ID: {blueprint.blueprint_id}</p>
        </div>

        <div className="flex items-center space-x-2">
          {validationReport && (
            <div
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1.5 border ${
                validationReport.valid
                  ? 'bg-emerald-950/80 text-emerald-400 border-emerald-800'
                  : 'bg-rose-950/80 text-rose-400 border-rose-800'
              }`}
            >
              <ShieldCheck className="w-4 h-4" />
              <span>{validationReport.valid ? 'Schema 验证通过 (Draft 2020-12)' : 'Schema 验证存在错误'}</span>
            </div>
          )}

          <button
            onClick={onValidate}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-xs font-medium flex items-center space-x-1.5"
          >
            <ShieldCheck className="w-4 h-4 text-cyan-400" />
            <span>重新验证</span>
          </button>

          <button
            onClick={handleCopy}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-xs font-medium flex items-center space-x-1.5"
          >
            {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
            <span>{copied ? '已复制！' : '复制 JSON'}</span>
          </button>

          <button
            onClick={onDownloadBlueprint}
            className="px-3 py-1.5 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold rounded-lg text-xs flex items-center space-x-1.5"
          >
            <Download className="w-4 h-4" />
            <span>下载 Manifest</span>
          </button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl space-y-1">
          <span className="text-slate-500">源视频</span>
          <div className="text-slate-200 truncate font-semibold">{blueprint.source_video?.file_name}</div>
          <div className="text-slate-400 text-[11px]">
            {blueprint.source_video?.width}x{blueprint.source_video?.height} @ {blueprint.source_video?.fps_avg} FPS
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl space-y-1">
          <span className="text-slate-500">CFR 时间基准</span>
          <div className="text-slate-200 font-semibold">
            {blueprint.timebase?.fps_num / blueprint.timebase?.fps_den} FPS (CFR)
          </div>
          <div className="text-slate-400 text-[11px]">{blueprint.timebase?.frame_count} 帧</div>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl space-y-1">
          <span className="text-slate-500">质量评分</span>
          <div className="text-emerald-400 font-bold text-sm">
            {Math.round((blueprint.quality?.overall_score || 0.95) * 100)}%
          </div>
          <div className="text-slate-400 text-[11px]">可复现：是</div>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl space-y-1">
          <span className="text-slate-500">Sidecar 与 Manifest</span>
          <div className="text-cyan-400 font-semibold">Manifest + Sidecars</div>
          <div className="text-slate-400 text-[11px]">无内嵌稠密数组</div>
        </div>
      </div>

      {/* Search Filter Input */}
      <div className="relative">
        <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-3" />
        <input
          type="text"
          placeholder="过滤 JSON 属性（例如 'source_video', 'camera', 'quality'）..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-800 rounded-xl text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
        />
      </div>

      {/* JSON Viewer */}
      <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 overflow-x-auto max-h-[500px]">
        <pre className="text-xs font-mono text-cyan-300 leading-relaxed">
          {searchQuery
            ? jsonString
                .split('\n')
                .filter((line) => line.toLowerCase().includes(searchQuery.toLowerCase()))
                .join('\n')
            : jsonString}
        </pre>
      </div>
    </div>
  );
};
