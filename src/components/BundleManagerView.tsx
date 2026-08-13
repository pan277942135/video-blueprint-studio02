import React, { useState } from 'react';
import { Download, Package, FileCode, Image as ImageIcon, FileText, CheckCircle2 } from 'lucide-react';
import { BlueprintManifest } from '../types';

interface BundleManagerViewProps {
  blueprint: BlueprintManifest | null;
  onDownloadBundle: () => void;
}

export const BundleManagerView: React.FC<BundleManagerViewProps> = ({ blueprint, onDownloadBundle }) => {
  const [activeArtifactTab, setActiveArtifactTab] = useState<'all' | 'sidecars' | 'keyframes' | 'reports'>('all');

  if (!blueprint) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center text-slate-400 space-y-3">
        <Package className="w-10 h-10 mx-auto text-slate-600" />
        <h3 className="text-base font-semibold text-slate-200">未选择打包包</h3>
        <p className="text-xs">请排队或选择一个分析任务以查看和下载其 ZIP 打包包。</p>
      </div>
    );
  }

  const mockBundleContents = [
    { path: 'blueprint.json', size: '3.8 KB', kind: 'manifest', description: '规范视频蓝图 JSON' },
    { path: 'bundle_manifest.json', size: '1.2 KB', kind: 'manifest', description: '打包包 Manifest 与内容哈希' },
    { path: 'validation_report.json', size: '2.1 KB', kind: 'reports', description: 'Schema Draft 2020-12 校验报告' },
    { path: 'artifacts/timeseries/source_pts_map.npz', size: '14.5 KB', kind: 'sidecars', description: 'CFR 时间基准 PTS 帧映射 (NPZ)' },
    { path: 'sidecars/camera_motion_affine.json', size: '28.0 KB', kind: 'sidecars', description: '2D 相机运动仿射矩阵' },
    { path: 'sidecars/character_char_000_track.json', size: '42.1 KB', kind: 'sidecars', description: '匿名角色边界框追踪' },
    { path: 'sidecars/character_char_000_landmarks.json', size: '98.4 KB', kind: 'sidecars', description: '3D 身体/人脸/手部关节点标定' },
    { path: 'sidecars/environment_background_mask.json', size: '64.2 KB', kind: 'sidecars', description: 'RLE 背景掩码与遮挡' },
    { path: 'artifacts/keyframes/shot_000_000000.png', size: '120 KB', kind: 'keyframes', description: '镜头 0 关键帧 0 图像' },
    { path: 'artifacts/overlays/pose_overlay.png', size: '185 KB', kind: 'keyframes', description: '3D 骨骼姿态覆盖图渲染' },
  ];

  const filteredContents =
    activeArtifactTab === 'all' ? mockBundleContents : mockBundleContents.filter((item) => item.kind === activeArtifactTab);

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <Package className="w-5 h-5 text-cyan-400" />
            <h2 className="font-bold text-slate-100 text-base">可验证打包包管理器</h2>
          </div>
          <p className="text-xs text-slate-400 mt-0.5 font-mono">
            打包包路径: <code className="text-cyan-400">bundle_{blueprint.blueprint_id.slice(0, 8)}.zip</code>
          </p>
        </div>

        <button
          onClick={onDownloadBundle}
          className="px-4 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold rounded-lg text-xs flex items-center space-x-2 shadow-lg shadow-emerald-500/10 transition"
        >
          <Download className="w-4 h-4" />
          <span>下载完整 Bundle ZIP</span>
        </button>
      </div>

      {/* Filter Tabs */}
      <div className="flex space-x-2 border-b border-slate-800 pb-2">
        {[
          { id: 'all', label: '全部文件 (10)' },
          { id: 'sidecars', label: 'Sidecar 数据 (5)' },
          { id: 'keyframes', label: '关键帧与覆盖图 (2)' },
          { id: 'reports', label: 'Manifest 与报告 (3)' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveArtifactTab(tab.id as any)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
              activeArtifactTab === tab.id
                ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* File Tree List */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
        <h3 className="font-semibold text-slate-200 text-sm">Bundle 文件内容预览</h3>
        <div className="space-y-2 font-mono text-xs">
          {filteredContents.map((file, idx) => (
            <div
              key={idx}
              className="p-3 bg-slate-950 border border-slate-800/80 rounded-lg flex items-center justify-between hover:border-slate-700 transition"
            >
              <div className="flex items-center space-x-3">
                {file.kind === 'keyframes' ? (
                  <ImageIcon className="w-4 h-4 text-purple-400 shrink-0" />
                ) : file.kind === 'sidecars' ? (
                  <FileCode className="w-4 h-4 text-cyan-400 shrink-0" />
                ) : (
                  <FileText className="w-4 h-4 text-emerald-400 shrink-0" />
                )}

                <div>
                  <div className="text-slate-200 font-medium">{file.path}</div>
                  <div className="text-[11px] text-slate-500">{file.description}</div>
                </div>
              </div>

              <div className="flex items-center space-x-4">
                <span className="text-slate-500">{file.size}</span>
                <span className="text-emerald-400 text-[10px] bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800">
                  就绪
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Visual Artifact Preview Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <h3 className="font-semibold text-slate-200 text-sm">关键帧与骨骼覆盖图样例预览</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 text-center space-y-2">
            <div className="w-full h-36 bg-slate-900 rounded border border-slate-800 flex items-center justify-center text-cyan-400 font-mono text-xs">
              [ 分镜头 0 关键帧预览图 ]
            </div>
            <p className="text-xs text-slate-400">shot_000_000000.png &bull; 第 0 帧</p>
          </div>

          <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 text-center space-y-2">
            <div className="w-full h-36 bg-slate-900 rounded border border-slate-800 flex items-center justify-center text-purple-400 font-mono text-xs">
              [ 姿态与相机网格覆盖图 ]
            </div>
            <p className="text-xs text-slate-400">pose_overlay.png &bull; 3D 骨骼线框</p>
          </div>
        </div>
      </div>
    </div>
  );
};
