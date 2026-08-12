import React, { useState } from 'react';
import { AnalysisJob, AnnotationOp } from '../types';
import { Edit3, Plus, Trash2, Save, History, CheckCircle2 } from 'lucide-react';

interface AnnotationsEditorViewProps {
  activeJob: AnalysisJob | null;
  onApplyPatch: (operations: AnnotationOp[]) => Promise<void>;
}

export const AnnotationsEditorView: React.FC<AnnotationsEditorViewProps> = ({ activeJob, onApplyPatch }) => {
  const [operations, setOperations] = useState<AnnotationOp[]>([
    { op: 'replace', path: '/shots/0/dominant_character_ids', value: '["char_000", "char_001"]' },
  ]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  if (!activeJob) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center text-slate-400 space-y-3">
        <Edit3 className="w-10 h-10 mx-auto text-slate-600" />
        <h3 className="text-base font-semibold text-slate-200">未选择标注任务</h3>
        <p className="text-xs">请排队或选择一个分析任务以应用 RFC 6902 手动标注补丁。</p>
      </div>
    );
  }

  const handleAddOp = () => {
    setOperations([...operations, { op: 'add', path: '/shots/0/quality/warnings', value: '["manual_flag"]' }]);
  };

  const handleRemoveOp = (idx: number) => {
    setOperations(operations.filter((_, i) => i !== idx));
  };

  const handleUpdateOp = (idx: number, field: keyof AnnotationOp, value: any) => {
    const updated = [...operations];
    updated[idx] = { ...updated[idx], [field]: value };
    setOperations(updated);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSuccessMessage(null);
    try {
      // Parse JSON string values if necessary
      const parsedOps = operations.map((op) => {
        let val = op.value;
        if (typeof val === 'string' && (val.startsWith('{') || val.startsWith('['))) {
          try {
            val = JSON.parse(val);
          } catch {
            // Keep as string if parsing fails
          }
        }
        return { op: op.op, path: op.path, value: val };
      });

      await onApplyPatch(parsedOps);
      setSuccessMessage('标注修订补丁已成功应用！');
      setTimeout(() => setSuccessMessage(null), 4000);
    } catch (err: any) {
      alert(`补丁应用失败: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <Edit3 className="w-5 h-5 text-cyan-400" />
            <h2 className="font-bold text-slate-100 text-base">手动标注与 JSON 补丁编辑器</h2>
          </div>
          <p className="text-xs text-slate-400 mt-0.5 font-mono">
            对镜头、轨迹或 ROI 应用 JSON Patch 操作 <code className="text-cyan-400">[add, replace, remove]</code>。
          </p>
        </div>

        <div className="text-xs font-mono text-slate-400">
          目标任务 ID：<span className="text-cyan-400">{activeJob.analysis_id.slice(0, 18)}...</span>
        </div>
      </div>

      {successMessage && (
        <div className="p-4 bg-emerald-950/80 border border-emerald-800 text-emerald-300 text-xs rounded-xl flex items-center space-x-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>{successMessage}</span>
        </div>
      )}

      {/* Operations Form */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <h3 className="font-semibold text-slate-200 text-sm">JSON Patch 操作列表 ({operations.length})</h3>
          <button
            type="button"
            onClick={handleAddOp}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-xs font-medium flex items-center space-x-1"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>添加操作</span>
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-3">
          {operations.map((op, idx) => (
            <div key={idx} className="p-3 bg-slate-950 border border-slate-800 rounded-lg grid grid-cols-1 md:grid-cols-12 gap-3 items-center text-xs">
              <div className="md:col-span-3">
                <label className="block text-[10px] text-slate-500 mb-1">操作类型</label>
                <select
                  value={op.op}
                  onChange={(e) => handleUpdateOp(idx, 'op', e.target.value as any)}
                  className="w-full px-2.5 py-1.5 bg-slate-900 border border-slate-800 rounded text-slate-200 font-mono text-xs"
                >
                  <option value="replace">replace (替换)</option>
                  <option value="add">add (添加)</option>
                  <option value="remove">remove (删除)</option>
                </select>
              </div>

              <div className="md:col-span-4">
                <label className="block text-[10px] text-slate-500 mb-1">目标路径</label>
                <input
                  type="text"
                  value={op.path}
                  onChange={(e) => handleUpdateOp(idx, 'path', e.target.value)}
                  placeholder="/shots/0/dominant_character_ids"
                  className="w-full px-2.5 py-1.5 bg-slate-900 border border-slate-800 rounded text-slate-200 font-mono text-xs"
                />
              </div>

              <div className="md:col-span-4">
                <label className="block text-[10px] text-slate-500 mb-1">属性值 (JSON 或字符串)</label>
                <input
                  type="text"
                  value={typeof op.value === 'object' ? JSON.stringify(op.value) : op.value || ''}
                  onChange={(e) => handleUpdateOp(idx, 'value', e.target.value)}
                  placeholder='["char_001"]'
                  className="w-full px-2.5 py-1.5 bg-slate-900 border border-slate-800 rounded text-slate-200 font-mono text-xs"
                />
              </div>

              <div className="md:col-span-1 text-right pt-4 md:pt-0">
                <button
                  type="button"
                  onClick={() => handleRemoveOp(idx)}
                  className="p-1.5 text-rose-400 hover:bg-rose-950/50 rounded transition"
                  title="删除操作"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}

          <button
            type="submit"
            disabled={isSubmitting || operations.length === 0}
            className="w-full py-2.5 px-4 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold rounded-lg text-xs flex items-center justify-center space-x-2 transition disabled:bg-slate-800"
          >
            <Save className="w-4 h-4" />
            <span>{isSubmitting ? '正在应用补丁...' : '应用补丁并创建修订版本'}</span>
          </button>
        </form>
      </div>

      {/* Revision History Log */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
        <h3 className="font-semibold text-slate-200 text-sm flex items-center space-x-2">
          <History className="w-4 h-4 text-cyan-400" />
          <span>标注修订历史日志 ({activeJob.annotations_history?.length || 0})</span>
        </h3>

        {activeJob.annotations_history && activeJob.annotations_history.length > 0 ? (
          <div className="space-y-2 font-mono text-xs">
            {activeJob.annotations_history.map((rev) => (
              <div key={rev.revision} className="p-3 bg-slate-950 border border-slate-800 rounded-lg space-y-1">
                <div className="flex justify-between text-slate-400">
                  <span className="text-cyan-400 font-semibold">修订版本 #{rev.revision}</span>
                  <span className="text-[11px]">{new Date(rev.timestamp).toLocaleTimeString()}</span>
                </div>
                <div className="text-slate-300 text-[11px]">应用操作数：{rev.operations.length}</div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-slate-500">该任务尚未应用手动标注。</p>
        )}
      </div>
    </div>
  );
};
