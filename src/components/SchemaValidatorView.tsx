import React from 'react';
import { ValidationReport } from '../types';
import { ShieldCheck, CheckCircle2, XCircle, AlertTriangle, RefreshCw, Cpu, FileJson } from 'lucide-react';

interface SchemaValidatorViewProps {
  report: ValidationReport | null;
  isValidating: boolean;
  onRunValidation: () => void;
  onRunExampleContractTest: () => void;
}

export const SchemaValidatorView: React.FC<SchemaValidatorViewProps> = ({
  report,
  isValidating,
  onRunValidation,
  onRunExampleContractTest,
}) => {
  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <ShieldCheck className="w-5 h-5 text-cyan-400" />
            <h2 className="font-bold text-slate-100 text-base">Schema 规范与工程不变性引擎</h2>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            根据 <code className="text-cyan-400 font-mono">contracts/video_blueprint.schema.json</code> (Draft 2020-12) 及 Epic E0 规则校验 Manifest。
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={onRunExampleContractTest}
            className="px-3.5 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition"
          >
            <FileJson className="w-4 h-4 text-cyan-400" />
            <span>校验 example_blueprint.json</span>
          </button>

          <button
            onClick={onRunValidation}
            disabled={isValidating}
            className="px-4 py-2 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold rounded-lg text-xs flex items-center space-x-1.5 transition disabled:bg-slate-800"
          >
            {isValidating ? <RefreshCw className="w-4 h-4 animate-spin" /> : <ShieldCheck className="w-4 h-4" />}
            <span>{isValidating ? '正在校验...' : '校验当前任务蓝图'}</span>
          </button>
        </div>
      </div>

      {/* Validation Result Summary Header */}
      {report && (
        <div className="space-y-6">
          <div
            className={`p-5 rounded-xl border flex items-center justify-between ${
              report.valid
                ? 'bg-emerald-950/40 border-emerald-800/80 text-emerald-300'
                : 'bg-rose-950/40 border-rose-800/80 text-rose-300'
            }`}
          >
            <div className="flex items-center space-x-4">
              <div
                className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                  report.valid ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'
                }`}
              >
                {report.valid ? <CheckCircle2 className="w-7 h-7" /> : <XCircle className="w-7 h-7" />}
              </div>
              <div>
                <h3 className="font-bold text-lg">{report.valid ? '契约验证通过' : '契约验证失败'}</h3>
                <p className="text-xs opacity-80 font-mono">
                  Schema 规范：Draft 2020-12 &bull; 验证时间：{new Date(report.validated_at).toLocaleTimeString()}
                </p>
              </div>
            </div>

            <div className="text-right font-mono text-xs">
              <div>通过规则数：{report.summary.passed_rules}</div>
              <div>错误/违规数：{report.summary.failed_rules}</div>
            </div>
          </div>

          {/* Invariant Checks Cards */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
            <h3 className="font-semibold text-slate-200 text-sm flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              <span>Epic E0 工程不变性审计</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {report.summary.invariant_checks.map((inv, idx) => (
                <div
                  key={idx}
                  className={`p-3.5 rounded-lg border text-xs space-y-1 ${
                    inv.status === 'pass'
                      ? 'bg-slate-950 border-emerald-800/50 text-slate-300'
                      : inv.status === 'fail'
                      ? 'bg-rose-950/20 border-rose-800/50 text-rose-300'
                      : 'bg-amber-950/20 border-amber-800/50 text-amber-300'
                  }`}
                >
                  <div className="flex items-center justify-between font-semibold">
                    <span className="text-slate-100">{inv.name}</span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-mono ${
                        inv.status === 'pass' ? 'bg-emerald-950 text-emerald-400' : 'bg-rose-950 text-rose-400'
                      }`}
                    >
                      {inv.status === 'pass' ? '通过' : '未通过'}
                    </span>
                  </div>
                  <p className="text-slate-400 text-[11px] font-mono">{inv.detail}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Detailed Schema Errors (if any) */}
          {report.errors.length > 0 && (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
              <h3 className="font-semibold text-rose-400 text-sm flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4" />
                <span>JSON Schema Draft 2020-12 校验错误 ({report.errors.length})</span>
              </h3>

              <div className="space-y-2 max-h-60 overflow-y-auto font-mono text-xs">
                {report.errors.map((err, idx) => (
                  <div key={idx} className="p-3 bg-slate-950 border border-rose-900/50 rounded-lg space-y-1 text-rose-300">
                    <div className="flex justify-between text-[11px]">
                      <span className="text-rose-400 font-bold">{err.keyword}</span>
                      <span className="text-slate-500">{err.instancePath || 'root'}</span>
                    </div>
                    <p className="text-slate-200">{err.message}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
