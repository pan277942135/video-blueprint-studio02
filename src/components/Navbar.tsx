import React from 'react';
import { Video, ShieldCheck, FileJson, Package, Edit3, TestTube2, Cpu } from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  contractStatus: 'valid' | 'invalid' | 'checking';
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, contractStatus }) => {
  const tabs = [
    { id: 'pipeline', label: '流水线与任务', icon: Video },
    { id: 'blueprint', label: '蓝图检查器', icon: FileJson },
    { id: 'validator', label: 'Schema 与不变性', icon: ShieldCheck },
    { id: 'bundle', label: '打包包管理器', icon: Package },
    { id: 'annotations', label: '手动标注', icon: Edit3 },
    { id: 'tests', label: '契约测试', icon: TestTube2 },
  ];

  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand & Epic Tag */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center text-black font-bold shadow-lg shadow-cyan-500/20">
              <Cpu className="w-6 h-6 text-slate-950" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg tracking-tight text-slate-100">Video Blueprint Studio</span>
                <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800/50">
                  Epic E0
                </span>
              </div>
              <p className="text-xs text-slate-400">离线可验证视频蓝图系统</p>
            </div>
          </div>

          {/* Right Status Badges */}
          <div className="flex items-center space-x-3 text-xs">
            <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 text-slate-300 border border-slate-700">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span className="font-mono">Draft 2020-12 Schema 规范</span>
            </div>

            <div
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border font-mono ${
                contractStatus === 'valid'
                  ? 'bg-emerald-950/60 text-emerald-400 border-emerald-800/60'
                  : contractStatus === 'invalid'
                  ? 'bg-rose-950/60 text-rose-400 border-rose-800/60'
                  : 'bg-slate-800 text-slate-400 border-slate-700'
              }`}
            >
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>{contractStatus === 'valid' ? '契约有效' : contractStatus === 'invalid' ? '契约异常' : '正在检查契约...'}</span>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-1 overflow-x-auto border-t border-slate-800/60 py-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap ${
                  isActive
                    ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
