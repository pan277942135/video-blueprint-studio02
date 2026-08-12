import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { PipelineDashboard } from './components/PipelineDashboard';
import { BlueprintInspector } from './components/BlueprintInspector';
import { SchemaValidatorView } from './components/SchemaValidatorView';
import { BundleManagerView } from './components/BundleManagerView';
import { AnnotationsEditorView } from './components/AnnotationsEditorView';
import { ContractTestSuiteView } from './components/ContractTestSuiteView';
import { VideoRecord, AnalysisJob, BlueprintManifest, ValidationReport, AnnotationOp } from './types';

export default function App() {
  const [activeTab, setActiveTab] = useState<string>('pipeline');
  const [videos, setVideos] = useState<VideoRecord[]>([]);
  const [jobs, setJobs] = useState<AnalysisJob[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [activeBlueprint, setActiveBlueprint] = useState<BlueprintManifest | null>(null);
  const [validationReport, setValidationReport] = useState<ValidationReport | null>(null);
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [contractStatus, setContractStatus] = useState<'valid' | 'invalid' | 'checking'>('valid');

  // Load initial videos & jobs list
  const fetchInitialData = async () => {
    try {
      const [vRes, jRes] = await Promise.all([fetch('/api/v1/videos'), fetch('/api/v1/analyses')]);
      const vType = vRes.headers.get('content-type');
      const jType = jRes.headers.get('content-type');
      
      if (vRes.ok && vType && vType.includes('application/json')) {
        setVideos(await vRes.json());
      }
      if (jRes.ok && jType && jType.includes('application/json')) {
        const jobsList: AnalysisJob[] = await jRes.json();
        setJobs(jobsList);
        if (jobsList.length > 0 && !selectedJobId) {
          setSelectedJobId(jobsList[0].analysis_id);
        }
      }
    } catch (err) {
      console.error('Failed to fetch initial data:', err);
    }
  };

  useEffect(() => {
    fetchInitialData();
  }, []);

  // Poll job status & fetch active blueprint when selectedJobId changes
  useEffect(() => {
    if (!selectedJobId) return;

    const fetchBlueprint = async () => {
      try {
        const bpRes = await fetch(`/api/v1/analyses/${selectedJobId}/blueprint`);
        const bpType = bpRes.headers.get('content-type');
        if (bpRes.ok && bpType && bpType.includes('application/json')) {
          const bpData = await bpRes.json();
          setActiveBlueprint(bpData);

          // Validate blueprint
          const valRes = await fetch(`/api/v1/analyses/${selectedJobId}/validate`, { method: 'POST' });
          const valType = valRes.headers.get('content-type');
          if (valRes.ok && valType && valType.includes('application/json')) {
            const valData = await valRes.json();
            setValidationReport(valData);
            setContractStatus(valData.valid ? 'valid' : 'invalid');
          }
        }
      } catch (err) {
        console.error('Failed to fetch blueprint:', err);
      }
    };

    fetchBlueprint();

    // Setup SSE or polling for live progress updates
    const interval = setInterval(async () => {
      try {
        const jRes = await fetch('/api/v1/analyses');
        const jType = jRes.headers.get('content-type');
        if (jRes.ok && jType && jType.includes('application/json')) {
          const updatedJobs: AnalysisJob[] = await jRes.json();
          setJobs(updatedJobs);
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [selectedJobId]);

  // Handler: Upload Video
  const handleUploadVideo = async (
    file: File | null,
    fileName: string,
    authAttested: boolean,
    adultAttested: boolean
  ) => {
    const formData = new FormData();
    if (file) formData.append('file', file);
    formData.append('file_name', fileName);
    formData.append('authorization_attested', String(authAttested));
    formData.append('adult_subject_attested', String(adultAttested));

    const res = await fetch('/api/v1/videos', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || 'Upload failed');
    }

    await fetchInitialData();
  };

  // Handler: Create Analysis Job
  const handleCreateAnalysis = async (videoId: string, selectedModules: string[]) => {
    const res = await fetch('/api/v1/analyses', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_id: videoId,
        modules: selectedModules,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || 'Failed to queue job');
    }

    const createdJob: AnalysisJob = await res.json();
    setSelectedJobId(createdJob.analysis_id);
    await fetchInitialData();
  };

  // Handler: Cancel Job
  const handleCancelJob = async (jobId: string) => {
    await fetch(`/api/v1/analyses/${jobId}/cancel`, { method: 'POST' });
    await fetchInitialData();
  };

  // Handler: Retry Job
  const handleRetryJob = async (jobId: string) => {
    await fetch(`/api/v1/analyses/${jobId}/retry`, { method: 'POST' });
    await fetchInitialData();
  };

  // Handler: View Blueprint
  const handleViewBlueprint = (jobId: string) => {
    setSelectedJobId(jobId);
    setActiveTab('blueprint');
  };

  // Handler: Download Bundle ZIP
  const handleDownloadBundle = (jobId: string) => {
    window.open(`/api/v1/analyses/${jobId}/bundle`, '_blank');
  };

  // Handler: Download Raw Blueprint JSON
  const handleDownloadBlueprintFile = () => {
    if (!activeBlueprint) return;
    const blob = new Blob([JSON.stringify(activeBlueprint, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `blueprint_${activeBlueprint.blueprint_id.slice(0, 8)}.json`;
    a.click();
  };

  // Handler: Re-run Schema Validation
  const handleRunValidation = async () => {
    if (!selectedJobId) return;
    setIsValidating(true);
    try {
      const res = await fetch(`/api/v1/analyses/${selectedJobId}/validate`, { method: 'POST' });
      if (res.ok) {
        const valData = await res.json();
        setValidationReport(valData);
        setContractStatus(valData.valid ? 'valid' : 'invalid');
      }
    } finally {
      setIsValidating(false);
    }
  };

  // Handler: Apply Annotations Patch
  const handleApplyPatch = async (operations: AnnotationOp[]) => {
    if (!selectedJobId) throw new Error('No active job selected.');
    const res = await fetch(`/api/v1/analyses/${selectedJobId}/annotations`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ operations }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || 'Failed to apply patch');
    }

    await fetchInitialData();
  };

  // Handler: Run Contract Test Suite
  const handleRunContractTest = async (): Promise<ValidationReport> => {
    const res = await fetch('/api/v1/test/run-contract-tests', { method: 'POST' });
    const data = await res.json();
    if (data.report) return data.report;
    throw new Error(data.error || 'Test suite failed');
  };

  const activeJob = jobs.find((j) => j.analysis_id === selectedJobId) || jobs[0] || null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} contractStatus={contractStatus} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'pipeline' && (
          <PipelineDashboard
            videos={videos}
            jobs={jobs}
            selectedJobId={selectedJobId}
            setSelectedJobId={setSelectedJobId}
            onUploadVideo={handleUploadVideo}
            onCreateAnalysis={handleCreateAnalysis}
            onCancelJob={handleCancelJob}
            onRetryJob={handleRetryJob}
            onViewBlueprint={handleViewBlueprint}
            onDownloadBundle={handleDownloadBundle}
          />
        )}

        {activeTab === 'blueprint' && (
          <BlueprintInspector
            blueprint={activeBlueprint}
            validationReport={validationReport}
            onValidate={handleRunValidation}
            onDownloadBlueprint={handleDownloadBlueprintFile}
          />
        )}

        {activeTab === 'validator' && (
          <SchemaValidatorView
            report={validationReport}
            isValidating={isValidating}
            onRunValidation={handleRunValidation}
            onRunExampleContractTest={async () => {
              const rep = await handleRunContractTest();
              setValidationReport(rep);
            }}
          />
        )}

        {activeTab === 'bundle' && (
          <BundleManagerView
            blueprint={activeBlueprint}
            onDownloadBundle={() => activeJob && handleDownloadBundle(activeJob.analysis_id)}
          />
        )}

        {activeTab === 'annotations' && (
          <AnnotationsEditorView activeJob={activeJob} onApplyPatch={handleApplyPatch} />
        )}

        {activeTab === 'tests' && <ContractTestSuiteView onRunContractTest={handleRunContractTest} />}
      </main>

      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <p>Video Blueprint Studio M1 &bull; Epic E0 Offline Verifiable Blueprint Engine</p>
      </footer>
    </div>
  );
}
