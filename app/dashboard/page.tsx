'use client';

import React, { useState } from 'react';
import axios from 'axios';
import { FileUpload } from '@/components/FileUpload';
import { ConfigPanel, CleanConfigByType } from '@/components/ConfigPanel';
import { Download, Sparkles, AlertCircle, Loader2, Github } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [config, setConfig] = useState<CleanConfigByType>({
    numeric_impute: 'median',
    numeric_impute_fill_value: null,
    categorical_impute: 'most_frequent',
    categorical_impute_fill_value: 'missing',
    outlier_method: 'iqr',
    iqr_multiplier: 1.5,
    outlier_strategy: 'winsorize',
    scaler: 'robust',
    encoder: 'onehot',
    drop_onehot: 'if_binary',
    variance_threshold: 0.0,
    corr_threshold: 0.95,
    target_column: null,
  });

  const handleProcess = async () => {
    if (!file) {
      setError("Please select a file first.");
      return;
    }
    setLoading(true);
    setError(null);
    setDownloadUrl(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('config', JSON.stringify(config));

    try {
      const response = await axios.post('/api/process', formData, {
        responseType: 'blob',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      setDownloadUrl(url);
    } catch (err: any) {
      console.error(err);
      if (err.response && err.response.data.text) {
        // Try to read blob as text to get error message
        try {
          const text = await err.response.data.text();
          const json = JSON.parse(text);
          setError(json.detail || "An error occurred during processing.");
        } catch {
          setError("An error occurred during processing.");
        }
      } else {
        setError(err.message || "Failed to process file.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans selection:bg-indigo-500/30">

      {/* Navbar */}
      <nav className="border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-indigo-600 p-2 rounded-lg text-white">
              <Sparkles size={20} />
            </div>
            <span className="font-bold text-lg tracking-tight">AutoClean</span>
          </div>
          <a
            href="https://github.com/AthunSujith/Auto_Tabular_File_Cleaner"
            target="_blank"
            rel="noreferrer"
            className="text-slate-500 hover:text-slate-900 dark:hover:text-white transition-colors"
          >
            <Github size={20} />
          </a>
        </div>
      </nav>

      <main className="max-w-4xl mx-auto px-6 py-12 space-y-12">

        {/* Header */}
        <div className="text-center space-y-4 max-w-2xl mx-auto">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl md:text-5xl font-extrabold tracking-tight text-slate-900 dark:text-white"
          >
            Clean your data <span className="text-indigo-600 dark:text-indigo-400">instantly</span>.
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-lg text-slate-600 dark:text-slate-400"
          >
            Automated preprocessing for your tabular data. Handle missing values, outliers, encoding, and scaling in seconds.
          </motion.p>
        </div>

        {/* Main Interface */}
        <div className="space-y-8">
          <FileUpload selectedFile={file} onFileSelect={setFile} />

          <AnimatePresence>
            {file && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
              >
                <ConfigPanel config={config} setConfig={setConfig} />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Actions */}
          <div className="flex flex-col items-center gap-6">
            {error && (
              <div className="flex items-center gap-2 text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 px-4 py-2 rounded-lg border border-red-200 dark:border-red-800">
                <AlertCircle size={18} />
                <span className="text-sm font-medium">{error}</span>
              </div>
            )}

            <button
              onClick={handleProcess}
              disabled={!file || loading}
              className={clsx(
                "relative group overflow-hidden px-8 py-4 rounded-xl flex items-center gap-3 font-semibold text-lg transition-all duration-300",
                !file || loading
                  ? "bg-slate-200 dark:bg-slate-800 text-slate-400 cursor-not-allowed"
                  : "bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg shadow-indigo-500/30 hover:shadow-indigo-500/40 hover:-translate-y-0.5"
              )}
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  Start Cleaning
                  <Sparkles size={18} className={file ? "animate-pulse" : ""} />
                </>
              )}
            </button>

            <AnimatePresence>
              {downloadUrl && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="w-full max-w-md bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 rounded-xl p-6 flex flex-col items-center gap-4 text-center"
                >
                  <div className="p-3 bg-emerald-100 dark:bg-emerald-900/50 rounded-full text-emerald-600 dark:text-emerald-400">
                    <CheckCircle size={32} />
                  </div>
                  <div>
                    <h3 className="font-bold text-emerald-800 dark:text-emerald-200">Processing Complete!</h3>
                    <p className="text-sm text-emerald-700 dark:text-emerald-300">Your cleaned data is ready for download.</p>
                  </div>
                  <a
                    href={downloadUrl}
                    download="cleaned_data.csv"
                    className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-700 text-white px-6 py-2 rounded-lg font-medium transition-colors"
                  >
                    <Download size={18} />
                    Download CSV
                  </a>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center text-slate-400 dark:text-slate-600 pt-12 pb-6 text-sm">
          <p>© {new Date().getFullYear()} AutoCleaner. Built with Next.js & Python.</p>
        </footer>

      </main>
    </div>
  );
}

// Helper icon
import { CheckCircle } from 'lucide-react';
