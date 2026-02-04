'use client';

import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { ArrowRight, Sparkles, Database, BarChart3, ShieldCheck, Github } from 'lucide-react';

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans selection:bg-indigo-500/30 overflow-hidden">

      {/* Navbar */}
      <nav className="border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-indigo-600 p-2 rounded-lg text-white">
              <Sparkles size={20} />
            </div>
            <span className="font-bold text-lg tracking-tight">AutoClean</span>
          </div>
          <div className="flex items-center gap-4">
            <a
              href="https://github.com/AthunSujith/Auto_Tabular_File_Cleaner"
              target="_blank"
              rel="noreferrer"
              className="text-slate-500 hover:text-slate-900 dark:hover:text-white transition-colors"
            >
              <Github size={20} />
            </a>
            <Link href="/dashboard">
              <button className="bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900 px-4 py-2 rounded-lg text-sm font-semibold hover:bg-slate-800 dark:hover:bg-slate-200 transition-colors">
                Launch App
              </button>
            </Link>
          </div>
        </div>
      </nav>

      <main className="relative">
        {/* Hero Section */}
        <section className="pt-24 pb-20 px-6 max-w-6xl mx-auto text-center relative z-10">

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 text-sm font-medium mb-8 border border-indigo-100 dark:border-indigo-800"
          >
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
            </span>
            v1.0 Now Available
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-5xl md:text-7xl font-extrabold tracking-tight text-slate-900 dark:text-white mb-6 leading-tight"
          >
            Data Cleaning, <br />
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500">
              Reimagined.
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-lg md:text-xl text-slate-600 dark:text-slate-400 max-w-2xl mx-auto mb-10 leading-relaxed"
          >
            Stop spending hours fixing broken CSVs. AutoClean uses intelligent algorithms to handle missing values, outliers, and scaling automatically.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link href="/dashboard" className="w-full sm:w-auto">
              <button className="w-full sm:w-auto px-8 py-4 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-lg shadow-lg shadow-indigo-500/30 hover:shadow-indigo-500/40 hover:-translate-y-1 transition-all flex items-center justify-center gap-2">
                Get Started for Free
                <ArrowRight size={20} />
              </button>
            </Link>
            <a href="https://github.com/AthunSujith/Auto_Tabular_File_Cleaner" target='_blank' rel='noreferrer' className="w-full sm:w-auto">
              <button className="w-full sm:w-auto px-8 py-4 rounded-xl bg-white dark:bg-slate-800 text-slate-900 dark:text-white border border-slate-200 dark:border-slate-700 font-semibold text-lg hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors flex items-center justify-center gap-2">
                <Github size={20} />
                View on GitHub
              </button>
            </a>
          </motion.div>
        </section>

        {/* Features Grid */}
        <section className="py-24 bg-slate-100/50 dark:bg-slate-900/50 border-t border-slate-200 dark:border-slate-800">
          <div className="max-w-6xl mx-auto px-6">
            <div className="text-center mb-16">
              <h2 className="text-3xl md:text-4xl font-bold text-slate-900 dark:text-white mb-4">
                Everything you need to prep data.
              </h2>
              <p className="text-lg text-slate-600 dark:text-slate-400">
                Powerful features wrapped in a simple, intuitive interface.
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              <FeatureCard
                icon={<Database className="text-blue-500" size={32} />}
                title="Smart Imputation"
                description="Automatically fill missing values using Mean, Median, or Most Frequent strategies tailored to your data type."
              />
              <FeatureCard
                icon={<ShieldCheck className="text-emerald-500" size={32} />}
                title="Outlier Handling"
                description="Detect and manage anomalies with robust IQR-based winsorization or clipping techniques."
              />
              <FeatureCard
                icon={<BarChart3 className="text-purple-500" size={32} />}
                title="ML Ready"
                description="One-hot encoding, Ordinal encoding, and robust scaling ensure your data is ready for any machine learning model."
              />
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="py-12 border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950">
          <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-2">
              <div className="bg-slate-900 dark:bg-white p-1.5 rounded-md text-white dark:text-slate-900">
                <Sparkles size={16} />
              </div>
              <span className="font-bold text-lg text-slate-900 dark:text-white">AutoClean</span>
            </div>
            <p className="text-slate-500 dark:text-slate-400 text-sm">
              © {new Date().getFullYear()} Athun Sujith. Open Source.
            </p>
          </div>
        </footer>
      </main>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) {
  return (
    <motion.div
      whileHover={{ y: -5 }}
      className="bg-white dark:bg-slate-900 p-8 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm hover:shadow-xl transition-all"
    >
      <div className="p-3 bg-slate-50 dark:bg-slate-800 w-fit rounded-xl mb-6">
        {icon}
      </div>
      <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-3">{title}</h3>
      <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
        {description}
      </p>
    </motion.div>
  )
}
