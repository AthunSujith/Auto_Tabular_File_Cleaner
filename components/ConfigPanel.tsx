'use client';

import React from 'react';
import { Settings2, HelpCircle } from 'lucide-react';
import { motion } from 'framer-motion';

export interface CleanConfigByType {
    numeric_impute: string;
    numeric_impute_fill_value: number | null;
    categorical_impute: string;
    categorical_impute_fill_value: string;
    outlier_method: string;
    iqr_multiplier: number;
    outlier_strategy: string;
    scaler: string;
    encoder: string;
    drop_onehot: string;
    variance_threshold: number;
    corr_threshold: number | null;
    target_column: string | null;
}

interface ConfigPanelProps {
    config: CleanConfigByType;
    setConfig: React.Dispatch<React.SetStateAction<CleanConfigByType>>;
}

export function ConfigPanel({ config, setConfig }: ConfigPanelProps) {
    const handleChange = (key: keyof CleanConfigByType, value: any) => {
        setConfig((prev) => ({ ...prev, [key]: value }));
    };

    return (
        <div className="w-full bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 overflow-hidden shadow-sm">
            <div className="p-4 border-b border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50 flex items-center gap-2">
                <Settings2 className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                <h3 className="font-semibold text-slate-800 dark:text-slate-200">
                    Cleaning Configuration
                </h3>
            </div>

            <div className="p-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Numeric Impute */}
                <div className="space-y-3">
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300 block">
                        Numeric Imputation
                    </label>
                    <select
                        value={config.numeric_impute}
                        onChange={(e) => handleChange('numeric_impute', e.target.value)}
                        className="w-full px-3 py-2 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                    >
                        <option value="median">Median (Default)</option>
                        <option value="mean">Mean</option>
                        <option value="most_frequent">Most Frequent</option>
                        <option value="constant">Constant</option>
                    </select>
                    {config.numeric_impute === 'constant' && (
                        <input
                            type="number"
                            placeholder="Fill Value"
                            value={config.numeric_impute_fill_value ?? ''}
                            onChange={(e) => handleChange('numeric_impute_fill_value', parseFloat(e.target.value))}
                            className="w-full px-3 py-2 mt-2 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                        />
                    )}
                </div>

                {/* Categorical Impute */}
                <div className="space-y-3">
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300 block">
                        Categorical Imputation
                    </label>
                    <select
                        value={config.categorical_impute}
                        onChange={(e) => handleChange('categorical_impute', e.target.value)}
                        className="w-full px-3 py-2 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                    >
                        <option value="most_frequent">Most Frequent (Default)</option>
                        <option value="constant">Constant (e.g. "Missing")</option>
                    </select>
                    {config.categorical_impute === 'constant' && (
                        <input
                            type="text"
                            placeholder="Fill Value"
                            value={config.categorical_impute_fill_value}
                            onChange={(e) => handleChange('categorical_impute_fill_value', e.target.value)}
                            className="w-full px-3 py-2 mt-2 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                        />
                    )}
                </div>

                {/* Outliers */}
                <div className="space-y-3">
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300 block">
                        Outlier Handling
                    </label>
                    <div className="flex gap-2">
                        <select
                            value={config.outlier_method}
                            onChange={(e) => handleChange('outlier_method', e.target.value)}
                            className="w-1/2 px-3 py-2 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                        >
                            <option value="iqr">IQR</option>
                            <option value="none">None</option>
                        </select>
                        <select
                            disabled={config.outlier_method === 'none'}
                            value={config.outlier_strategy}
                            onChange={(e) => handleChange('outlier_strategy', e.target.value)}
                            className="w-1/2 px-3 py-2 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-sm focus:ring-2 focus:ring-indigo-500 outline-none disabled:opacity-50"
                        >
                            <option value="winsorize">Winsorize (Cap)</option>
                            <option value="clip">Clip</option>
                        </select>
                    </div>
                </div>

                {/* Scaler */}
                <div className="space-y-3">
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300 block">
                        Scaling Method
                    </label>
                    <select
                        value={config.scaler}
                        onChange={(e) => handleChange('scaler', e.target.value)}
                        className="w-full px-3 py-2 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                    >
                        <option value="robust">Robust (Good for outliers)</option>
                        <option value="standard">Standard (Z-Score)</option>
                        <option value="minmax">MinMax (0-1)</option>
                        <option value="none">None</option>
                    </select>
                </div>

                {/* Encoder */}
                <div className="space-y-3">
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300 block">
                        Categorical Encoding
                    </label>
                    <select
                        value={config.encoder}
                        onChange={(e) => handleChange('encoder', e.target.value)}
                        className="w-full px-3 py-2 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                    >
                        <option value="onehot">One-Hot Encoding</option>
                        <option value="ordinal">Ordinal Encoding</option>
                    </select>
                </div>

                {/* Target Column */}
                <div className="space-y-3">
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300 block flex items-center gap-2">
                        Target Column (Optional)
                        <span title="Column to exclude from processing" className="cursor-help text-slate-400"><HelpCircle size={14} /></span>
                    </label>
                    <input
                        type="text"
                        placeholder="e.g. SalePrice"
                        value={config.target_column || ''}
                        onChange={(e) => handleChange('target_column', e.target.value || null)}
                        className="w-full px-3 py-2 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                    />
                </div>

            </div>
        </div>
    );
}
