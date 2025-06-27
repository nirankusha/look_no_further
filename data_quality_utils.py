"""
Data Quality Utilities for mBART Entropy Analysis
Run this script to diagnose and fix data quality issues
"""

import numpy as np
import json
from typing import Dict, Any, List, Tuple

def diagnose_data_issues(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive diagnosis of data quality issues
    """
    report = {
        'total_sentences': 0,
        'total_tokens': 0,
        'issues': {
            'infinite_values': 0,
            'nan_values': 0,
            'negative_probabilities': 0,
            'missing_metrics': 0,
            'empty_tokens': 0
        },
        'detailed_issues': [],
        'metrics_coverage': {},
        'recommendations': []
    }
    
    metrics_to_check = [
        'conditional_entropy_bits', 'surprisal_bits', 
        'token_probability', 'topk_entropy_bits',
        'full_entropy_bits', 'trgt_token_entropy_bits',
        'trgt_token_surprisal_bits'
    ]
    
    # Initialize metrics coverage tracking
    for metric in metrics_to_check:
        report['metrics_coverage'][metric] = {'present': 0, 'total': 0}
    
    for sentence_key, sentence_data in data.items():
        if 'source_info' not in sentence_data:
            continue
            
        source_info = sentence_data['source_info']
        tokens = source_info.get('tokens', [])
        
        report['total_sentences'] += 1
        
        if not tokens:
            report['issues']['empty_tokens'] += 1
            report['detailed_issues'].append({
                'sentence': sentence_key[:50] + "...",
                'issue': 'No tokens found',
                'severity': 'high'
            })
            continue
        
        report['total_tokens'] += len(tokens)
        
        # Check each metric
        for metric in metrics_to_check:
            report['metrics_coverage'][metric]['total'] += 1
            
            if metric in source_info:
                report['metrics_coverage'][metric]['present'] += 1
                metric_data = source_info[metric]
                
                if isinstance(metric_data, dict):
                    for token, value in metric_data.items():
                        if np.isinf(value):
                            report['issues']['infinite_values'] += 1
                            report['detailed_issues'].append({
                                'sentence': sentence_key[:50] + "...",
                                'issue': f'Infinite {metric} for token "{token}"',
                                'value': str(value),
                                'severity': 'medium'
                            })
                        elif np.isnan(value):
                            report['issues']['nan_values'] += 1
                            report['detailed_issues'].append({
                                'sentence': sentence_key[:50] + "...",
                                'issue': f'NaN {metric} for token "{token}"',
                                'value': str(value),
                                'severity': 'medium'
                            })
                        elif metric == 'token_probability' and value < 0:
                            report['issues']['negative_probabilities'] += 1
                            report['detailed_issues'].append({
                                'sentence': sentence_key[:50] + "...",
                                'issue': f'Negative probability for token "{token}"',
                                'value': str(value),
                                'severity': 'high'
                            })
            else:
                report['issues']['missing_metrics'] += 1
    
    # Generate recommendations
    total_issues = sum(report['issues'].values())
    
    if total_issues == 0:
        report['recommendations'].append("✅ Data quality is excellent! No issues detected.")
    else:
        if report['issues']['infinite_values'] > 0:
            report['recommendations'].append(
                f"🔧 Replace {report['issues']['infinite_values']} infinite values with finite alternatives"
            )
        
        if report['issues']['nan_values'] > 0:
            report['recommendations'].append(
                f"🔧 Replace {report['issues']['nan_values']} NaN values with 0 or interpolated values"
            )
        
        if report['issues']['negative_probabilities'] > 0:
            report['recommendations'].append(
                f"🔧 Fix {report['issues']['negative_probabilities']} negative probability values"
            )
        
        if report['issues']['missing_metrics'] > 0:
            report['recommendations'].append(
                f"📊 Consider computing missing metrics for better analysis coverage"
            )
    
    return report

def clean_dataset(data: Dict[str, Any], strategy: str = "replace_with_zero") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Clean dataset by handling problematic values
    
    Strategies:
    - "replace_with_zero": Replace inf/nan with 0
    - "interpolate": Try to interpolate missing values
    - "remove": Remove problematic entries
    """
    
    cleaned_data = {}
    cleaning_log = {'replaced': 0, 'removed': 0, 'interpolated': 0}
    
    for sentence_key, sentence_data in data.items():
        if 'source_info' not in sentence_data:
            cleaned_data[sentence_key] = sentence_data
            continue
        
        source_info = sentence_data['source_info'].copy()
        tokens = source_info.get('tokens', [])
        
        if not tokens:
            if strategy == "remove":
                cleaning_log['removed'] += 1
                continue
            else:
                cleaned_data[sentence_key] = sentence_data
                continue
        
        # Clean each metric
        metrics_to_clean = [
            'conditional_entropy_bits', 'surprisal_bits', 
            'token_probability', 'topk_entropy_bits',
            'full_entropy_bits', 'trgt_token_entropy_bits',
            'trgt_token_surprisal_bits'
        ]
        
        for metric in metrics_to_clean:
            if metric in source_info and isinstance(source_info[metric], dict):
                cleaned_metric = {}
                
                for token, value in source_info[metric].items():
                    if np.isfinite(value) and value >= 0:
                        cleaned_metric[token] = value
                    else:
                        if strategy == "replace_with_zero":
                            cleaned_metric[token] = 0.0
                            cleaning_log['replaced'] += 1
                        elif strategy == "interpolate":
                            # Simple interpolation: use average of valid values
                            valid_values = [v for v in source_info[metric].values() if np.isfinite(v) and v >= 0]
                            if valid_values:
                                cleaned_metric[token] = np.mean(valid_values)
                                cleaning_log['interpolated'] += 1
                            else:
                                cleaned_metric[token] = 0.0
                                cleaning_log['replaced'] += 1
                        # For "remove" strategy, we skip the problematic token
                
                source_info[metric] = cleaned_metric
        
        cleaned_data[sentence_key] = {'source_info': source_info}
    
    return cleaned_data, cleaning_log

def generate_quality_report(data: Dict[str, Any]) -> str:
    """Generate a comprehensive quality report"""
    
    diagnosis = diagnose_data_issues(data)
    
    report = []
    report.append("📊 DATA QUALITY REPORT")
    report.append("=" * 50)
    
    # Overview
    report.append(f"\n📋 OVERVIEW:")
    report.append(f"   Total sentences: {diagnosis['total_sentences']}")
    report.append(f"   Total tokens: {diagnosis['total_tokens']}")
    
    # Issues summary
    total_issues = sum(diagnosis['issues'].values())
    report.append(f"\n⚠️ ISSUES DETECTED: {total_issues}")
    
    for issue_type, count in diagnosis['issues'].items():
        if count > 0:
            report.append(f"   - {issue_type.replace('_', ' ').title()}: {count}")
    
    # Metrics coverage
    report.append(f"\n📊 METRICS COVERAGE:")
    for metric, coverage in diagnosis['metrics_coverage'].items():
        if coverage['total'] > 0:
            percentage = (coverage['present'] / coverage['total']) * 100
            report.append(f"   - {metric}: {coverage['present']}/{coverage['total']} ({percentage:.1f}%)")
    
    # Detailed issues (limit to first 10)
    if diagnosis['detailed_issues']:
        report.append(f"\n🔍 DETAILED ISSUES (showing first 10):")
        for issue in diagnosis['detailed_issues'][:10]:
            severity_emoji = "🔴" if issue['severity'] == 'high' else "🟡"
            report.append(f"   {severity_emoji} {issue['sentence']}: {issue['issue']}")
        
        if len(diagnosis['detailed_issues']) > 10:
            report.append(f"   ... and {len(diagnosis['detailed_issues']) - 10} more issues")
    
    # Recommendations
    if diagnosis['recommendations']:
        report.append(f"\n💡 RECOMMENDATIONS:")
        for rec in diagnosis['recommendations']:
            report.append(f"   {rec}")
    
    # Quality score
    if total_issues == 0:
        quality_score = "Excellent"
        score_emoji = "🟢"
    elif total_issues < diagnosis['total_tokens'] * 0.01:  # Less than 1% issues
        quality_score = "Good"
        score_emoji = "🟡"
    elif total_issues < diagnosis['total_tokens'] * 0.05:  # Less than 5% issues
        quality_score = "Fair"
        score_emoji = "🟠"
    else:
        quality_score = "Poor"
        score_emoji = "🔴"
    
    report.append(f"\n{score_emoji} OVERALL QUALITY: {quality_score}")
    
    return "\n".join(report)

def quick_fix_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Quick fix for common data issues"""
    
    print("🔧 Applying quick fixes to data...")
    
    cleaned_data, log = clean_dataset(data, strategy="replace_with_zero")
    
    print(f"✅ Quick fix completed:")
    print(f"   - Replaced values: {log['replaced']}")
    print(f"   - Removed entries: {log['removed']}")
    print(f"   - Interpolated values: {log['interpolated']}")
    
    return cleaned_data

# Convenience functions for use in the notebook
def analyze_current_data():
    """Analyze currently loaded data quality"""
    from widget_interface import loaded_datasets, processed_datasets
    
    print("🔍 ANALYZING CURRENT DATA QUALITY")
    print("=" * 50)
    
    if not loaded_datasets and not processed_datasets:
        print("❌ No data loaded. Please load some data first.")
        return
    
    # Analyze loaded datasets
    if loaded_datasets:
        print("\n📁 LOADED DATASETS:")
        for name, data in loaded_datasets.items():
            print(f"\n📊 Dataset: {name}")
            report = generate_quality_report(data)
            print(report)
    
    # Analyze processed datasets
    if processed_datasets:
        print("\n⚙️ PROCESSED DATASETS:")
        for name, data in processed_datasets.items():
            print(f"\n📊 Dataset: {name}")
            # Convert processed format back to analyze
            raw_data = {}
            for key, value in data.items():
                sentence_text = key.split(': ')[-1] if ': ' in key else key
                raw_data[sentence_text] = value
            
            report = generate_quality_report(raw_data)
            print(report)

def fix_current_data():
    """Fix quality issues in currently loaded data"""
    from widget_interface import loaded_datasets, processed_datasets
    
    if not loaded_datasets:
        print("❌ No loaded datasets to fix. Please load some data first.")
        return
    
    print("🔧 FIXING DATA QUALITY ISSUES")
    print("=" * 40)
    
    fixed_count = 0
    for name, data in loaded_datasets.items():
        print(f"\n🔧 Fixing dataset: {name}")
        
        # Check if fixes are needed
        diagnosis = diagnose_data_issues(data)
        total_issues = sum(diagnosis['issues'].values())
        
        if total_issues == 0:
            print("✅ No fixes needed - data quality is good!")
            continue
        
        # Apply fixes
        fixed_data = quick_fix_data(data)
        loaded_datasets[name] = fixed_data
        fixed_count += 1
    
    if fixed_count > 0:
        print(f"\n✅ Fixed {fixed_count} datasets!")
        print("💡 You may want to re-process the datasets for visualization.")
        
        # Update interfaces
        try:
            from widget_interface import update_all_interfaces
            update_all_interfaces()
        except:
            print("⚠️ Could not update interfaces automatically")
    else:
        print("\n✅ All datasets are already in good condition!")

if __name__ == "__main__":
    print("📊 Data Quality Utilities loaded!")
    print("\nAvailable functions:")
    print("- analyze_current_data(): Analyze quality of loaded data")
    print("- fix_current_data(): Fix quality issues in loaded data")
    print("- diagnose_data_issues(data): Detailed diagnosis")
    print("- clean_dataset(data): Clean a dataset")
    print("- generate_quality_report(data): Generate report")
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 14:49:24 2025

@author: niran
"""

