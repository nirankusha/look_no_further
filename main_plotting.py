"""
main_plotting.py - Main plotting interface functions (CORRECTED VERSION)
"""

from plotting_functions import plot_single_sentence, plot_multiple_sentences
from data_processing import filter_data_by_type

def plot_conditional_entropy_metrics(data, sentence_key=None, annotation_type=None,
                                    show_annotations=True, highlight_tags=None, 
                                    title="mBART Entropy Analysis", save_path=None, 
                                    highlight_lines=None, show_comet=True, 
                                    entropy_type='softmax'):
    """
    Main plotting function that handles different data formats and routing
    """
    
    try:
        if sentence_key is not None:
            # Single sentence specified
            if isinstance(data, dict) and sentence_key in data:
                source_info = data[sentence_key]['source_info']
                plot_single_sentence(
                    source_info, 
                    sent_title=f" — {sentence_key[:50]}{'...' if len(sentence_key) > 50 else ''}", 
                    show_annotations=show_annotations,
                    annotation_type=annotation_type, 
                    highlight_tags=highlight_tags, 
                    highlight_lines=highlight_lines,
                    title=title, 
                    save_path=save_path, 
                    show_comet=show_comet, 
                    entropy_type=entropy_type
                )
            else:
                raise ValueError(f"Sentence key '{sentence_key}' not found in data")

        elif isinstance(data, dict) and len(data) == 1:
            # Single sentence in dictionary format
            sentence_key = list(data.keys())[0]
            source_info = data[sentence_key]['source_info']
            plot_single_sentence(
                source_info, 
                sent_title=f" — {sentence_key[:50]}{'...' if len(sentence_key) > 50 else ''}", 
                show_annotations=show_annotations,
                annotation_type=annotation_type, 
                highlight_tags=highlight_tags, 
                highlight_lines=highlight_lines,
                title=title, 
                save_path=save_path, 
                show_comet=show_comet, 
                entropy_type=entropy_type
            )

        elif isinstance(data, dict) and 'tokens' in data:
            # Direct source_info format
            plot_single_sentence(
                data, 
                show_annotations=show_annotations,
                annotation_type=annotation_type, 
                highlight_tags=highlight_tags, 
                highlight_lines=highlight_lines,
                title=title, 
                save_path=save_path, 
                show_comet=show_comet, 
                entropy_type=entropy_type
            )

        elif isinstance(data, dict) and len(data) > 1:
            # Multiple sentences - create separate plots for source and target
            print(f"📊 Processing {len(data)} sentences for visualization...")
            
            source_data = filter_data_by_type(data, 'source')
            target_data = filter_data_by_type(data, 'target')

            print(f"   - Source sentences: {len(source_data)}")
            print(f"   - Target sentences: {len(target_data)}")

            if source_data:
                plot_multiple_sentences(
                    source_data, 
                    "Source", 
                    highlight_tags=highlight_tags,
                    annotation_type=annotation_type, 
                    highlight_lines=highlight_lines, 
                    save_path=save_path if not save_path or not save_path.endswith('.png') else save_path.replace('.png', '_source.png'), 
                    title=title
                )

            if target_data:
                target_save_path = None
                if save_path:
                    if save_path.endswith('.png'):
                        target_save_path = save_path.replace('.png', '_target.png')
                    else:
                        target_save_path = f"{save_path}_target.png"
                
                plot_multiple_sentences(
                    target_data, 
                    "Target", 
                    highlight_tags=highlight_tags,
                    annotation_type=annotation_type, 
                    highlight_lines=highlight_lines, 
                    save_path=target_save_path, 
                    title=title
                )

            if not source_data and not target_data:
                print("❌ No valid source or target data found for plotting")

        else:
            raise ValueError("Invalid data format. Expected dictionary with sentence data or direct source_info.")
    
    except Exception as e:
        print(f"❌ Error in plot_conditional_entropy_metrics: {e}")
        import traceback
        traceback.print_exc()

def create_dataset_overlay_plot(processed_datasets, dataset_keys, analysis_type='source', 
                               entropy_type='softmax', highlight_tags=None, 
                               annotation_type=None, save_path=None):
    """Create overlay plot for multiple datasets"""
    
    try:
        all_data = {}
        
        print(f"📈 Creating {analysis_type} overlay plot for {len(dataset_keys)} datasets...")
        
        # Combine data from multiple datasets
        for dataset_key in dataset_keys:
            if dataset_key in processed_datasets:
                data = processed_datasets[dataset_key]
                filtered_data = filter_data_by_type(data, analysis_type)
                
                print(f"   - {dataset_key}: {len(filtered_data)} {analysis_type} sentences")
                
                # Add dataset prefix to keys to avoid conflicts
                for key, value in filtered_data.items():
                    new_key = f"[{dataset_key}] {key}"
                    all_data[new_key] = value
            else:
                print(f"   - ⚠️ {dataset_key} not found in processed datasets")
        
        if not all_data:
            print(f"❌ No {analysis_type} data found for selected datasets")
            return
        
        print(f"📊 Total {analysis_type} sentences for overlay: {len(all_data)}")
        
        # Create the overlay plot
        plot_multiple_sentences(
            all_data,
            f"{analysis_type.title()} ({len(dataset_keys)} datasets)",
            highlight_tags=highlight_tags,
            annotation_type=annotation_type,
            save_path=save_path,
            title=f"Multi-Dataset Comparison - {entropy_type.title()}"
        )
    
    except Exception as e:
        print(f"❌ Error creating dataset overlay plot: {e}")
        import traceback
        traceback.print_exc()

def create_comparison_plot(processed_datasets, base_dataset_names, entropy_types, 
                         analysis_type='source', save_path=None):
    """Create comparison boxplot across datasets and entropy types"""
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        print(f"📊 Creating comparison plot for {len(base_dataset_names)} datasets and {len(entropy_types)} entropy types...")
        
        fig, axes = plt.subplots(len(base_dataset_names), len(entropy_types), 
                                figsize=(6*len(entropy_types), 5*len(base_dataset_names)))
        
        # Handle single subplot cases
        if len(base_dataset_names) == 1 and len(entropy_types) == 1:
            axes = np.array([[axes]])
        elif len(base_dataset_names) == 1:
            axes = axes.reshape(1, -1)
        elif len(entropy_types) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, dataset_name in enumerate(base_dataset_names):
            for j, entropy_type in enumerate(entropy_types):
                if len(base_dataset_names) == 1 and len(entropy_types) == 1:
                    ax = axes[0][0]
                elif len(base_dataset_names) == 1:
                    ax = axes[0][j]
                elif len(entropy_types) == 1:
                    ax = axes[i][0]
                else:
                    ax = axes[i][j]
                
                # Find matching datasets
                matching_keys = [k for k in processed_datasets.keys() 
                               if dataset_name.lower() in k.lower() and entropy_type.lower() in k.lower()]
                
                print(f"   - {dataset_name} + {entropy_type}: Found {len(matching_keys)} matching datasets")
                
                all_entropies = []
                all_surprisals = []
                labels = []
                
                for dataset_key in matching_keys:
                    data = processed_datasets[dataset_key]
                    filtered_data = filter_data_by_type(data, analysis_type)
                    
                    if not filtered_data:
                        continue
                    
                    # Collect all entropy and surprisal values
                    dataset_entropies = []
                    dataset_surprisals = []
                    
                    for sent_data in filtered_data.values():
                        source_info = sent_data['source_info']
                        tokens = source_info.get('tokens', [])
                        
                        entropies = [source_info.get('conditional_entropy_bits', {}).get(t, 0) for t in tokens]
                        surprisals = [source_info.get('surprisal_bits', {}).get(t, 0) for t in tokens]
                        
                        dataset_entropies.extend([e for e in entropies if e > 0])
                        dataset_surprisals.extend([s for s in surprisals if s > 0])
                    
                    if dataset_entropies:
                        all_entropies.append(dataset_entropies)
                        all_surprisals.append(dataset_surprisals)
                        
                        # Extract model type from key
                        model_type = "Conditional" if "conditional" in dataset_key.lower() else "Seq2Seq"
                        labels.append(model_type)
                
                # Create box plots
                if all_entropies:
                    positions = range(1, len(all_entropies) + 1)
                    bp1 = ax.boxplot(all_entropies, positions=[p - 0.2 for p in positions], widths=0.3, 
                                   patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
                    bp2 = ax.boxplot(all_surprisals, positions=[p + 0.2 for p in positions], widths=0.3,
                                   patch_artist=True, boxprops=dict(facecolor='lightcoral', alpha=0.7))
                    
                    ax.set_xticks(positions)
                    ax.set_xticklabels(labels)
                    ax.set_title(f"{dataset_name} - {entropy_type.title()} ({analysis_type.title()})", fontweight='bold')
                    ax.set_ylabel("Bits", fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    # Add legend only to first subplot
                    if i == 0 and j == 0:
                        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], 
                                 ['Conditional Entropy', 'Surprisal'], loc='upper right')
                else:
                    ax.set_title(f"{dataset_name} - {entropy_type.title()} (No Data)", fontweight='bold')
                    ax.text(0.5, 0.5, 'No Data Available', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, fontsize=12, color='gray')
        
        plt.suptitle("Entropy Comparison Across Datasets", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Comparison plot saved to: {save_path}")
        
        plt.show()
    
    except Exception as e:
        print(f"❌ Error creating comparison plot: {e}")
        import traceback
        traceback.print_exc()

def plot_entropy_surprisal_simple(results, title):
    """
    Simple plotting function for quick visualization of entropy and surprisal results.
    Compatible with the original format from your entropy analysis functions.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        words = [r["curr_token"] for r in results]
        surprisals = [r["surprisal"] for r in results]
        entropies = [r["entropy"] for r in results]
        x = range(len(words))

        fig, ax1 = plt.subplots(figsize=(max(12, len(words) * 0.8), 6))

        # Plot surprisal
        l1, = ax1.plot(x, surprisals, color="tab:blue", marker='o', linewidth=2, markersize=6, label="Surprisal")
        ax1.set_ylabel("Surprisal (bits)", color="tab:blue", fontweight='bold')
        ax1.tick_params(axis='y', labelcolor="tab:blue")
        ax1.set_xticks([])
        ax1.grid(True, alpha=0.3)

        # Plot entropy on twin axis
        ax2 = ax1.twinx()
        l2, = ax2.plot(x, entropies, color="tab:orange", marker='x', linewidth=2, markersize=8, label="Entropy")
        ax2.set_ylabel("Entropy (bits)", color="tab:orange", fontweight='bold')
        ax2.tick_params(axis='y', labelcolor="tab:orange")

        # Add token labels with better spacing
        for i, word in enumerate(words):
            y_pos = max(surprisals[i], entropies[i]) + max(max(surprisals), max(entropies)) * 0.05
            ax1.text(i, y_pos, word, rotation=45, ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Mark maximums with enhanced styling
        max_surp_idx = max(range(len(surprisals)), key=lambda i: surprisals[i])
        ax1.plot(max_surp_idx, surprisals[max_surp_idx], marker='*', color='red', markersize=16, markeredgecolor='black')
        ax1.annotate(
            f"Max Surprisal\n({words[max_surp_idx]}: {surprisals[max_surp_idx]:.2f})",
            (max_surp_idx, surprisals[max_surp_idx]),
            textcoords="offset points", xytext=(0, 20), ha='center', color='red', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='red')
        )

        max_ent_idx = max(range(len(entropies)), key=lambda i: entropies[i])
        ax2.plot(max_ent_idx, entropies[max_ent_idx], marker='*', color='green', markersize=16, markeredgecolor='black')
        ax2.annotate(
            f"Max Entropy\n({words[max_ent_idx]}: {entropies[max_ent_idx]:.2f})",
            (max_ent_idx, entropies[max_ent_idx]),
            textcoords="offset points", xytext=(0, 20), ha='center', color='green', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='green')
        )

        # Joint score with enhanced visualization
        joint_scores = [s * e for s, e in zip(surprisals, entropies)]
        max_joint_idx = max(range(len(joint_scores)), key=lambda i: joint_scores[i])

        ax1.plot(max_joint_idx, surprisals[max_joint_idx], marker='*', color='purple', markersize=16, markeredgecolor='black')
        ax1.annotate(
            f"Max Joint\n({words[max_joint_idx]}: S={surprisals[max_joint_idx]:.2f}, E={entropies[max_joint_idx]:.2f})",
            (max_joint_idx, surprisals[max_joint_idx]),
            textcoords="offset points", xytext=(0, -30), ha='center', color='purple', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='plum', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='purple')
        )
        ax2.plot(max_joint_idx, entropies[max_joint_idx], marker='*', color='purple', markersize=16, markeredgecolor='black')

        # Enhanced legend
        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", fontsize=12, frameon=True, fancybox=True, shadow=True)

        ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"❌ Error in plot_entropy_surprisal_simple: {e}")
        import traceback
        traceback.print_exc()

def create_entropy_distribution_plot(processed_datasets, analysis_type='source'):
    """Create distribution plots for entropy values across datasets"""
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        
        for idx, (dataset_key, data) in enumerate(processed_datasets.items()):
            if idx >= 4:  # Limit to 4 datasets for visualization
                break
                
            filtered_data = filter_data_by_type(data, analysis_type)
            
            all_entropies = []
            for sent_data in filtered_data.values():
                source_info = sent_data['source_info']
                tokens = source_info.get('tokens', [])
                entropies = [source_info.get('conditional_entropy_bits', {}).get(t, 0) for t in tokens]
                all_entropies.extend([e for e in entropies if e > 0])
            
            if all_entropies:
                axes[idx].hist(all_entropies, bins=30, alpha=0.7, color=colors[idx % len(colors)], edgecolor='black')
                axes[idx].set_title(f"{dataset_key}\n({len(all_entropies)} tokens)", fontweight='bold')
                axes[idx].set_xlabel("Entropy (bits)", fontweight='bold')
                axes[idx].set_ylabel("Frequency", fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
                
                # Add statistics
                mean_ent = np.mean(all_entropies)
                std_ent = np.std(all_entropies)
                axes[idx].axvline(mean_ent, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ent:.2f}')
                axes[idx].legend()
        
        # Hide unused subplots
        for idx in range(len(processed_datasets), 4):
            axes[idx].set_visible(False)
        
        plt.suptitle(f"Entropy Distribution - {analysis_type.title()} Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"❌ Error creating distribution plot: {e}")
        import traceback
        traceback.print_exc()

def create_token_level_analysis_plot(sentence_data, title="Token-Level Analysis"):
    """Create detailed token-level analysis plot"""
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        source_info = sentence_data['source_info']
        tokens = source_info['tokens']
        
        # Extract all available metrics
        metrics = {}
        metric_labels = {}
        
        if 'conditional_entropy_bits' in source_info:
            metrics['entropy'] = [source_info['conditional_entropy_bits'].get(t, 0) for t in tokens]
            metric_labels['entropy'] = 'Conditional Entropy'
        
        if 'surprisal_bits' in source_info:
            metrics['surprisal'] = [source_info['surprisal_bits'].get(t, 0) for t in tokens]
            metric_labels['surprisal'] = 'Surprisal'
        
        if 'token_probability' in source_info:
            metrics['probability'] = [source_info['token_probability'].get(t, 0) for t in tokens]
            metric_labels['probability'] = 'Token Probability'
        
        if 'topk_entropy_bits' in source_info:
            metrics['topk_entropy'] = [source_info['topk_entropy_bits'].get(t, 0) for t in tokens]
            metric_labels['topk_entropy'] = 'Top-k Entropy'
        
        # Create subplots for each metric
        n_metrics = len(metrics)
        if n_metrics == 0:
            print("❌ No metrics found for token-level analysis")
            return
        
        fig, axes = plt.subplots(n_metrics, 1, figsize=(max(12, len(tokens)), 3 * n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        x = np.arange(len(tokens))
        
        for idx, (metric_key, values) in enumerate(metrics.items()):
            ax = axes[idx]
            
            # Create bar plot
            bars = ax.bar(x, values, color=colors[idx % len(colors)], alpha=0.7, edgecolor='black')
            
            # Highlight maximum
            max_idx = np.argmax(values)
            bars[max_idx].set_color('red')
            bars[max_idx].set_alpha(1.0)
            
            ax.set_title(f"{metric_labels[metric_key]} - Max: {tokens[max_idx]} ({values[max_idx]:.3f})", 
                        fontweight='bold')
            ax.set_ylabel(metric_labels[metric_key], fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=8, rotation=45)
        
        # Set x-axis labels only for bottom subplot
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(tokens, rotation=45, ha='right')
        axes[-1].set_xlabel("Tokens", fontweight='bold')
        
        # Remove x-axis labels for other subplots
        for ax in axes[:-1]:
            ax.set_xticks([])
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"❌ Error creating token-level analysis plot: {e}")
        import traceback
        traceback.print_exc()

# Export all functions
__all__ = [
    'plot_conditional_entropy_metrics',
    'create_dataset_overlay_plot',
    'create_comparison_plot',
    'plot_entropy_surprisal_simple',
    'create_entropy_distribution_plot',
    'create_token_level_analysis_plot'
]
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 23:13:15 2025

@author: niran
"""

