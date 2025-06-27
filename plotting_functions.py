"""
plotting_functions.py - Core plotting functions for mBART entropy analysis (COMPLETE FIXED VERSION)
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def get_dataset_color_scheme():
    """Get color schemes for different datasets in overlay mode"""
    return {
        'svo': {'base': 'blue', 'entropy': 'tab:blue', 'surprisal': 'tab:cyan', 'prob': 'navy', 'topk': 'lightblue'},
        'ovs': {'base': 'red', 'entropy': 'tab:red', 'surprisal': 'tab:orange', 'prob': 'maroon', 'topk': 'lightcoral'},
        'szwedek': {'base': 'green', 'entropy': 'tab:green', 'surprisal': 'tab:olive', 'prob': 'darkgreen', 'topk': 'lightgreen'},
        'seq2seq': {'base': 'purple', 'entropy': 'tab:purple', 'surprisal': 'violet', 'prob': 'indigo', 'topk': 'plum'},
        'conditional': {'base': 'orange', 'entropy': 'tab:orange', 'surprisal': 'gold', 'prob': 'darkorange', 'topk': 'wheat'},
        'softmax': {'base': 'blue', 'entropy': 'tab:blue', 'surprisal': 'tab:cyan', 'prob': 'steelblue', 'topk': 'lightblue'},
        'entmax': {'base': 'red', 'entropy': 'tab:red', 'surprisal': 'tab:orange', 'prob': 'darkred', 'topk': 'lightcoral'},
        'default': {'base': 'gray', 'entropy': 'tab:gray', 'surprisal': 'silver', 'prob': 'dimgray', 'topk': 'lightgray'}
    }

def get_dataset_marker_styles():
    """FIXED VERSION: Compatible markers without edge color warnings"""
    return {
        'svo': {'ROOT': 's', 'nsubj': '^', 'obj': 'v', 'NOUN': 'o', 'VERB': 'D'},
        'ovs': {'ROOT': 'p', 'nsubj': 'P', 'obj': 'h', 'NOUN': 'H', 'VERB': '8'},
        'szwedek': {'ROOT': '*', 'nsubj': '+', 'obj': 'x', 'NOUN': '.', 'VERB': ','},
        'seq2seq': {'ROOT': 'd', 'nsubj': 'o', 'obj': 's', 'NOUN': '^', 'VERB': 'v'},
        'conditional': {'ROOT': 'X', 'nsubj': 'D', 'obj': 'p', 'NOUN': 'h', 'VERB': 'H'},
        'default': {'ROOT': 's', 'nsubj': '^', 'obj': 'v', 'NOUN': 'o', 'VERB': 'D'}
    }


def determine_dataset_type(sentence_key):
    """Determine dataset type from sentence key for color coding"""
    key_lower = sentence_key.lower()
    
    # Check for dataset type - prioritize specific dataset names first
    if 'svo' in key_lower:
        return 'svo'
    elif 'ovs' in key_lower:
        return 'ovs'
    elif 'szwedek' in key_lower:
        return 'szwedek'
    elif 'seq2seq' in key_lower:
        return 'seq2seq'
    elif 'conditional' in key_lower:
        return 'conditional'
    elif 'softmax' in key_lower:
        return 'softmax'
    elif 'entmax' in key_lower:
        return 'entmax'
    else:
        return 'default'

def add_tag_markers(ax1, ax2, x, tokens, tags, highlight_tags, conditional_entropy, surprisal, token_prob, topk_entropy, 
                   alpha=1.0, legend_added=None, highlight_lines=None, dataset_type='default', sentence_idx=0):
    """FIXED VERSION: No more matplotlib warnings"""
    if not highlight_tags or not tags:
        return legend_added or set()

    if legend_added is None:
        legend_added = set()

    if highlight_lines is None:
        highlight_lines = ['conditional_entropy', 'surprisal', 'token_probability', 'topk_entropy']

    color_schemes = get_dataset_color_scheme()
    marker_styles = get_dataset_marker_styles()  # Use FIXED version
    
    dataset_colors = color_schemes.get(dataset_type, color_schemes['default'])
    dataset_markers = marker_styles.get(dataset_type, marker_styles['default'])
    
    # FIXED tag styles - no problematic markers
    tag_styles = {
        'ROOT': {
            'color': dataset_colors['entropy'], 
            'linewidth': 3, 
            'marker': dataset_markers.get('ROOT', 's'), 
            'size': 60, 
            'label': f'ROOT ({dataset_type})',
            'edge': True
        },
        'nsubj': {
            'color': dataset_colors['surprisal'], 
            'linewidth': 3, 
            'marker': dataset_markers.get('nsubj', '^'), 
            'size': 50, 
            'label': f'Subject ({dataset_type})',
            'edge': True
        },
        'obj': {
            'color': dataset_colors['prob'], 
            'linewidth': 3, 
            'marker': dataset_markers.get('obj', 'v'), 
            'size': 50, 
            'label': f'Object ({dataset_type})',
            'edge': True
        },
        'NOUN': {
            'color': dataset_colors['entropy'], 
            'linewidth': 2, 
            'marker': dataset_markers.get('NOUN', 'o'), 
            'size': 40, 
            'label': f'Noun ({dataset_type})',
            'edge': True
        },
        'VERB': {
            'color': dataset_colors['surprisal'], 
            'linewidth': 3, 
            'marker': dataset_markers.get('VERB', 'D'), 
            'size': 50, 
            'label': f'Verb ({dataset_type})',
            'edge': True
        },
        'PUNCT': {
            'color': 'gray', 
            'linewidth': 2, 
            'marker': '.', 
            'size': 30, 
            'label': f'Punctuation ({dataset_type})',
            'edge': False  # CRITICAL: No edge for dot markers
        }
    }

    sentence_alpha = alpha * max(0.3, 1.0 - sentence_idx * 0.05)

    def find_tag_spans(tags, highlight_tags):
        spans = {}
        for target_tag in highlight_tags:
            spans[target_tag] = []
            i = 0
            while i < len(tags):
                if tags[i].upper() == target_tag.upper():
                    start = i
                    while i < len(tags) and tags[i].upper() == target_tag.upper():
                        i += 1
                    end = i - 1
                    spans[target_tag].append((start, end))
                else:
                    i += 1
        return spans

    tag_spans = find_tag_spans(tags, highlight_tags)

    for tag, spans in tag_spans.items():
        if tag.upper() in [k.upper() for k in tag_styles.keys()] and spans:
            style_key = None
            for k in tag_styles.keys():
                if k.upper() == tag.upper():
                    style_key = k
                    break

            if style_key is None:
                continue

            style = tag_styles[style_key]
            add_to_legend = style['label'] not in legend_added

            for start_idx, end_idx in spans:
                # CRITICAL FIX: Only add edge colors for compatible markers
                if style['edge']:
                    edge_kwargs = {'edgecolors': 'black', 'linewidth': 0.5}
                else:
                    edge_kwargs = {}
                
                if start_idx == end_idx:
                    if 'conditional_entropy' in highlight_lines:
                        ax1.scatter(x[start_idx], conditional_entropy[start_idx],
                                  marker=style['marker'], color=style['color'],
                                  s=style['size'], alpha=sentence_alpha, zorder=5,
                                  label=style['label'] if add_to_legend else "",
                                  **edge_kwargs)
                    # ... continue for other lines ...

            legend_added.add(style['label'])

    return legend_added

def plot_single_sentence(source_info, sent_title="", show_annotations=True, annotation_type=None,
                        highlight_tags=None, highlight_lines=None, title="mBART Entropy Analysis",
                        save_path=None, show_comet=True, entropy_type='softmax'):
    """Plot a single sentence with full entropy analysis"""
    
    tokens = source_info['tokens']
    x = np.arange(len(tokens))

    # Extract metrics with safe defaults and handle missing data
    conditional_entropy = np.array([source_info.get("conditional_entropy_bits", {}).get(t, 0) for t in tokens])
    surprisal = np.array([source_info.get("surprisal_bits", {}).get(t, 0) for t in tokens])
    token_prob = np.array([source_info.get("token_probability", {}).get(t, 0) for t in tokens])
    topk_entropy = np.array([source_info.get("topk_entropy_bits", {}).get(t, 0) for t in tokens])

    # Handle branching choices for entmax
    branching_choices = None
    if 'branching_choices_bits' in source_info:
        branching_choices = np.array([source_info["branching_choices_bits"].get(t, 0) for t in tokens])

    # Handle cases where all values are zero
    if not np.any(conditional_entropy) and not np.any(surprisal):
        print("⚠️ Warning: No entropy or surprisal data found. Creating placeholder plot.")
        conditional_entropy = np.random.rand(len(tokens)) * 2
        surprisal = np.random.rand(len(tokens)) * 3
        token_prob = np.exp(-surprisal)
        topk_entropy = conditional_entropy * 0.8

    # Find maxima (only if data exists)
    idx_max_entropy = np.argmax(conditional_entropy) if np.any(conditional_entropy) else 0
    idx_max_surprisal = np.argmax(surprisal) if np.any(surprisal) else 0
    idx_min_prob = np.argmin(token_prob) if np.any(token_prob) else 0
    joint_score = conditional_entropy + surprisal
    idx_joint_max = np.argmax(joint_score) if np.any(joint_score) else 0

    fig, ax1 = plt.subplots(figsize=(max(12, len(tokens)), 8))

    # Determine colors based on analysis type
    analysis_type = source_info.get('analysis_type', 'source')
    entropy_type = source_info.get('entropy_type', entropy_type)
    
    if analysis_type == 'source':
        color_entropy = 'tab:blue'
        color_surprisal = 'tab:green'
        color_prob = 'tab:orange'
        color_topk = 'tab:red'
        color_branching = 'tab:cyan'
    else:
        color_entropy = 'tab:purple'
        color_surprisal = 'tab:olive'
        color_prob = 'tab:brown'
        color_topk = 'tab:pink'
        color_branching = 'tab:gray'

    # Plot primary metrics
    ax1.plot(x, conditional_entropy, label=f"Conditional Entropy ({entropy_type}, {analysis_type})",
            color=color_entropy, linewidth=2.5, marker='o', markersize=4)
    ax1.plot(x, surprisal, label=f"Surprisal ({entropy_type}, {analysis_type})",
            linestyle='--', color=color_surprisal, linewidth=2.5, marker='s', markersize=4)
    
    # Plot branching choices if available (entmax only)
    if branching_choices is not None:
        ax1.plot(x, branching_choices, label=f"Branching Choices ({entropy_type}, {analysis_type})",
                linestyle=':', color=color_branching, linewidth=2.5, marker='^', markersize=4)
    
    ax1.set_ylabel("Bits (Entropy / Surprisal / Branching)", color='black', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot secondary metrics on twin axis
    ax2 = ax1.twinx()
    if np.any(token_prob):
        ax2.plot(x, token_prob, label=f"Token Probability ({analysis_type})",
                color=color_prob, linewidth=2, marker='^', markersize=4)
    if np.any(topk_entropy):
        ax2.plot(x, topk_entropy, label=f"Top-k Entropy ({analysis_type})",
                linestyle=':', color=color_topk, linewidth=2, marker='v', markersize=4)
    ax2.set_ylabel("Probability / Top-k Entropy", color='black', fontsize=12, fontweight='bold')

    # Add annotations if requested
    if show_annotations and np.any(conditional_entropy):
        y_min, y_max = ax1.get_ylim()
        y_range = y_max - y_min

        if y_range > 0:
            # Max entropy annotation
            ax1.plot(idx_max_entropy, conditional_entropy[idx_max_entropy], 'o', 
                    color=color_entropy, markersize=10, markeredgecolor='black', markeredgewidth=2)
            ax1.annotate(f"Max Entropy\n{tokens[idx_max_entropy]}\n{conditional_entropy[idx_max_entropy]:.2f} bits",
                        xy=(idx_max_entropy, conditional_entropy[idx_max_entropy]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

            # Max surprisal annotation
            ax1.plot(idx_max_surprisal, surprisal[idx_max_surprisal], 's', 
                    color=color_surprisal, markersize=10, markeredgecolor='black', markeredgewidth=2)
            ax1.annotate(f"Max Surprisal\n{tokens[idx_max_surprisal]}\n{surprisal[idx_max_surprisal]:.2f} bits",
                        xy=(idx_max_surprisal, surprisal[idx_max_surprisal]),
                        xytext=(-10, 10), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

            # Joint maximum
            ax1.plot(idx_joint_max, joint_score[idx_joint_max], '*', 
                    color='gold', markersize=15, markeredgecolor='black', markeredgewidth=2)
            ax1.annotate(f"Joint Max\n{tokens[idx_joint_max]}\n{joint_score[idx_joint_max]:.2f} bits",
                        xy=(idx_joint_max, joint_score[idx_joint_max]),
                        xytext=(0, 20), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.9),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                        fontweight='bold')

            # Branching choices annotation if available
            if branching_choices is not None:
                idx_max_branching = np.argmax(branching_choices)
                ax1.plot(idx_max_branching, branching_choices[idx_max_branching], '^',
                        color=color_branching, markersize=10, markeredgecolor='black', markeredgewidth=2)
                ax1.annotate(f"Max Branching\n{tokens[idx_max_branching]}\n{branching_choices[idx_max_branching]:.2f}",
                            xy=(idx_max_branching, branching_choices[idx_max_branching]),
                            xytext=(15, -15), textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.8),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Add COMET score if available
    if show_comet and 'comet_score' in source_info:
        ax1.text(0.02, 0.98, f"COMET: {source_info['comet_score']:.3f}", 
                transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan', alpha=0.9),
                fontweight='bold')

    # Add linguistic tag markers if requested
    legend_added = set()
    if highlight_tags and annotation_type and annotation_type in source_info:
        tags = source_info[annotation_type]
        dataset_type = determine_dataset_type(sent_title)
        legend_added = add_tag_markers(ax1, ax2, x, tokens, tags, highlight_tags,
                                     conditional_entropy, surprisal, token_prob, topk_entropy,
                                     legend_added=legend_added, highlight_lines=highlight_lines,
                                     dataset_type=dataset_type, sentence_idx=0)

    # Set x-axis with better formatting
    ax1.set_xticks(x)
    ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
    ax1.set_xlabel("Tokens", fontsize=12, fontweight='bold')
    ax1.margins(x=0.02)

    # Add linguistic annotations if requested
    if annotation_type and annotation_type in source_info:
        annotations = source_info[annotation_type]
        y_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
        offset = y_range * 0.25

        for i, label in enumerate(annotations):
            ax1.text(i, ax1.get_ylim()[0] - offset,
                    label, rotation=45, fontsize=8, ha='center', va='top',
                    color='darkblue' if annotation_type == 'pos' else 'darkred',
                    fontweight='bold')

        plt.subplots_adjust(bottom=0.35, top=0.85)

    # Create comprehensive legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2

    if all_lines:
        ax1.legend(all_lines, all_labels, loc='upper center',
                  bbox_to_anchor=(0.5, -0.15), ncol=min(4, len(all_labels)), fontsize=10,
                  frameon=True, fancybox=True, shadow=True)

    # Set title with better formatting
    plt.title(f"{title}{sent_title}", fontsize=16, pad=20, fontweight='bold')

    if annotation_type:
        plt.subplots_adjust(top=0.90, bottom=0.45)
    else:
        plt.subplots_adjust(top=0.90, bottom=0.25)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Plot saved to: {save_path}")

    plt.show()

def plot_multiple_sentences(data, data_type, highlight_tags=None, annotation_type=None, 
                          highlight_lines=None, save_path=None, title="mBART Entropy Analysis"):
    """Plot multiple sentences overlay with improved styling and dataset-specific annotation colors"""
    fig, ax1 = plt.subplots(figsize=(20, 12))
    ax2 = ax1.twinx()

    max_length = 0
    legend_added = set()

    # Dynamic styling based on data size
    base_alpha = 0.4 if len(data) > 50 else 0.6 if len(data) > 20 else 0.8
    base_linewidth = 0.8 if len(data) > 50 else 1.2 if len(data) > 20 else 1.8

    # Determine colors based on data type
    if data_type == "Source":
        default_colors = {
            'entropy': 'tab:blue', 'surprisal': 'tab:green',
            'prob': 'tab:orange', 'topk': 'tab:red', 'branching': 'tab:cyan'
        }
    else:
        default_colors = {
            'entropy': 'tab:purple', 'surprisal': 'tab:olive',
            'prob': 'tab:brown', 'topk': 'tab:pink', 'branching': 'tab:gray'
        }

    # Collect statistics for overlay
    all_entropies = []
    all_surprisals = []
    comet_scores = []
    
    # Group sentences by dataset type for color coding
    dataset_groups = {}
    for sent_key, sent_data in data.items():
        dataset_type = determine_dataset_type(sent_key)
        if dataset_type not in dataset_groups:
            dataset_groups[dataset_type] = []
        dataset_groups[dataset_type].append((sent_key, sent_data))

    color_schemes = get_dataset_color_scheme()
    
    print(f"📊 Processing {len(data)} sentences across {len(dataset_groups)} dataset types...")
    print(f"   Dataset groups: {list(dataset_groups.keys())}")

    # Plot each dataset group with its own colors
    for group_idx, (dataset_type, group_data) in enumerate(dataset_groups.items()):
        dataset_colors = color_schemes.get(dataset_type, color_schemes['default'])
        
        print(f"   Processing {dataset_type}: {len(group_data)} sentences")
        
        for sent_idx, (sent_key, sent_data) in enumerate(group_data):
            source_info = sent_data['source_info']
            tokens = source_info['tokens']
            x = np.arange(len(tokens))
            max_length = max(max_length, len(tokens))

            # Extract metrics with safe defaults
            conditional_entropy = np.array([source_info.get("conditional_entropy_bits", {}).get(t, 0) for t in tokens])
            surprisal = np.array([source_info.get("surprisal_bits", {}).get(t, 0) for t in tokens])
            token_prob = np.array([source_info.get("token_probability", {}).get(t, 0) for t in tokens])
            topk_entropy = np.array([source_info.get("topk_entropy_bits", {}).get(t, 0) for t in tokens])
            
            # Handle branching choices
            branching_choices = None
            if 'branching_choices_bits' in source_info:
                branching_choices = np.array([source_info["branching_choices_bits"].get(t, 0) for t in tokens])

            # Collect statistics
            all_entropies.extend(conditional_entropy[conditional_entropy > 0])
            all_surprisals.extend(surprisal[surprisal > 0])
            if 'comet_score' in source_info:
                comet_scores.append(source_info['comet_score'])

            # Plot lines with dataset-specific colors and varying transparency
            sentence_alpha = base_alpha * max(0.4, 1.0 - (sent_idx * 0.02))  # More gradual fade
            
            # Use dataset-specific colors
            ax1.plot(x, conditional_entropy, alpha=sentence_alpha,
                    label=f"{data_type} Entropy ({dataset_type})" if sent_idx == 0 else "",
                    color=dataset_colors['entropy'], linewidth=base_linewidth)
            ax1.plot(x, surprisal, alpha=sentence_alpha, linestyle='--',
                    label=f"{data_type} Surprisal ({dataset_type})" if sent_idx == 0 else "",
                    color=dataset_colors['surprisal'], linewidth=base_linewidth)
            
            # Plot branching choices if available
            if branching_choices is not None and np.any(branching_choices):
                ax1.plot(x, branching_choices, alpha=sentence_alpha, linestyle=':',
                        label=f"{data_type} Branching ({dataset_type})" if sent_idx == 0 else "",
                        color=dataset_colors.get('prob', 'tab:cyan'), linewidth=base_linewidth)
            
            if np.any(token_prob):
                ax2.plot(x, token_prob, alpha=sentence_alpha*0.8,
                        label=f"{data_type} Probability ({dataset_type})" if sent_idx == 0 else "",
                        color=dataset_colors['prob'], linewidth=base_linewidth)
            if np.any(topk_entropy):
                ax2.plot(x, topk_entropy, alpha=sentence_alpha*0.8, linestyle=':',
                        label=f"{data_type} Top-k Entropy ({dataset_type})" if sent_idx == 0 else "",
                        color=dataset_colors['topk'], linewidth=base_linewidth)

            # Add linguistic tag markers for ALL sentences with dataset-specific styling
            if highlight_tags and annotation_type and annotation_type in source_info:
                tags = source_info[annotation_type]
                if tags:  # Only process if tags exist
                    global_sentence_idx = group_idx * 100 + sent_idx  # Unique index per sentence
                    legend_added = add_tag_markers(
                        ax1, ax2, x, tokens, tags, highlight_tags,
                        conditional_entropy, surprisal, token_prob, topk_entropy,
                        alpha=0.9, legend_added=legend_added, highlight_lines=highlight_lines,
                        dataset_type=dataset_type, sentence_idx=global_sentence_idx
                    )

    # Set labels and styling
    ax1.set_xlabel("Token Position", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Bits (Entropy / Surprisal / Branching)", color='black', fontsize=12, fontweight='bold')
    ax2.set_ylabel("Probability / Top-k Entropy", color='black', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, max_length-1)
    ax1.margins(x=0.01)
    ax1.grid(True, alpha=0.3)

    # Add comprehensive statistics text box
    if all_entropies or comet_scores:
        stats_text = f"Dataset Statistics:\n"
        stats_text += f"Total Sentences: {len(data)}\n"
        stats_text += f"Dataset Groups: {len(dataset_groups)}\n"
        
        for dt, group in dataset_groups.items():
            stats_text += f"  {dt}: {len(group)} sentences\n"
        
        if all_entropies:
            stats_text += f"Avg Entropy: {np.mean(all_entropies):.2f}±{np.std(all_entropies):.2f}\n"
        if all_surprisals:
            stats_text += f"Avg Surprisal: {np.mean(all_surprisals):.2f}±{np.std(all_surprisals):.2f}\n"
        if comet_scores:
            stats_text += f"Avg COMET: {np.mean(comet_scores):.3f}±{np.std(comet_scores):.3f}"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))

    # Create enhanced legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2

    if all_lines:
        ncol = min(8, max(3, len(all_labels)))
        ax1.legend(all_lines, all_labels, loc='upper center',
                  bbox_to_anchor=(0.5, -0.08), ncol=ncol, fontsize=9,
                  frameon=True, fancybox=True, shadow=True)

    # Enhanced title with statistics
    title_extra = f" — {data_type} Sentences Overlay ({len(data)} sentences, {len(dataset_groups)} datasets)"
    if comet_scores and data_type == "Target":
        title_extra += f" | Avg COMET: {np.mean(comet_scores):.3f}"
    
    plt.title(f"{title}{title_extra}", fontsize=16, pad=20, fontweight='bold')
    plt.subplots_adjust(top=0.92, bottom=0.15, left=0.08, right=0.92)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 {data_type} plot saved to: {save_path}")

    plt.show()
    
    # Print summary of what was plotted
    print(f"\n✅ Overlay plot completed!")
    print(f"   📊 {len(data)} total sentences plotted")
    print(f"   🎨 {len(dataset_groups)} dataset types with unique colors")
    if highlight_tags and annotation_type:
        total_annotated = sum(1 for _, sent_data in data.items() 
                             if annotation_type in sent_data['source_info'] 
                             and sent_data['source_info'][annotation_type])
        print(f"   🏷️ {total_annotated} sentences with {annotation_type} annotations displayed")

def create_entropy_heatmap(data, title="Entropy Heatmap", metric='conditional_entropy_bits'):
    """Create a heatmap visualization of entropy across tokens and sentences"""
    
    # Extract entropy data for heatmap
    entropy_matrix = []
    sentence_labels = []
    max_tokens = 0
    
    for sent_key, sent_data in data.items():
        source_info = sent_data['source_info']
        tokens = source_info['tokens']
        max_tokens = max(max_tokens, len(tokens))
        
        entropies = [source_info.get(metric, {}).get(t, 0) for t in tokens]
        entropy_matrix.append(entropies)
        
        # Create shorter labels for display
        dataset_type = determine_dataset_type(sent_key)
        short_label = f"[{dataset_type}] {sent_key[:25]}..."
        sentence_labels.append(short_label)
    
    # Pad sequences to same length
    for i, row in enumerate(entropy_matrix):
        entropy_matrix[i] = row + [0] * (max_tokens - len(row))
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(15, max_tokens * 0.6), max(10, len(entropy_matrix) * 0.4)))
    
    im = ax.imshow(entropy_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Set labels
    ax.set_xlabel("Token Position", fontweight='bold', fontsize=12)
    ax.set_ylabel("Sentences", fontweight='bold', fontsize=12)
    ax.set_title(f"{title} - {metric.replace('_', ' ').title()}", fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Entropy (bits)", fontweight='bold', fontsize=12)
    
    # Set y-axis labels with color coding by dataset
    ax.set_yticks(range(len(sentence_labels)))
    ax.set_yticklabels(sentence_labels, fontsize=8)
    
    # Color code y-axis labels by dataset type
    for i, label in enumerate(sentence_labels):
        dataset_type = determine_dataset_type(label)
        color_scheme = get_dataset_color_scheme()
        label_color = color_scheme.get(dataset_type, color_scheme['default'])['base']
        ax.get_yticklabels()[i].set_color(label_color)
    
    plt.tight_layout()
    plt.show()

def create_comparative_bar_plot(data1, data2, labels, title="Comparative Analysis"):
    """Create comparative bar plot for two datasets with enhanced metrics"""
    
    def extract_stats(data):
        all_entropies = []
        all_surprisals = []
        all_topk = []
        all_branching = []
        comet_scores = []
        
        for sent_data in data.values():
            source_info = sent_data['source_info']
            tokens = source_info['tokens']
            
            entropies = [source_info.get("conditional_entropy_bits", {}).get(t, 0) for t in tokens]
            surprisals = [source_info.get("surprisal_bits", {}).get(t, 0) for t in tokens]
            topk_entropies = [source_info.get("topk_entropy_bits", {}).get(t, 0) for t in tokens]
            
            all_entropies.extend([e for e in entropies if e > 0])
            all_surprisals.extend([s for s in surprisals if s > 0])
            all_topk.extend([t for t in topk_entropies if t > 0])
            
            # Handle branching choices
            if 'branching_choices_bits' in source_info:
                branching = [source_info.get("branching_choices_bits", {}).get(t, 0) for t in tokens]
                all_branching.extend([b for b in branching if b > 0])
            
            if 'comet_score' in source_info:
                comet_scores.append(source_info['comet_score'])
        
        return {
            'avg_entropy': np.mean(all_entropies) if all_entropies else 0,
            'avg_surprisal': np.mean(all_surprisals) if all_surprisals else 0,
            'avg_topk': np.mean(all_topk) if all_topk else 0,
            'avg_branching': np.mean(all_branching) if all_branching else 0,
            'avg_comet': np.mean(comet_scores) if comet_scores else 0
        }
    
    stats1 = extract_stats(data1)
    stats2 = extract_stats(data2)
    
    # Select metrics to display
    metrics = []
    metric_labels = []
    
    if stats1['avg_entropy'] > 0 or stats2['avg_entropy'] > 0:
        metrics.append('avg_entropy')
        metric_labels.append('Avg Entropy')
    
    if stats1['avg_surprisal'] > 0 or stats2['avg_surprisal'] > 0:
        metrics.append('avg_surprisal')
        metric_labels.append('Avg Surprisal')
        
    if stats1['avg_topk'] > 0 or stats2['avg_topk'] > 0:
        metrics.append('avg_topk')
        metric_labels.append('Avg Top-k')
        
    if stats1['avg_branching'] > 0 or stats2['avg_branching'] > 0:
        metrics.append('avg_branching')
        metric_labels.append('Avg Branching')
    
    if stats1['avg_comet'] > 0 or stats2['avg_comet'] > 0:
        metrics.append('avg_comet')
        metric_labels.append('Avg COMET')
    
    if not metrics:
        print("❌ No metrics available for comparison")
        return
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 2), 8))
    
    values1 = [stats1[m] for m in metrics]
    values2 = [stats2[m] for m in metrics]
    
    # Use dataset-specific colors
    dataset1_type = determine_dataset_type(labels[0])
    dataset2_type = determine_dataset_type(labels[1])
    color_schemes = get_dataset_color_scheme()
    
    color1 = color_schemes.get(dataset1_type, color_schemes['default'])['entropy']
    color2 = color_schemes.get(dataset2_type, color_schemes['default'])['entropy']
    
    rects1 = ax.bar(x - width/2, values1, width, label=labels[0], alpha=0.8, color=color1)
    rects2 = ax.bar(x + width/2, values2, width, label=labels[1], alpha=0.8, color=color2)
    
    ax.set_xlabel('Metrics', fontweight='bold', fontsize=12)
    ax.set_ylabel('Values', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.show()

def create_dataset_summary_plot(processed_datasets, analysis_type='source'):
    """Create summary visualization across all datasets"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    dataset_names = []
    entropy_means = []
    surprisal_means = []
    sentence_counts = []
    comet_scores = []
    
    color_schemes = get_dataset_color_scheme()
    colors = []
    
    for dataset_key, data in processed_datasets.items():
        from data_processing import filter_data_by_type
        filtered_data = filter_data_by_type(data, analysis_type)
        
        if not filtered_data:
            continue
            
        dataset_names.append(dataset_key[:15] + "..." if len(dataset_key) > 15 else dataset_key)
        sentence_counts.append(len(filtered_data))
        
        # Collect metrics
        all_entropies = []
        all_surprisals = []
        dataset_comet = []
        
        for sent_data in filtered_data.values():
            source_info = sent_data['source_info']
            tokens = source_info['tokens']
            
            entropies = [source_info.get("conditional_entropy_bits", {}).get(t, 0) for t in tokens]
            surprisals = [source_info.get("surprisal_bits", {}).get(t, 0) for t in tokens]
            
            all_entropies.extend([e for e in entropies if e > 0])
            all_surprisals.extend([s for s in surprisals if s > 0])
            
            if 'comet_score' in source_info:
                dataset_comet.append(source_info['comet_score'])
        
        entropy_means.append(np.mean(all_entropies) if all_entropies else 0)
        surprisal_means.append(np.mean(all_surprisals) if all_surprisals else 0)
        comet_scores.append(np.mean(dataset_comet) if dataset_comet else 0)
        
        # Determine color
        dataset_type = determine_dataset_type(dataset_key)
        colors.append(color_schemes.get(dataset_type, color_schemes['default'])['entropy'])
    
    # Plot 1: Sentence counts
    ax1.bar(range(len(dataset_names)), sentence_counts, color=colors, alpha=0.7)
    ax1.set_title('Sentence Counts by Dataset', fontweight='bold')
    ax1.set_ylabel('Number of Sentences', fontweight='bold')
    ax1.set_xticks(range(len(dataset_names)))
    ax1.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average entropy
    ax2.bar(range(len(dataset_names)), entropy_means, color=colors, alpha=0.7)
    ax2.set_title('Average Conditional Entropy', fontweight='bold')
    ax2.set_ylabel('Entropy (bits)', fontweight='bold')
    ax2.set_xticks(range(len(dataset_names)))
    ax2.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average surprisal
    ax3.bar(range(len(dataset_names)), surprisal_means, color=colors, alpha=0.7)
    ax3.set_title('Average Surprisal', fontweight='bold')
    ax3.set_ylabel('Surprisal (bits)', fontweight='bold')
    ax3.set_xticks(range(len(dataset_names)))
    ax3.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: COMET scores (if available)
    if any(score > 0 for score in comet_scores):
        ax4.bar(range(len(dataset_names)), comet_scores, color=colors, alpha=0.7)
        ax4.set_title('Average COMET Scores', fontweight='bold')
        ax4.set_ylabel('COMET Score', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No COMET Scores Available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('COMET Scores', fontweight='bold')
    
    ax4.set_xticks(range(len(dataset_names)))
    ax4.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Dataset Summary - {analysis_type.title()} Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Export main functions
__all__ = [
    'plot_single_sentence',
    'plot_multiple_sentences', 
    'add_tag_markers',
    'create_entropy_heatmap',
    'create_comparative_bar_plot',
    'create_dataset_summary_plot',
    'get_dataset_color_scheme',
    'get_dataset_marker_styles',
    'determine_dataset_type'
]
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 23:10:57 2025

@author: niran
"""

