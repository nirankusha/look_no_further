"""
widget_interface_fixed.py - Complete working version with all fixes
Use this instead of the broken widget_interface.py
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import json
import traceback
import numpy as np
from data_processing import (
    load_preloaded_datasets, 
    process_all_datasets, 
    get_available_highlight_tags, 
    filter_data_by_type, 
    get_dataset_statistics,
    validate_dataset_structure,
    safe_get_metric_data,
    get_entropy_fields_dynamic
)
from main_plotting import plot_conditional_entropy_metrics, create_dataset_overlay_plot, create_comparison_plot

# Global variables for data storage
loaded_datasets = {}
processed_datasets = {}

# Global update function references
update_dataset_selector_ref = None
update_visualization_interface_ref = None
update_sentence_analysis_interface_ref = None

def safe_statistics(values, metric_name="values"):
    """Safely calculate statistics, handling inf and nan values"""
    if not values:
        return {
            'count': 0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'valid_count': 0
        }
    
    # Convert to numpy array and filter out inf/nan values
    arr = np.array(values)
    
    # Remove infinite and NaN values
    valid_mask = np.isfinite(arr)
    valid_values = arr[valid_mask]
    
    if len(valid_values) == 0:
        return {
            'count': len(values),
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'valid_count': 0,
            'warning': f"All {metric_name} values are infinite or NaN"
        }
    
    return {
        'count': len(values),
        'mean': float(np.mean(valid_values)),
        'std': float(np.std(valid_values)),
        'min': float(np.min(valid_values)),
        'max': float(np.max(valid_values)),
        'valid_count': len(valid_values),
        'invalid_count': len(values) - len(valid_values)
    }

def safe_argmax(values, tokens):
    """Safely find argmax, handling inf and nan values"""
    if not values or not tokens:
        return 0, "N/A"
    
    arr = np.array(values)
    valid_mask = np.isfinite(arr)
    
    if not np.any(valid_mask):
        return 0, tokens[0] if tokens else "N/A"
    
    # Find max among valid values
    valid_indices = np.where(valid_mask)[0]
    valid_values = arr[valid_mask]
    
    max_idx_in_valid = np.argmax(valid_values)
    actual_idx = valid_indices[max_idx_in_valid]
    
    return actual_idx, tokens[actual_idx] if actual_idx < len(tokens) else "N/A"

def clean_entropy_data(source_info, tokens):
    """Clean entropy data using proper field detection and cleaning"""
    
    # Get entropy type and analysis type
    entropy_type = source_info.get('entropy_type', 'softmax')
    analysis_type = source_info.get('analysis_type', 'source')
    
    print(f"🔧 CLEANING: {entropy_type} {analysis_type} data")
    
    # Use field detection
    fields = get_entropy_fields_dynamic(entropy_type, analysis_type, source_info)
    
    if not fields:
        print(f"   ❌ NO FIELDS DETECTED! Using fallback structure.")
        # Return basic structure with zeros
        return {
            'conditional_entropy_bits': {token: 0.0 for token in tokens},
            'surprisal_bits': {token: 0.0 for token in tokens},
            'token_probability': {token: 0.01 for token in tokens},
            'topk_entropy_bits': {token: 0.0 for token in tokens}
        }
    
    cleaned_info = {}
    
    # Clean each field using safe extraction
    for metric_name, field_name in fields.items():
        cleaned_data = safe_get_metric_data(source_info, field_name, tokens)
        cleaned_info[f'{metric_name}_bits'] = cleaned_data
        
        # Debug: show actual data
        non_zero = sum(1 for v in cleaned_data.values() if v > 0)
        if non_zero > 0:
            max_val = max(cleaned_data.values())
            print(f"   ✅ {metric_name}_bits: {non_zero}/{len(tokens)} non-zero, max={max_val:.3f}")
        else:
            print(f"   ⚠️ {metric_name}_bits: ALL ZEROS - check field mapping!")
    
    # Ensure token_probability exists
    if 'token_probability' not in cleaned_info:
        surprisal_data = cleaned_info.get('surprisal_bits', {})
        if any(surprisal_data.values()):
            cleaned_info['token_probability'] = {
                token: np.exp(-surprisal) if surprisal > 0 else 0.01
                for token, surprisal in surprisal_data.items()
            }
        else:
            cleaned_info['token_probability'] = {token: 0.01 for token in tokens}
    
    return cleaned_info

def create_file_upload_interface():
    """Create interface for file upload with preloaded datasets"""
    
    upload_widget = widgets.FileUpload(
        accept='.json',
        multiple=True,
        description="Upload JSON files",
        layout=widgets.Layout(width='300px')
    )
    
    load_preloaded_button = widgets.Button(
        description="Load Preloaded Datasets",
        button_style='info',
        layout=widgets.Layout(width='200px')
    )
    
    upload_output = widgets.Output()
    
    def handle_upload(change):
        global loaded_datasets
        with upload_output:
            clear_output()
            
            if not change['new']:
                return
                
            print("📁 Processing uploaded files...")
            
            for uploaded_file in change['new'].values():
                filename = uploaded_file['metadata']['name']
                content = uploaded_file['content']
                
                try:
                    print(f"Processing {filename}...")
                    
                    # Handle different content types more robustly
                    if hasattr(content, 'tobytes'):
                        file_content = content.tobytes().decode('utf-8')
                    elif isinstance(content, bytes):
                        file_content = content.decode('utf-8')
                    elif hasattr(content, 'read'):
                        file_content = content.read()
                        if isinstance(file_content, bytes):
                            file_content = file_content.decode('utf-8')
                    else:
                        file_content = str(content)
                    
                    # Parse JSON
                    data = json.loads(file_content)
                    
                    # Validate structure
                    is_valid, message = validate_dataset_structure(data)
                    if not is_valid:
                        print(f"❌ Invalid dataset structure in {filename}: {message}")
                        continue
                    
                    # Store dataset
                    dataset_name = filename.replace('.json', '')
                    loaded_datasets[dataset_name] = data
                    
                    print(f"✅ Successfully loaded: {filename}")
                    print(f"   - Dataset name: {dataset_name}")
                    print(f"   - Number of entries: {len(data)}")
                    
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing error in {filename}: {e}")
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
                    traceback.print_exc()
            
            print(f"\n📊 Total loaded datasets: {len(loaded_datasets)}")
            update_all_interfaces()
    
    def on_load_preloaded_click(b):
        global loaded_datasets
        with upload_output:
            clear_output()
            print("🔄 Loading preloaded datasets...")
            
            try:
                preloaded = load_preloaded_datasets()
                loaded_datasets.update(preloaded)
                
                if preloaded:
                    print(f"\n✅ Successfully loaded {len(preloaded)} preloaded datasets")
                    print(f"📊 Total datasets available: {len(loaded_datasets)}")
                    update_all_interfaces()
                else:
                    print("⚠️ No preloaded datasets found. Check file paths.")
            except Exception as e:
                print(f"❌ Error loading preloaded datasets: {e}")
                traceback.print_exc()
    
    upload_widget.observe(handle_upload, names='value')
    load_preloaded_button.on_click(on_load_preloaded_click)
    
    upload_interface = widgets.VBox([
        widgets.HTML("<h3>📁 File Upload</h3>"),
        widgets.HTML("<p><strong>Option 1:</strong> Upload your own JSON files</p>"),
        upload_widget,
        widgets.HTML("<p><strong>Option 2:</strong> Load preloaded datasets (SVO, OVS, Szwedek)</p>"),
        load_preloaded_button,
        upload_output
    ])
    
    return upload_interface

def create_processing_interface():
    """Create interface for data processing"""
    
    dataset_selector = widgets.SelectMultiple(
        options=[],
        description="Datasets:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px', height='150px')
    )
    
    process_button = widgets.Button(
        description="Process All Selected",
        button_style='primary',
        layout=widgets.Layout(width='200px')
    )
    
    process_output = widgets.Output()
    
    def on_process_click(b):
        global processed_datasets
        with process_output:
            clear_output()
            
            selected_datasets = list(dataset_selector.value)
            if not selected_datasets:
                print("❌ Please select at least one dataset to process")
                return
            
            print(f"⚙️ Processing {len(selected_datasets)} datasets...")
            
            try:
                # Create a subset of loaded_datasets with only selected ones
                selected_data = {name: loaded_datasets[name] for name in selected_datasets if name in loaded_datasets}
                
                if not selected_data:
                    print("❌ No valid datasets found in selection")
                    return
                
                # Process all selected datasets for both entropy types
                new_processed = process_all_datasets(selected_data)
                processed_datasets.update(new_processed)
                
                print(f"\n✅ Processing complete! {len(new_processed)} dataset variants ready for analysis.")
                print(f"📊 Total processed datasets: {len(processed_datasets)}")
                
                # List processed datasets
                print("\nProcessed datasets:")
                for key in new_processed.keys():
                    print(f"  - {key}")
                
                update_all_interfaces()
                
            except Exception as e:
                print(f"❌ Error processing datasets: {e}")
                traceback.print_exc()
    
    process_button.on_click(on_process_click)
    
    def update_dataset_selector():
        """Update dataset selector with available datasets"""
        available_datasets = list(loaded_datasets.keys())
        dataset_selector.options = available_datasets
        print(f"🔄 Updated dataset selector with {len(available_datasets)} datasets")
    
    processing_interface = widgets.VBox([
        widgets.HTML("<h3>⚙️ Data Processing</h3>"),
        widgets.HTML("<p>Select datasets to process for both softmax and entmax entropies:</p>"),
        dataset_selector,
        process_button,
        process_output
    ])
    
    return processing_interface, update_dataset_selector

def create_visualization_interface():
    """Create interface for dataset-level visualization"""
    
    viz_dataset_selector = widgets.SelectMultiple(
        options=[],
        description="Datasets:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px', height='120px')
    )
    
    viz_mode_selector = widgets.RadioButtons(
        options=[('Single Dataset', 'single'), ('Multi-Dataset Overlay', 'overlay'), ('Comparison Plot', 'comparison')],
        value='single',
        description="Mode:",
        style={'description_width': 'initial'}
    )
    
    analysis_type_selector = widgets.RadioButtons(
        options=[('Source', 'source'), ('Target', 'target'), ('Both', 'both')],
        value='source',
        description="Analysis:",
        style={'description_width': 'initial'}
    )
    
    annotation_type_selector = widgets.Dropdown(
        options=[('None', None), ('Part-of-Speech', 'pos'), ('Dependency', 'dep')],
        value=None,
        description="Annotations:",
        style={'description_width': 'initial'},
    )
    
    highlight_tags_selector = widgets.SelectMultiple(
        options=[],
        description="Highlight Tags:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px', height='100px')
    )
    
    highlight_lines_selector = widgets.SelectMultiple(
        options=[
            ('Conditional Entropy', 'conditional_entropy'),
            ('Surprisal', 'surprisal'),
            ('Token Probability', 'token_probability'),
            ('Top-k Entropy', 'topk_entropy')
        ],
        value=['conditional_entropy'],
        description="Lines:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(height='100px')
    )
    
    show_annotations_checkbox = widgets.Checkbox(
        value=True,
        description="Show Annotations"
    )
    
    show_comet_checkbox = widgets.Checkbox(
        value=True,
        description="Show COMET Scores"
    )
    
    save_path_text = widgets.Text(
        value="",
        placeholder="Optional: path to save plot",
        description="Save Path:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px')
    )
    
    plot_button = widgets.Button(
        description="Create Visualization",
        button_style='success',
        layout=widgets.Layout(width='200px')
    )
    
    plot_output = widgets.Output()
    
    def update_highlight_tags(change):
        """Update highlight tags based on annotation type"""
        annotation_type = change['new']
        if annotation_type:
            available_tags = get_available_highlight_tags()
            highlight_tags_selector.options = [(tag, tag) for tag in available_tags[annotation_type]]
        else:
            highlight_tags_selector.options = []
    
    annotation_type_selector.observe(update_highlight_tags, names='value')
    
    def on_plot_click(b):
        with plot_output:
            clear_output()
            
            selected_datasets = list(viz_dataset_selector.value)
            if not selected_datasets:
                print("❌ Please select at least one dataset")
                return
            
            viz_mode = viz_mode_selector.value
            analysis_type = analysis_type_selector.value
            
            print(f"📈 Creating {viz_mode} visualization...")
            print(f"   - Datasets: {selected_datasets}")
            print(f"   - Analysis type: {analysis_type}")
            
            try:
                if viz_mode == 'single':
                    # Single dataset plots
                    for dataset_key in selected_datasets:
                        if dataset_key not in processed_datasets:
                            print(f"❌ Dataset {dataset_key} not processed yet")
                            continue
                        
                        data = processed_datasets[dataset_key]
                        
                        if analysis_type == 'both':
                            # Plot both source and target
                            source_data = filter_data_by_type(data, 'source')
                            target_data = filter_data_by_type(data, 'target')
                            
                            if source_data:
                                plot_conditional_entropy_metrics(
                                    source_data,
                                    annotation_type=annotation_type_selector.value,
                                    show_annotations=show_annotations_checkbox.value,
                                    highlight_tags=list(highlight_tags_selector.value) if highlight_tags_selector.value else None,
                                    title=f"Source Analysis: {dataset_key}",
                                    save_path=f"{save_path_text.value}_source.png" if save_path_text.value.strip() else None,
                                    highlight_lines=list(highlight_lines_selector.value) if highlight_lines_selector.value else None,
                                    show_comet=show_comet_checkbox.value
                                )
                            
                            if target_data:
                                plot_conditional_entropy_metrics(
                                    target_data,
                                    annotation_type=annotation_type_selector.value,
                                    show_annotations=show_annotations_checkbox.value,
                                    highlight_tags=list(highlight_tags_selector.value) if highlight_tags_selector.value else None,
                                    title=f"Target Analysis: {dataset_key}",
                                    save_path=f"{save_path_text.value}_target.png" if save_path_text.value.strip() else None,
                                    highlight_lines=list(highlight_lines_selector.value) if highlight_lines_selector.value else None,
                                    show_comet=show_comet_checkbox.value
                                )
                        else:
                            # Single analysis type
                            filtered_data = filter_data_by_type(data, analysis_type)
                            
                            if filtered_data:
                                plot_conditional_entropy_metrics(
                                    filtered_data,
                                    annotation_type=annotation_type_selector.value,
                                    show_annotations=show_annotations_checkbox.value,
                                    highlight_tags=list(highlight_tags_selector.value) if highlight_tags_selector.value else None,
                                    title=f"{analysis_type.title()} Analysis: {dataset_key}",
                                    save_path=save_path_text.value.strip() if save_path_text.value.strip() else None,
                                    highlight_lines=list(highlight_lines_selector.value) if highlight_lines_selector.value else None,
                                    show_comet=show_comet_checkbox.value
                                )
                            else:
                                print(f"❌ No {analysis_type} data found for {dataset_key}")
                
                elif viz_mode == 'overlay':
                    # Multi-dataset overlay
                    if analysis_type == 'both':
                        # Create separate overlays for source and target
                        create_dataset_overlay_plot(
                            processed_datasets, selected_datasets, 'source',
                            highlight_tags=list(highlight_tags_selector.value) if highlight_tags_selector.value else None,
                            annotation_type=annotation_type_selector.value,
                            save_path=f"{save_path_text.value}_source_overlay.png" if save_path_text.value.strip() else None
                        )
                        
                        create_dataset_overlay_plot(
                            processed_datasets, selected_datasets, 'target',
                            highlight_tags=list(highlight_tags_selector.value) if highlight_tags_selector.value else None,
                            annotation_type=annotation_type_selector.value,
                            save_path=f"{save_path_text.value}_target_overlay.png" if save_path_text.value.strip() else None
                        )
                    else:
                        create_dataset_overlay_plot(
                            processed_datasets, selected_datasets, analysis_type,
                            highlight_tags=list(highlight_tags_selector.value) if highlight_tags_selector.value else None,
                            annotation_type=annotation_type_selector.value,
                            save_path=save_path_text.value.strip() if save_path_text.value.strip() else None
                        )
                
                elif viz_mode == 'comparison':
                    # Comparison plot
                    base_names = list(set([key.split('_')[0] for key in selected_datasets]))
                    entropy_types = list(set([key.split('_')[-1] for key in selected_datasets if '_' in key]))
                    
                    if not entropy_types:
                        entropy_types = ['softmax', 'entmax']
                    
                    create_comparison_plot(
                        processed_datasets, base_names, entropy_types,
                        analysis_type=analysis_type if analysis_type != 'both' else 'source',
                        save_path=save_path_text.value.strip() if save_path_text.value.strip() else None
                    )
                
                print("✅ Visualization created successfully!")
                
            except Exception as e:
                print(f"❌ Error creating visualization: {e}")
                traceback.print_exc()
    
    plot_button.on_click(on_plot_click)
    
    def update_visualization_interface():
        """Update visualization interface with available datasets"""
        available_datasets = list(processed_datasets.keys())
        viz_dataset_selector.options = available_datasets
        print(f"🔄 Updated visualization interface with {len(available_datasets)} processed datasets")
    
    visualization_interface = widgets.VBox([
        widgets.HTML("<h3>📈 Dataset Visualization</h3>"),
        viz_dataset_selector,
        viz_mode_selector,
        analysis_type_selector,
        annotation_type_selector,
        highlight_tags_selector,
        highlight_lines_selector,
        show_annotations_checkbox,
        show_comet_checkbox,
        save_path_text,
        plot_button,
        plot_output
    ])
    
    return visualization_interface, update_visualization_interface

def create_sentence_analysis_interface():
    """Create interface for individual sentence analysis"""
    
    sentence_dataset_selector = widgets.Dropdown(
        options=[],
        description="Dataset:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px')
    )
    
    sentence_selector = widgets.Dropdown(
        options=[],
        description="Sentence:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='600px')
    )
    
    sentence_annotation_type = widgets.Dropdown(
        options=[('None', None), ('Part-of-Speech', 'pos'), ('Dependency', 'dep')],
        value=None,
        description="Annotations:",
        style={'description_width': 'initial'}
    )
    
    sentence_highlight_tags = widgets.SelectMultiple(
        options=[],
        description="Highlight:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px', height='80px')
    )
    
    sentence_highlight_lines = widgets.SelectMultiple(
        options=[
            ('Conditional Entropy', 'conditional_entropy'),
            ('Surprisal', 'surprisal'),
            ('Token Probability', 'token_probability'),
            ('Top-k Entropy', 'topk_entropy')
        ],
        value=['conditional_entropy'],
        description="Lines:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%', height='80px')
    )

    sentence_show_annotations = widgets.Checkbox(
        value=True,
        description="Show Annotations"
    )
    
    sentence_show_comet = widgets.Checkbox(
        value=True,
        description="Show COMET Score"
    )
    
    sentence_plot_button = widgets.Button(
        description="Analyze Sentence",
        button_style='warning',
        layout=widgets.Layout(width='150px')
    )
    
    sentence_output = widgets.Output()
    
    def update_sentence_selector(change=None):
        """Update sentence selector based on selected dataset"""
        dataset_key = sentence_dataset_selector.value
        if dataset_key and dataset_key in processed_datasets:
            data = processed_datasets[dataset_key]
            sentence_options = []
            
            for key, value in data.items():
                source_info = value['source_info']
                analysis_type = source_info.get('analysis_type', 'unknown')
                comet_info = ""
                if 'comet_score' in source_info:
                    comet_info = f" (COMET: {source_info['comet_score']:.3f})"
                
                display_text = key[:80] + "..." if len(key) > 80 else key
                sentence_options.append((f"[{analysis_type.upper()}] {display_text}{comet_info}", key))
            
            sentence_selector.options = sentence_options
        else:
            sentence_selector.options = []
    
    def update_sentence_highlight_tags(change):
        """Update highlight tags for sentence analysis"""
        annotation_type = change['new']
        if annotation_type:
            available_tags = get_available_highlight_tags()
            sentence_highlight_tags.options = [(tag, tag) for tag in available_tags[annotation_type]]
        else:
            sentence_highlight_tags.options = []
    
    sentence_dataset_selector.observe(update_sentence_selector, names='value')
    sentence_annotation_type.observe(update_sentence_highlight_tags, names='value')
    
    def on_sentence_plot_click(b):
        with sentence_output:
            clear_output()

            dataset_key = sentence_dataset_selector.value
            sentence_key = sentence_selector.value

            if not dataset_key or not sentence_key:
                print("❌ Please select both dataset and sentence")
                return

            if dataset_key not in processed_datasets:
                print(f"❌ Dataset {dataset_key} not found")
                return

            if sentence_key not in processed_datasets[dataset_key]:
                print(f"❌ Sentence not found in dataset")
                return

            print(f"🔍 Analyzing sentence from {dataset_key}...")
            print(f"Sentence: {sentence_key[:100]}...")

            try:
                source_item = processed_datasets[dataset_key][sentence_key]
                source_info = source_item['source_info']
                tokens = source_info.get('tokens', [])

                if not tokens:
                    print("❌ No tokens found in sentence data")
                    return

                # Clean & merge stats
                cleaned_info = clean_entropy_data(source_info, tokens)
                cleaned_source_info = source_info.copy()
                cleaned_source_info.update(cleaned_info)

                # Plot using the real field names
                plot_conditional_entropy_metrics(
                    { sentence_key: {'source_info': cleaned_source_info} },
                    sentence_key=sentence_key,
                    annotation_type=sentence_annotation_type.value,
                    show_annotations=sentence_show_annotations.value,
                    highlight_tags=list(sentence_highlight_tags.value) if sentence_highlight_tags.value else None,
                    title="Individual Sentence Analysis",
                    highlight_lines=list(sentence_highlight_lines.value) if sentence_highlight_lines.value else None,
                    show_comet=sentence_show_comet.value,
                    entropy_type=source_info.get('entropy_type', 'softmax')
                )

                # Print enhanced statistics
                print(f"\n📊 Enhanced Statistics:")
                print(f"   Tokens: {len(tokens)}")
                
                metrics_to_analyze = [
                    ('conditional_entropy_bits', 'Entropy'),
                    ('surprisal_bits', 'Surprisal'),
                    ('token_probability', 'Token Probability'),
                    ('topk_entropy_bits', 'Top-k Entropy')
                ]
                
                for metric_key, metric_name in metrics_to_analyze:
                    if metric_key in cleaned_source_info:
                        values = [cleaned_source_info[metric_key].get(t, 0) for t in tokens]
                        stats = safe_statistics(values, metric_name)
                        if stats['valid_count'] > 0:
                            max_idx, max_token = safe_argmax(values, tokens)
                            print(f"\n   {metric_name}:")
                            print(f"     Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
                            print(f"     Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                            print(f"     Max: {stats['max']:.3f} at '{max_token}'")
                            if stats.get('invalid_count', 0) > 0:
                                print(f"     ⚠️ Invalid values: {stats['invalid_count']}/{stats['count']}")
                        else:
                            print(f"   ⚠️ {metric_name}: {stats.get('warning', 'No valid data')}")

                # Add COMET score if available
                if 'comet_score' in source_info:
                    comet_score = source_info['comet_score']
                    if np.isfinite(comet_score):
                        print(f"\n   COMET Score: {comet_score:.3f}")
                    else:
                        print(f"\n   ⚠️ COMET Score: Invalid ({comet_score})")
                
                # Data quality report
                print(f"\n🔍 Data Quality Report:")
                total_values = len(tokens) * len(metrics_to_analyze)
                valid_values = sum([
                    len([v for v in [cleaned_source_info.get(m[0], {}).get(t, 0) for t in tokens] if np.isfinite(v)])
                    for m in metrics_to_analyze if m[0] in cleaned_source_info
                ])
                
                quality_percentage = (valid_values / total_values * 100) if total_values > 0 else 0
                print(f"   Valid data points: {valid_values}/{total_values} ({quality_percentage:.1f}%)")
                
                if quality_percentage < 90:
                    print("   ⚠️ Data quality warning: Some metrics contain invalid values")
                
            except Exception as e:
                print(f"❌ Error analyzing sentence: {e}")
                traceback.print_exc()
    
    sentence_plot_button.on_click(on_sentence_plot_click)
    
    def update_sentence_analysis_interface():
        """Update sentence analysis interface with available datasets"""
        available_datasets = list(processed_datasets.keys())
        sentence_dataset_selector.options = available_datasets
        if available_datasets:
            sentence_dataset_selector.value = available_datasets[0]
            update_sentence_selector()
        print(f"🔄 Updated sentence analysis interface with {len(available_datasets)} processed datasets")
    
    sentence_interface = widgets.VBox([
        widgets.HTML("<h3>🔍 Individual Sentence Analysis (Enhanced Statistics)</h3>"),
        sentence_dataset_selector,
        sentence_selector,
        sentence_annotation_type,
        sentence_highlight_tags,
        sentence_highlight_lines,
        sentence_show_annotations,
        sentence_show_comet,
        sentence_plot_button,
        sentence_output
    ])
    
    return sentence_interface, update_sentence_analysis_interface

def create_summary_interface():
    """Create interface for summary and export"""
    
    summary_button = widgets.Button(
        description="Generate Summary",
        button_style='success',
        layout=widgets.Layout(width='150px')
    )
    
    export_button = widgets.Button(
        description="Export Data",
        button_style='info',
        layout=widgets.Layout(width='150px')
    )
    
    summary_output = widgets.Output()
    
    def on_summary_click(b):
        with summary_output:
            clear_output()
            if not processed_datasets:
                print("❌ No processed datasets available")
                return
            
            print("📊 Dataset Summary (with Data Quality Analysis)")
            print("="*90)
            
            try:
                stats = get_dataset_statistics(processed_datasets)
                
                print(f"{'Dataset':<25} {'Sources':<8} {'Targets':<8} {'Avg S-Ent':<10} {'Avg T-Ent':<10} {'Avg COMET':<10} {'Quality':<8}")
                print("-" * 90)
                
                for dataset_key, stat in stats.items():
                    source_ent = f"{stat.get('source_avg_entropy', 0):.3f}" if 'source_avg_entropy' in stat else "N/A"
                    target_ent = f"{stat.get('target_avg_entropy', 0):.3f}" if 'target_avg_entropy' in stat else "N/A"
                    comet = f"{stat.get('avg_comet', 0):.3f}" if 'avg_comet' in stat else "N/A"
                    
                    # Calculate data quality score
                    quality_score = "Good"
                    if stat.get('source_avg_entropy', 0) == 0 and stat.get('target_avg_entropy', 0) == 0:
                        quality_score = "Poor"
                    elif stat.get('source_avg_entropy', 0) == 0 or stat.get('target_avg_entropy', 0) == 0:
                        quality_score = "Fair"
                    
                    print(f"{dataset_key:<25} {stat['source_count']:<8} {stat['target_count']:<8} {source_ent:<10} {target_ent:<10} {comet:<10} {quality_score:<8}")
            
            except Exception as e:
                print(f"❌ Error generating summary: {e}")
                traceback.print_exc()
    
    def on_export_click(b):
        with summary_output:
            try:
                filename = "exported_entropy_data.json"
                
                # Clean data before export
                cleaned_export_data = {}
                for dataset_key, dataset_data in processed_datasets.items():
                    cleaned_dataset = {}
                    for sentence_key, sentence_data in dataset_data.items():
                        source_info = sentence_data['source_info']
                        tokens = source_info.get('tokens', [])
                        
                        if tokens:
                            cleaned_info = clean_entropy_data(source_info, tokens)
                            cleaned_source_info = source_info.copy()
                            cleaned_source_info.update(cleaned_info)
                            cleaned_dataset[sentence_key] = {'source_info': cleaned_source_info}
                        else:
                            cleaned_dataset[sentence_key] = sentence_data
                    
                    cleaned_export_data[dataset_key] = cleaned_dataset
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_export_data, f, indent=2, ensure_ascii=False)
                print(f"✅ Cleaned data exported to {filename}")
                print("📋 Note: Infinite and NaN values were replaced with 0 for compatibility")
                
            except Exception as e:
                print(f"❌ Export failed: {e}")
                traceback.print_exc()
    
    summary_button.on_click(on_summary_click)
    export_button.on_click(on_export_click)
    
    summary_interface = widgets.VBox([
        widgets.HTML("<h3>📋 Summary & Export (Enhanced)</h3>"),
        widgets.HBox([summary_button, export_button]),
        summary_output
    ])
    
    return summary_interface

def create_main_interface():
    """Create the main tabbed interface"""
    global update_dataset_selector_ref, update_visualization_interface_ref, update_sentence_analysis_interface_ref
    
    # Create all components
    upload_interface = create_file_upload_interface()
    processing_interface, update_dataset_selector_ref = create_processing_interface()
    visualization_interface, update_visualization_interface_ref = create_visualization_interface()
    sentence_interface, update_sentence_analysis_interface_ref = create_sentence_analysis_interface()
    summary_interface = create_summary_interface()
    
    # Create tabs
    tab = widgets.Tab()
    tab.children = [
        upload_interface,
        processing_interface,
        visualization_interface,
        sentence_interface,
        summary_interface
    ]
    
    tab.set_title(0, "📁 Upload")
    tab.set_title(1, "⚙️ Process")
    tab.set_title(2, "📈 Visualize")
    tab.set_title(3, "🔍 Analyze")
    tab.set_title(4, "📋 Summary")
    
    # Header
    header = widgets.HTML("""
    <div style="background-color: #f0f8ff; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
        <h2>🎯 mBART Entropy Analysis Dashboard (Fixed)</h2>
        <p><strong>Features:</strong> Softmax & Entmax entropies • COMET scores • Linguistic annotations • Data quality checks • Robust statistics</p>
        <p><strong>Instructions:</strong> Upload/Load data → Process → Visualize → Analyze sentences → Export</p>
        <p><strong>Fixed:</strong> Handles infinite/NaN values • Enhanced error reporting • Data quality analysis • Proper syntax</p>
    </div>
    """)
    
    main_interface = widgets.VBox([header, tab])
    
    return main_interface, (update_dataset_selector_ref, update_visualization_interface_ref, update_sentence_analysis_interface_ref)

def update_all_interfaces():
    """Update all interfaces when data changes"""
    try:
        print("🔄 Updating all interfaces...")
        if update_dataset_selector_ref:
            update_dataset_selector_ref()
        if update_visualization_interface_ref:
            update_visualization_interface_ref()
        if update_sentence_analysis_interface_ref:
            update_sentence_analysis_interface_ref()
        print("✅ All interfaces updated successfully")
    except Exception as e:
        print(f"❌ Error updating interfaces: {e}")
        traceback.print_exc()

def load_example_data():
    """Load example data for testing"""
    global loaded_datasets
    
    example_data = {
        "Example Polish sentence.": {
            "source_info": {
                "tokens": ["▁Example", "▁Polish", "▁sentence", "."],
                "conditional_entropy_bits": {
                    "▁Example": 2.45, "▁Polish": 3.21, "▁sentence": 1.67, ".": 0.12
                },
                "surprisal_bits": {
                    "▁Example": 3.45, "▁Polish": 4.21, "▁sentence": 2.67, ".": 1.12
                },
                "full_entropy_bits": {
                    "▁Example": 2.35, "▁Polish": 3.11, "▁sentence": 1.57, ".": 0.10
                },
                "topk_entropy_k5_bits": {
                    "▁Example": 1.45, "▁Polish": 2.21, "▁sentence": 1.07, ".": 0.08
                },
                "pos": ["NOUN", "ADJ", "NOUN", "PUNCT"],
                "dep": ["nsubj", "amod", "compound", "punct"]
            },
            "candidates": [
                {
                    "sentence": "Przykład polskiego zdania.",
                    "tokens": ["▁Przy", "kł", "ad", "▁pol", "skiego", "▁z", "dan", "ia", "."],
                    "trgt_token_entropy_bits": {
                        "▁Przy": 2.1, "kł": 1.2, "ad": 0.8, "▁pol": 2.8, 
                        "skiego": 1.9, "▁z": 1.5, "dan": 1.3, "ia": 0.9, ".": 0.1
                    },
                    "trgt_token_surprisal_bits": {
                        "▁Przy": 3.1, "kł": 2.2, "ad": 1.8, "▁pol": 3.8,
                        "skiego": 2.9, "▁z": 2.5, "dan": 2.3, "ia": 1.9, ".": 1.1
                    },
                    "trgt_token_topk_entropy_k5_bits": {
                        "▁Przy": 1.1, "kł": 0.9, "ad": 0.6, "▁pol": 1.8,
                        "skiego": 1.2, "▁z": 1.0, "dan": 0.8, "ia": 0.7, ".": 0.05
                    },
                    "comet_score": 0.857,
                    "beam_score": -2.45,
                    "beam_idx": 0,
                    "pos": ["NOUN", "NOUN", "NOUN", "ADJ", "ADJ", "NOUN", "NOUN", "NOUN", "PUNCT"],
                    "dep": ["nsubj", "compound", "compound", "amod", "amod", "compound", "compound", "compound", "punct"]
                }
            ]
        }
    }
    
    loaded_datasets["example_data"] = example_data
    print("✅ Example data loaded!")
    print("📋 This example includes proper field names that will work with processing")
    print("💡 Next steps:")
    print("   1. Go to 'Process' tab and select 'example_data'")
    print("   2. Click 'Process All Selected' to generate variants")
    print("   3. Use 'Visualize' and 'Analyze' tabs to explore the data")
    update_all_interfaces()

# Debug functions
def debug_loaded_datasets():
    """Debug function to check loaded datasets"""
    print(f"🔍 Loaded datasets: {len(loaded_datasets)}")
    for name, data in loaded_datasets.items():
        print(f"  - {name}: {len(data)} entries")
        if data:
            sample_key = list(data.keys())[0]
            print(f"    Sample: {sample_key[:50]}...")

def debug_processed_datasets():
    """Debug function to check processed datasets"""
    print(f"🔍 Processed datasets: {len(processed_datasets)}")
    for name, data in processed_datasets.items():
        print(f"  - {name}: {len(data)} entries")

def debug_data_quality():
    """Debug function to check data quality issues"""
    print("🔍 DATA QUALITY ANALYSIS")
    print("=" * 40)
    
    if not processed_datasets:
        print("❌ No processed datasets to analyze")
        return
    
    total_issues = 0
    
    for dataset_key, dataset_data in processed_datasets.items():
        print(f"\n📊 Dataset: {dataset_key}")
        dataset_issues = 0
        
        for sentence_key, sentence_data in dataset_data.items():
            source_info = sentence_data['source_info']
            tokens = source_info.get('tokens', [])
            
            if not tokens:
                continue
            
            sentence_issues = 0
            metrics = ['conditional_entropy_bits', 'surprisal_bits', 'token_probability', 'topk_entropy_bits']
            
            for metric in metrics:
                if metric in source_info:
                    metric_data = source_info[metric]
                    for token in tokens:
                        if token in metric_data:
                            value = metric_data[token]
                            if not np.isfinite(value):
                                sentence_issues += 1
                                dataset_issues += 1
            
            if sentence_issues > 0:
                print(f"  ⚠️ {sentence_key[:50]}...: {sentence_issues} invalid values")
        
        if dataset_issues == 0:
            print(f"  ✅ No data quality issues found")
        else:
            print(f"  ⚠️ Total issues: {dataset_issues}")
        
        total_issues += dataset_issues
    
    print(f"\n📋 SUMMARY:")
    print(f"Total datasets analyzed: {len(processed_datasets)}")
    print(f"Total data quality issues: {total_issues}")
    
    if total_issues == 0:
        print("✅ All data appears to be of good quality!")
    else:
        print("⚠️ Some data quality issues detected. Use the enhanced analysis features for better handling.")

# Make functions available for external use
__all__ = [
    'create_main_interface',
    'update_all_interfaces', 
    'load_example_data',
    'loaded_datasets', 
    'processed_datasets',
    'debug_loaded_datasets',
    'debug_processed_datasets',
    'debug_data_quality',
    'safe_statistics',
    'clean_entropy_data'
]

if __name__ == "__main__":
    print("📊 Widget Interface module loaded successfully!")
    print("\n💡 Available functions:")
    print("- create_main_interface(): Create the main dashboard")
    print("- load_example_data(): Load example data for testing")
    print("- debug_loaded_datasets(): Inspect loaded datasets")
    print("- debug_processed_datasets(): Inspect processed datasets")
    print("- debug_data_quality(): Analyze data quality issues")
    print("\n🚀 To start:")
    print("1. Run: main_interface, update_funcs = create_main_interface()")
    print("2. Run: display(main_interface)")
    print("3. Load data and start analyzing!")