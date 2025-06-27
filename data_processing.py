"""data_processing.py - Data processing functions with DYNAMIC field detection (FIXED FOR NESTED STRUCTURE)"""

import json
import numpy as np
import os
from typing import Dict, Any, List


def load_preloaded_datasets() -> Dict[str, Any]:
    """Load preloaded datasets from comprehensive-with-comet JSONs - FIXED for nested structure"""
    datasets: Dict[str, Any] = {}

    dataset_names = ["svo", "svo_corpus", "ovs", "ovs_corpus",  "szwedek_pairs"]
    dataset_paths = {
        name: f"/content/drive/MyDrive/NMT/corpus_annotation/{name}/{name}_comprehensive_with_comet.json"
        for name in dataset_names
    }

    for dataset_name, path in dataset_paths.items():
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                
                # FIXED: Handle nested structure with analysis types
                if isinstance(raw_data, dict) and any(key in raw_data for key in ['full', 'conditional', 'manual_autoregressive']):
                    print(f"📊 {dataset_name}: Found nested structure with analysis types: {list(raw_data.keys())}")
                    
                    # Flatten the nested structure by analysis type
                    for analysis_type, analysis_data in raw_data.items():
                        if isinstance(analysis_data, dict) and analysis_data:
                            # Create separate dataset for each analysis type
                            flattened_name = f"{dataset_name}_{analysis_type}"
                            datasets[flattened_name] = analysis_data
                            print(f"✅ Loaded {flattened_name}: {len(analysis_data)} entries")
                else:
                    # Original flat structure
                    datasets[dataset_name] = raw_data
                    print(f"✅ Loaded {dataset_name}: {len(raw_data)} entries")
            else:
                print(f"⚠️ Path not found: {path}")
        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {e}")

    return datasets


def determine_model_type(dataset_key: str) -> str:
    key = dataset_key.lower()
    if 'seq2seq' in key:
        return 'seq2seq'
    elif 'conditional' in key:
        return 'conditional'
    elif 'manual_autoregressive' in key:
        return 'autoregressive'
    elif 'full' in key:
        return 'full'
    else:
        return 'unknown'


def detect_dataset_structure(sample_data: Dict[str, Any]) -> Dict[str, Any]:
    info = {
        'has_conditional_entropy': any(k for k in sample_data.keys() if 'conditional_entropy' in k),
        'has_entmax_conditional_entropy': any(k for k in sample_data.keys() if 'entmax_conditional_entropy' in k),
        'available_softmax_fields': [],
        'available_entmax_fields': [],
        'available_target_softmax_fields': [],
        'available_target_entmax_fields': [],
        'model_type': sample_data.get('model_type', 'unknown'),
        'detected_dataset_type': 'unknown',
        'field_prefix': None  # To detect the actual prefix used
    }
    
    # FIXED: Detect field prefixes from actual data
    field_prefixes = []
    for key in sample_data.keys():
        if '_entropy_bits' in key or '_surprisal_bits' in key:
            # Extract prefix (e.g., 'full_source' from 'full_source_entropy_bits')
            prefix = key.split('_entropy_bits')[0].split('_surprisal_bits')[0]
            if prefix and prefix not in field_prefixes:
                field_prefixes.append(prefix)
    
    info['detected_prefixes'] = field_prefixes
    if field_prefixes:
        info['field_prefix'] = field_prefixes[0]  # Use first detected prefix
    
    # Determine type based on detected fields
    if any('conditional' in p for p in field_prefixes):
        info['detected_dataset_type'] = 'conditional'
    elif any('full' in p for p in field_prefixes):
        info['detected_dataset_type'] = 'full'
    else:
        info['detected_dataset_type'] = 'seq2seq'

    # FIXED: Source field detection with actual prefixes
    src_field_patterns = [
        'entropy_bits', 'surprisal_bits', 'topk_entropy_k5_bits', 'topk_mass_k5',
        'entmax_entropy_bits', 'entmax_surprisal_bits', 'entmax_topk_entropy_k5_bits', 
        'entmax_topk_mass_k5', 'entmax_branching_choices',
        'bigram_source_entropy_bits', 'bigram_source_surprisal_bits',
        'bigram_source_topk_entropy_k5_bits', 'bigram_source_topk_mass_k5',
        'bigram_source_entmax_entropy_bits', 'bigram_source_entmax_surprisal_bits',
        'bigram_source_entmax_topk_entropy_k5_bits', 'bigram_source_entmax_topk_mass_k5',
        'bigram_source_entmax_branching_choices'
    ]
    
    for pattern in src_field_patterns:
        for prefix in field_prefixes:
            full_field = f"{prefix}_{pattern}" if prefix else pattern
            if full_field in sample_data:
                if 'entmax' in full_field:
                    info['available_entmax_fields'].append(full_field)
                else:
                    info['available_softmax_fields'].append(full_field)

    return info


def get_entropy_fields_dynamic(entropy_type: str, analysis_type: str, sample_data: Dict[str, Any] = None) -> Dict[str, str]:
    """FIXED: Dynamic field detection that works with PROCESSED data structure"""
    if sample_data is None:
        # Fallback for static usage
        return {
            'entropy': 'conditional_entropy_bits',
            'conditional_entropy': 'conditional_entropy_bits',
            'surprisal': 'surprisal_bits',
            'topk_entropy': 'topk_entropy_bits',
            'topk_mass': 'topk_mass_bits'
        }

    fields: Dict[str, str] = {}
    
    print(f"🔍 DEBUG: All keys in sample_data: {list(sample_data.keys())}")
    
    # FIXED: Work with the processed field names that are actually present
    available_fields = list(sample_data.keys())
    
    # Look for processed field patterns (these are the simplified names after processing)
    processed_patterns = {
        'entropy': ['conditional_entropy_bits', 'entropy_bits', 'full_entropy_bits'],
        'surprisal': ['surprisal_bits', 'conditional_surprisal_bits', 'full_surprisal_bits'],
        'topk_entropy': ['topk_entropy_bits', 'topk_entropy_k5_bits', 'conditional_topk_entropy_bits'],
        'topk_mass': ['topk_mass_bits', 'topk_mass_k5', 'conditional_topk_mass_bits'],
        'bigram_entropy': ['bigram_entropy_bits', 'conditional_bigram_entropy_bits', 'full_bigram_entropy_bits'],
        'bigram_surprisal': ['bigram_surprisal_bits', 'conditional_bigram_surprisal_bits', 'full_bigram_surprisal_bits'],
        'branching_choices': ['branching_choices_bits', 'entmax_branching_choices', 'conditional_branching_choices_bits']
    }
    
    # Special handling for entmax fields
    if entropy_type == 'entmax':
        entmax_patterns = {
            'entropy': ['entmax_entropy_bits', 'conditional_entmax_entropy_bits', 'entmax_conditional_entropy_bits'],
            'surprisal': ['entmax_surprisal_bits', 'conditional_entmax_surprisal_bits'],
            'topk_entropy': ['entmax_topk_entropy_bits', 'conditional_entmax_topk_entropy_bits'],
            'topk_mass': ['entmax_topk_mass_bits', 'conditional_entmax_topk_mass_bits'],
            'branching_choices': ['entmax_branching_choices_bits', 'branching_choices_bits']
        }
        # Merge entmax patterns with processed patterns
        for key in entmax_patterns:
            if key in processed_patterns:
                processed_patterns[key] = entmax_patterns[key] + processed_patterns[key]
            else:
                processed_patterns[key] = entmax_patterns[key]
    
    print(f"🔍 Looking for patterns based on entropy_type: {entropy_type}")
    
    # Find matching fields in the processed data
    for metric_name, candidates in processed_patterns.items():
        found = False
        for candidate in candidates:
            if candidate in available_fields:
                fields[metric_name] = candidate
                print(f"✅ Using {metric_name} field: {candidate}")
                found = True
                break
            else:
                print(f"🔍 Trying {metric_name}: {candidate} - not found")
        
        if not found:
            print(f"⚠️ No field found for {metric_name}")
    
    # Ensure entropy and conditional_entropy point to the same field
    if 'entropy' in fields:
        fields['conditional_entropy'] = fields['entropy']
    
    print(f"🎯 Final field mapping for {entropy_type} {analysis_type}: {fields}")
    return fields


def safe_get_metric_data(source_data: Dict[str, Any], field_name: str, tokens: List[str]) -> Dict[str, float]:
    """Handle inf/nan and missing values"""
    cleaned: Dict[str, float] = {}
    raw = source_data.get(field_name, {})
    if isinstance(raw, dict):
        for t in tokens:
            val = raw.get(t, 0.0)
            cleaned[t] = float(val) if np.isfinite(val) else 0.0
    elif isinstance(raw, list):
        for i, t in enumerate(tokens):
            val = raw[i] if i < len(raw) else 0.0
            cleaned[t] = float(val) if np.isfinite(val) else 0.0
    else:
        cleaned = {t: 0.0 for t in tokens}
    return cleaned


def process_source_data(source_info: Dict[str, Any], entropy_type: str, model_type: str) -> Dict[str, Any]:
    tokens = source_info.get('tokens', [])
    if not tokens:
        return {}
    
    structure = detect_dataset_structure(source_info)
    fields = get_entropy_fields_dynamic(entropy_type, 'source', source_info)
    
    out = {
        'tokens': tokens,
        'analysis_type': 'source',
        'entropy_type': entropy_type,
        'model_type': model_type,
        'detected_dataset_type': structure['detected_dataset_type'],
        'detected_prefixes': structure.get('detected_prefixes', [])
    }
    
    for m, f in fields.items():
        metric_data = safe_get_metric_data(source_info, f, tokens)
        out[f'{m}_bits'] = metric_data
        
        # Debug output
        non_zero = sum(1 for v in metric_data.values() if v > 0)
        if non_zero > 0:
            max_val = max(metric_data.values())
            print(f"   ✅ {m}_bits ({f}): {non_zero}/{len(tokens)} non-zero, max={max_val:.3f}")
    
    # token_probability fallback
    if 'token_probability' not in out:
        sp = out.get('surprisal_bits', {})
        if any(sp.values()):
            out['token_probability'] = {t: np.exp(-v) if v > 0 else 0.01 for t, v in sp.items()}
        else:
            out['token_probability'] = {t: 0.01 for t in tokens}
    
    # annotations
    for tag in ['pos', 'dep', 'spacy_tokens']:
        if tag in source_info:
            out[tag] = source_info[tag]
    
    return out


def process_target_data(candidate: Dict[str, Any], entropy_type: str, model_type: str, source_sentence: str) -> Dict[str, Any]:
    tokens = candidate.get('tokens') or candidate.get('sentence', '').split()
    fields = get_entropy_fields_dynamic(entropy_type, 'target', candidate)
    
    out = {
        'tokens': tokens, 
        'analysis_type': 'target', 
        'entropy_type': entropy_type, 
        'model_type': model_type, 
        'source_sentence': source_sentence
    }
    
    for m, f in fields.items():
        metric_data = safe_get_metric_data(candidate, f, tokens)
        out[f'{m}_bits'] = metric_data
        
        # Debug output
        non_zero = sum(1 for v in metric_data.values() if v > 0)
        if non_zero > 0:
            max_val = max(metric_data.values())
            print(f"   ✅ TARGET {m}_bits ({f}): {non_zero}/{len(tokens)} non-zero, max={max_val:.3f}")
    
    if 'token_probability' not in out:
        sp = out.get('surprisal_bits', {})
        if any(sp.values()):
            out['token_probability'] = {t: np.exp(-v) if v > 0 else 0.01 for t, v in sp.items()}
        else:
            out['token_probability'] = {t: 0.01 for t in tokens}
    
    for score in ['comet_score', 'beam_score', 'beam_idx', 'candidate_idx']:
        if score in candidate:
            out[score] = candidate[score]
    
    for tag in ['pos', 'dep', 'spacy_tokens']:
        if tag in candidate:
            out[tag] = candidate[tag]
    
    return out


def process_single_dataset(results: Dict[str, Any], name: str, entropy_type: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    model_type = determine_model_type(name)
    
    print(f"\n🔄 Processing {name} with {entropy_type} entropy...")
    
    for i, (src, data) in enumerate(results.items()):
        si = data.get('source_info', {})
        if si and si.get('tokens'):
            processed_src = process_source_data(si, entropy_type, model_type)
            if processed_src:
                key = f"SOURCE_{i+1}_{entropy_type.upper()}_{model_type.upper()}: {src}"
                out[key] = {'source_info': processed_src}
                print(f"✅ Processed source: {src[:50]}...")
        
        for j, cand in enumerate(data.get('candidates', [])):
            if cand and cand.get('tokens'):
                proc = process_target_data(cand, entropy_type, model_type, src)
                if proc:
                    sent = cand.get('sentence', f'Cand{j+1}')
                    key = f"TARGET_{i+1}_{entropy_type.upper()}_{model_type.upper()}_C{j+1}: {sent}"
                    if 'comet_score' in cand:
                        key += f" (COMET: {cand['comet_score']:.3f})"
                    out[key] = {'source_info': proc}
                    print(f"✅ Processed target: {sent[:50]}...")
    
    print(f"📊 {name}_{entropy_type}: Generated {len(out)} processed entries")
    return out


def process_all_datasets(loaded: Dict[str, Any]) -> Dict[str, Any]:
    processed: Dict[str, Any] = {}
    for name, data in loaded.items():
        for et in ['softmax', 'entmax']:
            pd = process_single_dataset(data, name, et)
            if pd:
                processed[f"{name}_{et}"] = pd
    return processed


def filter_data_by_type(data: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
    return {k: v for k, v in data.items() if v.get('source_info', {}).get('analysis_type') == analysis_type}


def get_dataset_statistics(processed: Dict[str, Any]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for dk, data in processed.items():
        src = filter_data_by_type(data, 'source')
        tgt = filter_data_by_type(data, 'target')
        sents = [d['source_info'] for d in src.values()]
        tents = [d['source_info'] for d in tgt.values()]
        st, sv, sk = [], [], []
        for info in sents:
            ts = info.get('conditional_entropy_bits', {})
            sp = info.get('surprisal_bits', {})
            tk = info.get('topk_entropy_bits', {})
            st += [v for v in ts.values() if v > 0]
            sv += [v for v in sp.values() if v > 0]
            sk += [v for v in tk.values() if v > 0]
        ds = {'source_count': len(sents), 'target_count': len(tents)}
        if st:
            ds.update({'source_avg_entropy': np.mean(st), 'source_std_entropy': np.std(st)})
        if sv:
            ds.update({'source_avg_surprisal': np.mean(sv), 'source_std_surprisal': np.std(sv)})
        if sk:
            ds.update({'source_avg_topk': np.mean(sk)})
        ct, cv, ck, comet, beam = [], [], [], [], []
        for info in tents:
            ts = info.get('conditional_entropy_bits', {})
            sp = info.get('surprisal_bits', {})
            tk = info.get('topk_entropy_bits', {})
            ct += [v for v in ts.values() if v > 0]
            cv += [v for v in sp.values() if v > 0]
            ck += [v for v in tk.values() if v > 0]
            if 'comet_score' in info:
                comet.append(info['comet_score'])
            if 'beam_score' in info:
                beam.append(info['beam_score'])
        if ct:
            ds.update({'target_avg_entropy': np.mean(ct), 'target_std_entropy': np.std(ct)})
        if cv:
            ds.update({'target_avg_surprisal': np.mean(cv), 'target_std_surprisal': np.std(cv)})
        if ck:
            ds.update({'target_avg_topk': np.mean(ck)})
        if comet:
            ds.update({'avg_comet': np.mean(comet), 'std_comet': np.std(comet)})
        if beam:
            ds.update({'avg_beam_score': np.mean(beam), 'std_beam_score': np.std(beam)})
        stats[dk] = ds
    return stats


def get_available_highlight_tags() -> Dict[str, List[str]]:
    return {
        'pos': ["NOUN", "VERB", "DET", "ADJ", "PUNCT", "PRON", "ADP", "NUM", "CONJ", "ADV", "PROPN"],
        'dep': ["ROOT", "nsubj", "obj", "dobj", "det", "amod", "nmod", "compound", "case", "punct", "advmod", "aux"]
    }


def validate_dataset_structure(data: Any) -> (bool, str):
    if not isinstance(data, dict):
        return False, "Dataset must be a dict"
    
    # Check if it's the nested structure with analysis types
    if any(key in data for key in ['full', 'conditional', 'manual_autoregressive']):
        # Validate nested structure
        for analysis_type, analysis_data in data.items():
            if not isinstance(analysis_data, dict):
                return False, f"Analysis type {analysis_type} must be a dict"
            for sk, sd in analysis_data.items():
                if not isinstance(sd, dict):
                    return False, f"Sentence {sk} in {analysis_type} not a dict"
                if 'source_info' not in sd or 'tokens' not in sd['source_info']:
                    return False, f"Missing source_info/tokens in {sk} ({analysis_type})"
        return True, "OK - Nested structure"
    
    # Validate flat structure
    for sk, sd in data.items():
        if not isinstance(sd, dict):
            return False, f"Sentence {sk} not a dict"
        if 'source_info' not in sd or 'tokens' not in sd['source_info']:
            return False, f"Missing source_info/tokens in {sk}"
        if 'candidates' in sd:
            if not isinstance(sd['candidates'], list):
                return False, f"Candidates not a list in {sk}"
            for c in sd['candidates']:
                if 'sentence' not in c or 'tokens' not in c:
                    return False, f"Missing fields in candidate of {sk}"
    return True, "OK - Flat structure"


def get_entropy_fields(entropy_type: str, analysis_type: str) -> Dict[str, str]:
    return get_entropy_fields_dynamic(entropy_type, analysis_type, None)


if __name__ == "__main__":
    print("Data Processing module loaded with nested structure support.")