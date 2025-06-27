# Enhanced Translation Entropy Analysis Report

**Analysis Date:** 2025-06-27 07:59:55

**Dataset:** svo

## Configuration

- **model_name:** facebook/mbart-large-50-many-to-many-mmt
- **src_lang:** pl_PL
- **tgt_lang:** en_XX
- **num_beams:** 10
- **num_return_sequences:** 10
- **output_dir:** /content/drive/MyDrive/NMT/corpus_annotation/svo
- **run_comet:** True
- **run_individual_analyses:** True

## Dataset Information

- **Number of sentences:** 59
- **Source language:** pl_PL
- **Target language:** en_XX
- **Translation candidates per sentence:** 10

## Analysis Methods

### FULL
FULL Analysis: Natural seq2seq autoregressive generation
            - Uses seq2seq_model (AutoModelForSeq2SeqLM) for autoregressive source analysis and target generation
            - Uses conditional_model (MBartForConditionalGeneration) for bigram analysis
            - Formula: P(token | autoregressive_context) via generation scores
            - Source: Autoregressive generation + bigram decoder context
            - Target: Translation via beam search generation + bigram conditional analysis
            - Models: seq2seq_model (primary) + conditional_model (bigrams)

### CONDITIONAL
CONDITIONAL Analysis: Conditional probability reconstruction
            - Uses conditional_model (MBartForConditionalGeneration) exclusively
            - Formula: P(token | prefix_context) via conditional forward passes
            - Source: Step-by-step conditional reconstruction + bigram decoder context
            - Target: Conditional reconstruction with source context + bigram analysis
            - Model: conditional_model only

### MANUAL_AUTOREGRESSIVE
MANUAL_AUTOREGRESSIVE Analysis: Manual masking with autoregressive context
            - Uses both seq2seq_model and conditional_model for comprehensive analysis
            - Formula: P(token | all_previous_tokens) via masking at each position
            - Source: Autoregressive context with manual masking (both models) + bigram decoder masking
            - Target: Same approach applied to translation candidates
            - Models: Both seq2seq_model and conditional_model
            - Method: Places MASK token at target position, uses all previous tokens as context
            - Keys: manual_autoregressive_seq2seq_*, manual_autoregressive_conditional_*, manual_bigram_decoder_*

### MANUAL_BIGRAM_DECODER
MANUAL_BIGRAM_DECODER Analysis: Manual masking with bigram decoder context
            - Uses conditional_model (MBartForConditionalGeneration) for bigram context
            - Formula: P(token | previous_token) via decoder masking
            - Source: Bigram context with decoder masking
            - Target: Same approach applied to translation candidates
            - Model: conditional_model only
            - Method: Uses only immediate previous token as context, masks current position
            - Keys: manual_bigram_decoder_*

## Results Summary

### FULL Analysis
- Sentences processed: 57
- Translation candidates: 570
- Average candidates per sentence: 10.0
- Entropy metrics per token: 18

### CONDITIONAL Analysis
- Sentences processed: 57
- Translation candidates: 570
- Average candidates per sentence: 10.0
- Entropy metrics per token: 18

### MANUAL_AUTOREGRESSIVE Analysis
- Sentences processed: 57
- Translation candidates: 570
- Average candidates per sentence: 10.0
- Entropy metrics per token: 27

## Entropy Statistics

| Method | Source Entropy (μ±σ) | Target Entropy (μ±σ) | Source Surprisal (μ±σ) | Target Surprisal (μ±σ) |
|--------|---------------------|---------------------|----------------------|----------------------|
| Full | 4.425±3.175 | 3.561±1.535 | 2.209±4.369 | 2.529±5.712 |
| Conditional | 7.455±3.786 | 8.549±3.539 | 15.446±4.657 | 13.751±6.045 |
| Manual Autoregressive | 7.455±3.786 | N/A | 15.446±4.657 | N/A |

## Method Comparison

Comparison based on sample sentence entropy metrics:

| Method | Average Entropy | Std Deviation | Entropy Metric |
|--------|----------------|---------------|----------------|
| Full | 4.6063 | 2.6536 | full_source_entropy_bits |
| Conditional | 6.9710 | 3.3952 | conditional_source_entropy_bits |
| Manual Autoregressive | 6.9710 | 3.3952 | manual_autoregressive_seq2seq_source_entropy_bits |

## Files Generated

### Core Results
- `svo_full.json` - Complete analysis results
- `svo_full.csv` - Token-level metrics
- `svo_conditional.json` - Complete analysis results
- `svo_conditional.csv` - Token-level metrics
- `svo_manual_autoregressive.json` - Complete analysis results
- `svo_manual_autoregressive.csv` - Token-level metrics

### Analysis Files
- `svo_comprehensive.json` - All methods combined
- `svo_method_comparison.json` - Cross-method comparison
- `svo_enhanced_token_analysis.csv` - Comprehensive token data

### Visualizations
- `svo_method_entropy_comparison.png` - Entropy distributions
- `svo_method_summary_stats.png` - Summary statistics
- `svo_token_level_analysis.png` - Token-level analysis

## Methodology

This analysis implements four different approaches to measuring translation entropy:

1. **FULL Analysis**: Uses seq2seq model for autoregressive generation and conditional model for bigram analysis
2. **CONDITIONAL Analysis**: Uses conditional model exclusively for all calculations
3. **MANUAL_AUTOREGRESSIVE Analysis**: Uses both models with manual masking and autoregressive context
4. **MANUAL_BIGRAM_DECODER Analysis**: Uses conditional model for bigram context with decoder masking

Each method calculates:
- Standard entropy and surprisal
- Top-k entropy and mass (k=5)
- Entmax15 variants
- Branching factor metrics
- Complete linguistic annotation (POS/DEP tags)
