# Enhanced Translation Entropy Analysis Report

**Analysis Date:** 2025-06-27 08:36:43

**Dataset:** szwedek_pairs

## Configuration

- **model_name:** facebook/mbart-large-50-many-to-many-mmt
- **src_lang:** pl_PL
- **tgt_lang:** en_XX
- **num_beams:** 10
- **num_return_sequences:** 10
- **output_dir:** /content/drive/MyDrive/NMT/corpus_annotation/szwedek_pairs
- **run_comet:** True
- **run_individual_analyses:** True

## Dataset Information

- **Number of sentences:** 11
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
- Sentences processed: 11
- Translation candidates: 110
- Average candidates per sentence: 10.0
- Entropy metrics per token: 18

### CONDITIONAL Analysis
- Sentences processed: 11
- Translation candidates: 110
- Average candidates per sentence: 10.0
- Entropy metrics per token: 18

### MANUAL_AUTOREGRESSIVE Analysis
- Sentences processed: 11
- Translation candidates: 110
- Average candidates per sentence: 10.0
- Entropy metrics per token: 27

## Entropy Statistics

| Method | Source Entropy (μ±σ) | Target Entropy (μ±σ) | Source Surprisal (μ±σ) | Target Surprisal (μ±σ) |
|--------|---------------------|---------------------|----------------------|----------------------|
| Full | 3.576±2.544 | 3.634±1.831 | 6.082±7.663 | 4.377±6.592 |
| Conditional | 7.721±4.809 | 9.311±3.344 | 16.392±4.918 | 13.518±5.009 |
| Manual Autoregressive | 7.721±4.809 | N/A | 16.392±4.918 | N/A |

## Method Comparison

Comparison based on sample sentence entropy metrics:

| Method | Average Entropy | Std Deviation | Entropy Metric |
|--------|----------------|---------------|----------------|
| Full | 3.1988 | 2.4385 | full_source_entropy_bits |
| Conditional | 7.7557 | 4.5662 | conditional_source_entropy_bits |
| Manual Autoregressive | 7.7557 | 4.5662 | manual_autoregressive_seq2seq_source_entropy_bits |

## Files Generated

### Core Results
- `szwedek_pairs_full.json` - Complete analysis results
- `szwedek_pairs_full.csv` - Token-level metrics
- `szwedek_pairs_conditional.json` - Complete analysis results
- `szwedek_pairs_conditional.csv` - Token-level metrics
- `szwedek_pairs_manual_autoregressive.json` - Complete analysis results
- `szwedek_pairs_manual_autoregressive.csv` - Token-level metrics

### Analysis Files
- `szwedek_pairs_comprehensive.json` - All methods combined
- `szwedek_pairs_method_comparison.json` - Cross-method comparison
- `szwedek_pairs_enhanced_token_analysis.csv` - Comprehensive token data

### Visualizations
- `szwedek_pairs_method_entropy_comparison.png` - Entropy distributions
- `szwedek_pairs_method_summary_stats.png` - Summary statistics
- `szwedek_pairs_token_level_analysis.png` - Token-level analysis

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
