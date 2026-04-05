# LJSpeech Vocos Preprocessing - Completion Summary

## Overview

This document summarizes the complete preprocessing of the LJSpeech dataset using the Vocos neural vocoder pipeline.

**Date Completed**: April 5, 2026  
**Total Samples**: 13,100  
**Status**: ✅ Complete

## Preprocessing Pipeline

### Audio Pipeline: WAV → MEL → WAV

1. **Input**: Raw WAV audio files from LJSpeech dataset
2. **Preprocessing**:
   - Sample rate conversion to 22,050 Hz
   - RMS normalization to -20dB

3. **Feature Extraction**:
   - Vocos feature extractor (BSC-LT/vocos-mel-22khz)
   - MEL spectrograms: 80 bins, log-mel
   - Parameters: n_fft=1024, hop_length=256, win_length=1024, fmin=0, fmax=8000
4. **Reconstruction**:
   - Vocos neural vocoder (BSC-LT/vocos-mel-22khz)
   - Decode MEL spectrogram to audio
5. **Quality Assessment**:
   - DNSMOS evaluation

### Text Pipeline: Text → IPA → Text

1. **Input**: Raw text from LJSpeech metadata
2. **Normalization**:
   - Lowercase conversion
   - Abbreviation expansion
   - Number conversion
3. **Phonemization**:
   - phonemizer with espeak backend
   - Preserve punctuation, include stress markers

## Data Split

| Split | Count | Percentage |
|-------|-------|------------|
| **Train** | 13,000 | 99.24% |
| **Test** | 100 | 0.76% |
| **Validation** | 0 | 0% |
| **Total** | 13,100 | 100% |

## Output Files

### Training Data (13,000 samples)

- **Audio**: `/data/tts/outputs/full_pipeline/train/audio/` (3.6 GB)
- **MEL**: `/data/tts/outputs/full_pipeline/train/mel/` (2.3 GB)
- **Results**: `/data/tts/outputs/full_pipeline/train/results.json`
- **Statistics**: `/data/tts/outputs/full_pipeline/train/statistics.json`

### Test Data (100 samples)

- **Audio**: `/data/tts/outputs/full_pipeline/test/audio/` (29 MB)
- **MEL**: `/data/tts/outputs/full_pipeline/test/mel/` (18 MB)
- **Results**: `/data/tts/outputs/full_pipeline/test/results.json`
- **Statistics**: `/data/tts/outputs/full_pipeline/test/statistics.json`

### Metadata

- **File**: `/data/tts/processed/metadata_with_split.csv`
- **Format**: Pipe-separated CSV
- **Columns**: audio_id, raw_text, norm_text, split

## Quality Metrics

### DNSMOS Evaluation (Test Set - 100 samples)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **DNSMOS Mean** | 4.04 | ≈ 3.0 | ✅ Exceeded |
| **DNSMOS Std** | 0.13 | - | - |
| **DNSMOS Min** | 3.63 | - | - |
| **DNSMOS Max** | 4.29 | - | - |

**Conclusion**: Audio reconstruction quality significantly exceeds the target threshold of 3.0.

## Processing Statistics

### Training Set (13,000 samples)

- **Success Rate**: 100%
- **Errors**: 0
- **Average Processing Time**: ~0.5 seconds per sample
- **Total Processing Time**: ~1.8 hours

### Test Set (100 samples)

- **Success Rate**: 100%
- **Errors**: 0
- **Average Processing Time**: ~0.5 seconds per sample

## Key Findings

1. **Vocos Feature Extractor is Critical**: Using the Vocos feature extractor (log-mel) produces DNSMOS ≈ 4.0, while using librosa directly (standardized raw mel) produces DNSMOS ≈ 2.4.

2. **80 MEL Bins**: The BSC-LT/vocos-mel-22khz model uses 80 MEL bins, not 100 as initially specified.

3. **RMS Normalization**: Normalizing audio to -20dB RMS improves reconstruction quality.

4. **Sample Rate**: 22,050 Hz is optimal for the Vocos 22kHz model.

## Files Generated

```
/workspace/tts/
├── src/
│   ├── audio_pipeline.py          # Audio processing (WAV → MEL → WAV)
│   ├── text_pipeline.py           # Text processing (Text → IPA)
│   ├── data_split.py              # Dataset splitting utility
│   ├── evaluation.py              # DNSMOS evaluation
│   ├── complete_preprocessing.py  # Full dataset preprocessing
│   ├── finish_preprocessing.py    # Preprocessing completion script
│   ├── generate_summary.py        # Summary generation
│   ├── process_full_dataset.py    # Full dataset processing
│   └── run_pipeline.py            # Pipeline runner
├── config/
│   └── pipeline.yaml              # Pipeline configuration
├── README.md                      # Main README
├── DATASET_GUIDE.md              # Comprehensive dataset usage guide
├── FULL_PREPROCESSING_SUMMARY.md # This file
└── SUMMARY.md                     # Benchmark results
```

## Usage

See `DATASET_GUIDE.md` for comprehensive instructions on using the preprocessed dataset.

## Repository

This codebase is available at: https://github.com/roatienza/ljspeech-vocos

---

*Completed: April 5, 2026*  
*Version: 1.0*
