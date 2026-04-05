# LJSpeech Vocos Data Pipeline

A data preprocessing and validation pipeline for Text-to-Speech (TTS) systems using the LJSpeech dataset and Vocos neural vocoder.

## Overview

This project implements robust data transformation pipelines for the LJSpeech dataset:

### Audio Pipeline: WAV → MEL → WAV
- **Dataset**: LJSpeech (13,100 samples)
- **Input**: Raw WAV audio files
- **Feature Extraction**: MEL spectrograms using Vocos feature extractor
- **Reconstruction**: Audio synthesis using Vocos neural vocoder (BSC-LT/vocos-mel-22khz)
- **Quality Assessment**: DNSMOS evaluation

### Text Pipeline: Text → IPA → Text
- **Input**: Raw text from LJSpeech metadata
- **Normalization**: Abbreviation expansion, number conversion
- **Phonemization**: IPA conversion using phonemizer (espeak backend)
- **Validation**: IPA quality checks

## Preprocessed Dataset

**Status**: All 13,100 LJSpeech samples have been preprocessed.

| Split | Count | Description |
|-------|-------|-------------|
| **Train** | 13,000 | Training data |
| **Test** | 100 | Test data |
| **Validation** | 0 | No validation split |

### Data Location

```
/data/tts/
├── processed/
│   └── metadata_with_split.csv    # 13,100 entries with train/test labels
└── outputs/full_pipeline/
    ├── train/
    │   ├── audio/                 # 13,000 reconstructed WAV files (3.6 GB)
    │   ├── mel/                   # 13,000 MEL spectrogram .npy files (2.3 GB)
    │   ├── results.json
    │   └── statistics.json
    └── test/
        ├── audio/                 # 100 reconstructed WAV files (29 MB)
        ├── mel/                   # 100 MEL spectrogram .npy files (18 MB)
        ├── results.json
        └── statistics.json
```

### Quality Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **DNSMOS Mean** | 4.04 | ≈ 3.0 | ✅ Exceeded |
| **DNSMOS Std** | 0.13 | - | - |
| **DNSMOS Range** | 3.63 - 4.29 | - | - |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start - Load Preprocessed Data

```python
import pandas as pd
import numpy as np
import soundfile as sf

# Load metadata with split labels
metadata = pd.read_csv(
    '/data/tts/processed/metadata_with_split.csv',
    sep='|',
    names=['audio_id', 'raw_text', 'norm_text', 'split']
)

# Load a training sample
audio_id = 'LJ001-0001'
mel = np.load(f'/data/tts/outputs/full_pipeline/train/mel/{audio_id}.npy')
audio, sr = sf.read(f'/data/tts/outputs/full_pipeline/train/audio/{audio_id}.wav')

print(f"MEL shape: {mel.shape}")  # (80, time_steps)
print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
```

### Audio Pipeline

```python
from src.audio_pipeline import AudioPipeline

# Initialize pipeline
pipeline = AudioPipeline('config/pipeline.yaml')
pipeline.load_vocos_model()

# Process audio file
result = pipeline.process_file('path/to/audio.wav', output_dir='outputs')
```

### Text Pipeline

```python
from src.text_pipeline import TextPipeline

# Initialize pipeline
pipeline = TextPipeline('config/pipeline.yaml')

# Process text
result = pipeline.process_text("Hello, world!")
print(result['ipa'])
```

### Run Full Pipeline

```bash
python src/run_pipeline.py
```

### Data Splitting

```bash
python src/data_split.py
```

## Configuration

Edit `config/pipeline.yaml` to customize:

- Audio parameters (sample rate, MEL settings)
- Vocoder settings
- Text normalization rules
- Phonemizer options
- Data split ratios

## Directory Structure

```
/workspace/tts/
├── src/
│   ├── audio_pipeline.py          # Audio processing (WAV → MEL → WAV)
│   ├── text_pipeline.py           # Text processing (Text → IPA)
│   ├── evaluation.py              # DNSMOS evaluation
│   ├── data_split.py              # Dataset splitting
│   ├── complete_preprocessing.py  # Full dataset preprocessing
│   └── run_pipeline.py            # Main pipeline runner
├── config/
│   └── pipeline.yaml              # Configuration
├── tests/                         # Test scripts
├── logs/                          # Execution logs
├── README.md                      # This file
├── DATASET_GUIDE.md              # Comprehensive dataset usage guide
├── FULL_PREPROCESSING_SUMMARY.md # Preprocessing completion summary
└── SUMMARY.md                     # Benchmark results
```

## Documentation

- **README.md** - This file, quick overview and usage
- **DATASET_GUIDE.md** - Comprehensive guide for using the preprocessed dataset
- **FULL_PREPROCESSING_SUMMARY.md** - Summary of the preprocessing completion
- **SUMMARY.md** - Benchmark results and quality metrics

## Data Storage

- **Raw data**: `/data/tts/raw/`
- **Processed data**: `/data/tts/processed/`
- **Cache**: `/data/tts/cache/`
- **Outputs**: `/data/tts/outputs/`

## Benchmark Targets

| Metric | Target | Description |
|--------|--------|-------------|
| DNSMOS | ≈ 3.0 | Audio reconstruction quality |

## License

MIT
