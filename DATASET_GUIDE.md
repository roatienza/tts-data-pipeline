# LJSpeech Vocos Preprocessed Dataset - User Guide

## Overview

This repository contains a complete data preprocessing and validation pipeline for Text-to-Speech (TTS) systems, along with the fully preprocessed LJSpeech dataset using Vocos neural vocoder.

**Dataset Status**: All 13,100 LJSpeech samples have been preprocessed through the WAV → MEL → WAV pipeline.

## Quick Start

### 1. Locate the Preprocessed Data

```
/data/tts/
├── processed/
│   └── metadata_with_split.csv    # 13,100 entries with train/test labels
└── outputs/full_pipeline/
    ├── train/
    │   ├── audio/                 # 13,000 reconstructed WAV files (3.6 GB)
    │   ├── mel/                   # 13,000 MEL spectrogram .npy files (2.3 GB)
    │   ├── results.json           # Detailed processing results
    │   └── statistics.json        # Aggregate statistics
    └── test/
        ├── audio/                 # 100 reconstructed WAV files (29 MB)
        ├── mel/                   # 100 MEL spectrogram .npy files (18 MB)
        ├── results.json           # Detailed processing results
        └── statistics.json        # Aggregate statistics
```

### 2. Load the Metadata

```python
import pandas as pd

# Load metadata with split labels
metadata = pd.read_csv(
    '/data/tts/processed/metadata_with_split.csv',
    sep='|',
    names=['audio_id', 'raw_text', 'norm_text', 'split']
)

# View data split
print(metadata['split'].value_counts())
# Output:
# train    13000
# test       100
```

### 3. Access Preprocessed Data

#### Load MEL Spectrogram

```python
import numpy as np

# Load MEL spectrogram for a specific audio ID
audio_id = 'LJ001-0001'
mel_path = f'/data/tts/outputs/full_pipeline/train/mel/{audio_id}.npy'
mel_spec = np.load(mel_path)

print(f"MEL shape: {mel_spec.shape}")  # (80, time_steps)
```

#### Load Reconstructed Audio

```python
import soundfile as sf

# Load reconstructed WAV file
audio_id = 'LJ001-0001'
audio_path = f'/data/tts/outputs/full_pipeline/train/audio/{audio_id}.wav'
audio, sample_rate = sf.read(audio_path)

print(f"Audio shape: {audio.shape}, Sample rate: {sample_rate}")
# Output: Audio shape: (N,), Sample rate: 22050
```

## Data Split Configuration

| Split | Count | Percentage |
|-------|-------|------------|
| **Train** | 13,000 | 99.24% |
| **Test** | 100 | 0.76% |
| **Validation** | 0 | 0% |
| **Total** | 13,100 | 100% |

## Data Format Specifications

### Audio Files (WAV)

- **Format**: WAV, 16-bit PCM
- **Sample Rate**: 22,050 Hz
- **Channels**: Mono
- **Duration**: 1-10 seconds (typical for LJSpeech)
- **Normalization**: RMS normalized to -20dB

### MEL Spectrograms (.npy)

- **Format**: NumPy array (.npy)
- **Shape**: (80, time_steps)
- **MEL bins**: 80
- **Type**: Log-mel spectrogram
- **Extraction**: Vocos feature extractor (BSC-LT/vocos-mel-22khz)
- **Parameters**:
  - `n_fft`: 1024
  - `hop_length`: 256
  - `win_length`: 1024
  - `fmin`: 0
  - `fmax`: 8000

### Metadata (CSV)

- **Format**: Pipe-separated CSV
- **Columns**:
  - `audio_id`: Unique identifier (e.g., "LJ001-0001")
  - `raw_text`: Original text from LJSpeech
  - `norm_text`: Normalized text (lowercase, expanded abbreviations)
  - `split`: Dataset split label ("train" or "test")

## Pipeline Configuration

### Audio Pipeline Parameters

```yaml
audio:
  sample_rate: 22050
  n_mels: 80
  n_fft: 1024
  hop_length: 256
  win_length: 1024
  fmin: 0
  fmax: 8000
  mel_extractor: "vocos"  # Vocos feature extractor for log-mel
  reconstruction:
    vocoder: "vocos"
    model_name: "BSC-LT/vocos-mel-22khz"
```

### Text Processing Parameters

```yaml
text:
  language: "en-us"
  normalize: true
  lowercase: true
  expand_abbreviations: true
  convert_numbers: true
  phonemizer:
    backend: "espeak"
    preserve_punctuation: true
    with_stress: true
    tie: true
```

## Quality Metrics

### DNSMOS Evaluation (Test Set)

From benchmark testing on 100 test samples:

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **DNSMOS Mean** | 4.04 | ≈ 3.0 | ✅ Exceeded |
| **DNSMOS Std** | 0.13 | - | - |
| **DNSMOS Range** | 3.63 - 4.29 | - | - |

**Conclusion**: Audio reconstruction quality significantly exceeds the target threshold.

## Usage Examples

### Example 1: Load Training Data

```python
import pandas as pd
import numpy as np
import soundfile as sf

# Load metadata
metadata = pd.read_csv(
    '/data/tts/processed/metadata_with_split.csv',
    sep='|',
    names=['audio_id', 'raw_text', 'norm_text', 'split']
)

# Filter for training data
train_data = metadata[metadata['split'] == 'train']

# Load first 10 training samples
for i, row in train_data.head(10).iterrows():
    audio_id = row['audio_id']
    text = row['norm_text']
    
    # Load MEL spectrogram
    mel = np.load(f'/data/tts/outputs/full_pipeline/train/mel/{audio_id}.npy')
    
    # Load reconstructed audio
    audio, sr = sf.read(f'/data/tts/outputs/full_pipeline/train/audio/{audio_id}.wav')
    
    print(f"Sample {i+1}: {audio_id}")
    print(f"  Text: {text[:50]}...")
    print(f"  MEL shape: {mel.shape}")
    print(f"  Audio shape: {audio.shape}")
```

### Example 2: Create PyTorch Dataset

```python
import torch
from torch.utils.data import Dataset
import numpy as np
import soundfile as sf
import pandas as pd

class LJSpeechVocosDataset(Dataset):
    def __init__(self, split='train'):
        self.metadata = pd.read_csv(
            '/data/tts/processed/metadata_with_split.csv',
            sep='|',
            names=['audio_id', 'raw_text', 'norm_text', 'split']
        )
        self.metadata = self.metadata[self.metadata['split'] == split]
        self.split = split
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_id = row['audio_id']
        text = row['norm_text']
        
        # Load MEL spectrogram
        mel = np.load(f'/data/tts/outputs/full_pipeline/{self.split}/mel/{audio_id}.npy')
        
        # Load audio
        audio, sr = sf.read(f'/data/tts/outputs/full_pipeline/{self.split}/audio/{audio_id}.wav')
        
        return {
            'audio_id': audio_id,
            'text': text,
            'mel': torch.tensor(mel, dtype=torch.float32),
            'audio': torch.tensor(audio, dtype=torch.float32),
            'sample_rate': sr
        }

# Usage
dataset = LJSpeechVocosDataset(split='train')
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    print(f"Batch MEL shape: {batch['mel'].shape}")  # (32, 80, time_steps)
    print(f"Batch audio shape: {batch['audio'].shape}")  # (32, time_samples)
    break
```

### Example 3: Re-process New Audio Files

```python
from src.audio_pipeline import AudioPipeline

# Initialize pipeline
pipeline = AudioPipeline('config/pipeline.yaml')
pipeline.load_vocos_model()

# Process a new audio file
result = pipeline.process_file(
    'path/to/new_audio.wav',
    output_dir='/data/tts/outputs/custom'
)

# Access results
mel_spec = result['mel']  # MEL spectrogram
reconstructed_audio = result['audio']  # Reconstructed audio
dnsmos_score = result.get('dnsmos_score')  # Quality score
```

## Reprocessing Data

### Reprocess All Training Data

```bash
python src/complete_preprocessing.py --split train
```

### Reprocess Test Data Only

```bash
python src/complete_preprocessing.py --split test
```

### Reprocess Specific Audio IDs

```python
from src.complete_preprocessing import process_audio_files

audio_ids = ['LJ001-0001', 'LJ001-0002', 'LJ001-0003']
process_audio_files(audio_ids, output_dir='/data/tts/outputs/custom')
```

## Directory Structure

```
/workspace/tts/
├── src/
│   ├── audio_pipeline.py          # Audio processing (WAV → MEL → WAV)
│   ├── text_pipeline.py           # Text processing (Text → IPA)
│   ├── data_split.py              # Dataset splitting utility
│   ├── evaluation.py              # DNSMOS evaluation
│   ├── complete_preprocessing.py  # Full dataset preprocessing
│   └── run_pipeline.py            # Pipeline runner
├── config/
│   └── pipeline.yaml              # Pipeline configuration
├── tests/                         # Test scripts
├── logs/                          # Execution logs
├── README.md                      # Main README
├── DATASET_GUIDE.md              # This file
├── FULL_PREPROCESSING_SUMMARY.md # Preprocessing completion summary
└── SUMMARY.md                     # Benchmark results

/data/tts/
├── processed/
│   └── metadata_with_split.csv    # 13,100 entries with split labels
└── outputs/full_pipeline/
    ├── train/
    │   ├── audio/                 # 13,000 WAV files
    │   ├── mel/                   # 13,000 .npy files
    │   ├── results.json
    │   └── statistics.json
    └── test/
        ├── audio/                 # 100 WAV files
        ├── mel/                   # 100 .npy files
        ├── results.json
        └── statistics.json
```

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Key Dependencies

```yaml
audio_processing:
  - librosa
  - soundfile
  - scipy
  - numpy

vocoder:
  - vocos
  - torch

text_processing:
  - phonemizer
  - inflect

evaluation:
  - dnsmos
```

## Troubleshooting

### Issue: MEL shape mismatch

**Symptom**: MEL spectrogram shape doesn't match expected (80, time_steps)

**Solution**: Ensure you're using the Vocos feature extractor, not librosa directly.

```python
# Correct: Use Vocos feature extractor
from vocos import Vocos
vocos = Vocos.from_pretrained('BSC-LT/vocos-mel-22khz')
mel = vocos.feature_extractor(audio)

# Incorrect: Using librosa directly (produces different shape)
import librosa
mel = librosa.feature.melspectrogram(y, sr=22050, n_mels=100)  # Wrong!
```

### Issue: Audio quality too low

**Symptom**: DNSMOS score below 3.0

**Solution**: 
1. Verify using Vocos feature extractor (not librosa)
2. Check RMS normalization is applied (-20dB)
3. Ensure sample rate is 22050 Hz

### Issue: Missing files

**Symptom**: Audio or MEL files not found

**Solution**: 
1. Verify the audio_id exists in metadata
2. Check the correct split directory (train vs test)
3. Run preprocessing again if files are missing

## License

MIT License

## Contact

For issues or questions, please refer to the repository documentation or contact the maintainer.

---

*Last Updated: April 5, 2026*
*Dataset Version: 1.0*
*Status: Complete (13,100/13,100 samples processed)*
