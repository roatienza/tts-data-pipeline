# LibriTTS Dataset Processing Guide

This document describes how to process the LibriTTS dataset using the TTS data pipeline.

## Dataset Structure

LibriTTS is organized as:
```
LibriTTS/
├── train-clean-100/
│   └── {speaker_id}/
│       └── {chapter_id}/
│           ├── {audio_id}.wav
│           ├── {audio_id}.normalized.txt
│           └── {audio_id}.original.txt
├── train-clean-360/
├── train-other-500/
├── dev-clean/
├── dev-other/
├── test-clean/
└── test-other/
```

## Processing Script

Use `src/process_libritts_dataset.py` to process LibriTTS:

```bash
cd /workspace/tts-data-pipeline
python src/process_libritts_dataset.py
```

This will:
1. Scan the LibriTTS dataset
2. Create train/val/test splits
3. Process audio files through the MEL extraction pipeline
4. Save results to `/data/tts/outputs/libritts_pipeline/`

## Output

- **Audio files**: `/data/tts/outputs/libritts_pipeline/{split}/audio/`
- **MEL spectrograms**: `/data/tts/outputs/libritts_pipeline/{split}/mel/`
- **Results**: `/data/tts/outputs/libritts_pipeline/{split}/results.json`
- **Splits**: `/data/tts/processed/libritts_splits/`

## Configuration

Edit `config/pipeline.yaml` to adjust:
- Sample rate (default: 22050 Hz)
- MEL parameters (n_mels, n_fft, hop_length, etc.)
- Data split ratios

## Processing All Subsets

To process all LibriTTS subsets, modify the `subset_dirs` list in `process_libritts_dataset.py`:

```python
subset_dirs = ['train-clean-100', 'train-clean-360', 'train-other-500',
               'dev-clean', 'dev-other', 'test-clean', 'test-other']
```

## Notes

- The pipeline currently processes a subset of data for demonstration
- Vocos vocoder is used for audio reconstruction when available
- If Vocos fails to load, the pipeline falls back to librosa-only processing
- Processing is resumable - existing files are skipped
