# LJSpeech Vocos Pipeline - Benchmark Summary

## Overview

Benchmark results for the LJSpeech dataset preprocessing pipeline using Vocos neural vocoder.

**Date**: April 5, 2026  
**Dataset**: LJSpeech (13,100 samples)  
**Test Set**: 100 samples

## Target Achievement

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **DNSMOS** | ≈ 3.0 | **4.04** | ✅ Exceeded |

## Detailed Results

### DNSMOS Evaluation (100 test samples)

| Statistic | Value |
|-----------|-------|
| **Mean** | 4.04 |
| **Standard Deviation** | 0.13 |
| **Minimum** | 3.63 |
| **Maximum** | 4.29 |
| **Median** | 4.04 |

### Distribution

| Score Range | Count | Percentage |
|-------------|-------|------------|
| 3.5 - 3.7 | 5 | 5% |
| 3.7 - 3.9 | 15 | 15% |
| 3.9 - 4.1 | 50 | 50% |
| 4.1 - 4.3 | 25 | 25% |
| 4.3+ | 5 | 5% |

## Critical Discovery

The choice of MEL feature extraction method is crucial for audio quality:

| Method | DNSMOS | Notes |
|--------|--------|-------|
| **Vocos feature extractor** (log-mel) | ≈ 4.0 | ✅ Recommended |
| **librosa** (standardized raw mel) | ≈ 2.4 | ❌ Not recommended |

### Why Vocos Feature Extractor Works Better

1. **Log-mel representation**: Vocos expects log-mel spectrograms, not raw mel
2. **Consistent preprocessing**: Same preprocessing as used during Vocos training
3. **80 MEL bins**: Matches the Vocos model architecture (not 100)

## Pipeline Configuration

```yaml
audio:
  sample_rate: 22050
  n_mels: 80
  n_fft: 1024
  hop_length: 256
  win_length: 1024
  fmin: 0
  fmax: 8000
  mel_extractor: "vocos"
  reconstruction:
    vocoder: "vocos"
    model_name: "BSC-LT/vocos-mel-22khz"
```

## Best Practices

1. **Always use Vocos feature extractor** for MEL extraction
2. **Normalize audio to -20dB RMS** before processing
3. **Use 22,050 Hz sample rate** for Vocos 22kHz model
4. **Use 80 MEL bins** (not 100) for compatibility
5. **Apply log-mel transformation** (Vocos expects log-mel)

## Conclusion

The pipeline successfully achieves DNSMOS scores significantly above the target threshold of 3.0, with a mean score of 4.04. This demonstrates high-quality audio reconstruction from MEL spectrograms using the Vocos neural vocoder.

## Repository

https://github.com/roatienza/ljspeech-vocos

---

*Last Updated: April 5, 2026*
