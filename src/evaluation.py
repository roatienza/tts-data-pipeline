"""
Evaluation module for audio quality using DNSMOS (via torchmetrics)
"""

import os
import numpy as np
import torch
import librosa
from typing import Dict, List, Optional
from pathlib import Path


def evaluate_audio_quality(audio_path: str, sample_rate: int = 16000) -> Dict:
    """
    Evaluate audio quality using DNSMOS via torchmetrics.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate for DNSMOS (default: 16000)
        
    Returns:
        Dictionary with DNSMOS scores
    """
    try:
        from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Convert to tensor - shape should be (batch, time)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)  # (1, 1, time)
        
        # Initialize DNSMOS metric
        dnsmos_metric = DeepNoiseSuppressionMeanOpinionScore(
            fs=sample_rate,
            personalized=False
        )
        
        # Compute DNSMOS
        score = dnsmos_metric(audio_tensor)
        
        # Handle different output shapes
        if score.dim() == 1 and score.numel() == 4:
            # (4,) - [overall, background_noise, audio_quality, similarity]
            return {
                'overall': float(score[0].item()),
                'background_noise': float(score[1].item()),
                'audio_quality': float(score[2].item()),
                'similarity': float(score[3].item()),
                'valid': True,
            }
        elif score.dim() == 2:
            # (batch, 4) - [overall, background_noise, audio_quality, similarity]
            return {
                'overall': float(score[0, 0].item()),
                'background_noise': float(score[0, 1].item()),
                'audio_quality': float(score[0, 2].item()),
                'similarity': float(score[0, 3].item()),
                'valid': True,
            }
        elif score.dim() == 1 and score.numel() == 1:
            # Single score
            return {
                'overall': float(score.item()),
                'background_noise': float(score.item()),
                'audio_quality': float(score.item()),
                'similarity': float(score.item()),
                'valid': True,
            }
        else:
            return {
                'error': f'Unexpected score shape: {score.shape}, numel: {score.numel()}',
                'valid': False,
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'valid': False,
        }


def evaluate_batch(audio_files: List[str], sample_rate: int = 16000) -> List[Dict]:
    """
    Evaluate multiple audio files.
    
    Args:
        audio_files: List of audio file paths
        sample_rate: Target sample rate
        
    Returns:
        List of evaluation results
    """
    results = []
    for audio_path in audio_files:
        if os.path.exists(audio_path):
            result = evaluate_audio_quality(audio_path, sample_rate)
            result['file'] = audio_path
            results.append(result)
        else:
            results.append({
                'file': audio_path,
                'error': 'File not found',
                'valid': False,
            })
    return results


def compute_statistics(results: List[Dict]) -> Dict:
    """
    Compute statistics from evaluation results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Statistics dictionary
    """
    valid_results = [r for r in results if r.get('valid', False)]
    
    if not valid_results:
        return {'error': 'No valid results'}
    
    metrics = ['overall', 'background_noise', 'audio_quality', 'similarity']
    
    stats = {}
    for metric in metrics:
        values = [r[metric] for r in valid_results if metric in r]
        if values:
            stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values),
            }
    
    stats['total_valid'] = len(valid_results)
    stats['total_files'] = len(results)
    
    return stats


def main():
    """Test DNSMOS evaluation."""
    # Test with a sample file
    test_audio = '/data/tts/outputs/test/reconstructed_LJ001-0001.wav'
    
    if os.path.exists(test_audio):
        print(f"Evaluating: {test_audio}")
        result = evaluate_audio_quality(test_audio)
        print(f"DNSMOS Results: {result}")
    else:
        print(f"Test audio not found: {test_audio}")
        print("Run audio_pipeline.py first to generate test output.")


if __name__ == '__main__':
    main()
