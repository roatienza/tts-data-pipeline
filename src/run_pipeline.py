"""
Main pipeline script for processing TTS data.
Processes audio files through WAV → MEL → WAV pipeline and evaluates quality.
"""

import os
import sys
import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json

# Add src to path
sys.path.insert(0, '/workspace/tts/src')

from audio_pipeline import AudioPipeline
from text_pipeline import TextPipeline
from data_split import load_metadata, split_dataset
from evaluation import evaluate_audio_quality, compute_statistics


def process_dataset(
    wav_dir: str,
    metadata_path: str,
    output_dir: str,
    config_path: str,
    test_samples: int = 100,
    max_samples: int = None
):
    """
    Process a subset of the dataset through the full pipeline.
    
    Args:
        wav_dir: Directory containing WAV files
        metadata_path: Path to metadata.csv
        output_dir: Output directory for processed files
        config_path: Path to configuration file
        test_samples: Number of test samples to process
        max_samples: Maximum samples to process (None for all test samples)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check which MEL extractor to use
    mel_extractor = config.get('audio', {}).get('mel_extractor', 'vocos')
    use_vocos_extractor = (mel_extractor == 'vocos')
    
    # Initialize pipelines
    audio_pipeline = AudioPipeline(config_path)
    audio_pipeline.load_vocos_model()
    
    text_pipeline = TextPipeline(config_path)
    
    # Load metadata and get test split
    print("Loading metadata...")
    data = load_metadata(metadata_path)
    audio_ids = [item[0] for item in data]
    
    # Use test split
    splits = split_dataset(metadata_path, test_samples=test_samples, seed=42)
    test_ids = splits['test']
    
    if max_samples:
        test_ids = test_ids[:max_samples]
    
    print(f"Processing {len(test_ids)} test samples...")
    print(f"Using MEL extractor: {mel_extractor}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    audio_output_dir = os.path.join(output_dir, 'audio')
    
    # Process samples
    results = []
    dnsmos_scores = []
    
    for i, audio_id in enumerate(tqdm(test_ids)):
        audio_path = os.path.join(wav_dir, f"{audio_id}.wav")
        
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            continue
        
        try:
            # Process audio
            audio_result = audio_pipeline.process_file(
                audio_path,
                output_dir=audio_output_dir,
                use_vocos_extractor=use_vocos_extractor
            )
            
            # Evaluate quality
            if audio_result.get('output_path'):
                eval_result = evaluate_audio_quality(
                    audio_result['output_path'],
                    sample_rate=16000
                )
                audio_result['dnsmos'] = eval_result
                
                if eval_result.get('valid', False):
                    dnsmos_scores.append(eval_result['overall'])
            else:
                audio_result['dnsmos'] = {'valid': False, 'error': 'No output path'}
            
            # Process text
            try:
                idx = audio_ids.index(audio_id)
                text = data[idx][1]
                text_result = text_pipeline.process_text(text)
                audio_result['text_result'] = text_result
            except Exception as e:
                audio_result['text_result'] = {'error': str(e)}
            
            results.append(audio_result)
            
        except Exception as e:
            print(f"Error processing {audio_id}: {e}")
            continue
    
    # Compute statistics
    if dnsmos_scores:
        print(f"\n=== DNSMOS Statistics ===")
        print(f"Mean: {np.mean(dnsmos_scores):.3f}")
        print(f"Std: {np.std(dnsmos_scores):.3f}")
        print(f"Min: {np.min(dnsmos_scores):.3f}")
        print(f"Max: {np.max(dnsmos_scores):.3f}")
        print(f"Count: {len(dnsmos_scores)}")
        
        # Check if target achieved
        target = 3.0
        if np.mean(dnsmos_scores) >= target:
            print(f"\n✓ Target DNSMOS ({target}) ACHIEVED!")
        else:
            print(f"\n✗ Target DNSMOS ({target}) NOT achieved. Current: {np.mean(dnsmos_scores):.3f}")
    else:
        print("\nNo valid DNSMOS scores to compute statistics.")
    
    # Save results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        json.dump(convert(results), f, indent=2, default=str)
    print(f"Results saved to: {results_path}")
    
    # Save statistics
    stats = {
        'mean': float(np.mean(dnsmos_scores)),
        'std': float(np.std(dnsmos_scores)),
        'min': float(np.min(dnsmos_scores)),
        'max': float(np.max(dnsmos_scores)),
        'count': int(len(dnsmos_scores)),
        'target': 3.0,
        'achieved': bool(np.mean(dnsmos_scores) >= 3.0)
    }
    stats_path = os.path.join(output_dir, 'statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_path}")
    
    return results


def main():
    """Run the full pipeline."""
    # Configuration
    config_path = '/workspace/tts/config/pipeline.yaml'
    wav_dir = '/data/tts/datasets/LJSpeech-1.1/wavs'
    metadata_path = '/data/tts/datasets/LJSpeech-1.1/metadata.csv'
    output_dir = '/data/tts/outputs/pipeline'
    
    # Process test samples
    print("=== TTS Data Pipeline ===\n")
    print(f"Config: {config_path}")
    print(f"WAV dir: {wav_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Process all 100 test samples
    results = process_dataset(
        wav_dir=wav_dir,
        metadata_path=metadata_path,
        output_dir=output_dir,
        config_path=config_path,
        test_samples=100,
        max_samples=None  # Process all 100 test samples
    )
    
    print(f"\nProcessed {len(results)} samples successfully.")


if __name__ == '__main__':
    main()
