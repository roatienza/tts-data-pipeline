"""
Final preprocessing script - processes remaining samples efficiently.
"""

import os
import sys
import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json
import random
import gc
import warnings

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

sys.path.insert(0, '/workspace/tts/src')

from audio_pipeline import AudioPipeline
from text_pipeline import TextPipeline
from data_split import load_metadata


def main():
    """Process remaining train samples."""
    config_path = '/workspace/tts/config/pipeline.yaml'
    wav_dir = '/data/tts/datasets/LJSpeech-1.1/wavs'
    metadata_path = '/data/tts/processed/metadata_with_split.csv'
    output_dir = '/data/tts/outputs/full_pipeline'
    
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    mel_extractor = config.get('audio', {}).get('mel_extractor', 'vocos')
    use_vocos_extractor = (mel_extractor == 'vocos')
    
    print("Initializing pipelines...")
    audio_pipeline = AudioPipeline(config_path)
    audio_pipeline.load_vocos_model()
    text_pipeline = TextPipeline(config_path)
    
    # Load metadata with split info
    data = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 4:
                audio_id, raw_text, norm_text, split = parts[0], parts[1], parts[2], parts[3]
                data.append((audio_id, raw_text, norm_text, split))
    
    # Get train IDs
    train_data = [(d[0], d[1]) for d in data if d[3] == 'train']
    train_ids = [d[0] for d in train_data]
    
    print(f"Total train samples: {len(train_ids)}")
    
    # Check existing files
    audio_output_dir = os.path.join(output_dir, 'train', 'audio')
    mel_output_dir = os.path.join(output_dir, 'train', 'mel')
    existing_files = set(os.listdir(audio_output_dir))
    existing_count = len(existing_files)
    print(f"Existing files: {existing_count}")
    
    # Get IDs to process
    ids_to_process = [id for id in train_ids if f"{id}.wav" not in existing_files]
    print(f"Remaining to process: {len(ids_to_process)}")
    
    # Load existing results
    results_path = os.path.join(output_dir, 'train', 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results")
    else:
        all_results = []
    
    # Create text lookup
    text_lookup = {d[0]: d[1] for d in train_data}
    
    # Process remaining
    success_count = len(all_results)
    errors = []
    
    for i, audio_id in enumerate(tqdm(ids_to_process)):
        audio_path = os.path.join(wav_dir, f"{audio_id}.wav")
        
        if not os.path.exists(audio_path):
            errors.append({'audio_id': audio_id, 'error': 'File not found'})
            continue
        
        try:
            text = text_lookup.get(audio_id, "")
            
            audio_result = audio_pipeline.process_file(
                audio_path,
                output_dir=audio_output_dir,
                mel_output_dir=mel_output_dir,
                use_vocos_extractor=use_vocos_extractor
            )
            
            try:
                text_result = text_pipeline.process_text(text)
                audio_result['text_result'] = text_result
            except Exception as e:
                audio_result['text_result'] = {'error': str(e)}
            
            audio_result['status'] = 'success'
            all_results.append(audio_result)
            success_count += 1
            
        except Exception as e:
            errors.append({'audio_id': audio_id, 'error': str(e)})
            continue
        
        # Periodic save
        if (i + 1) % 100 == 0:
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            gc.collect()
            print(f"Progress: {i + 1}/{len(ids_to_process)} - Success: {success_count}")
    
    # Save final results
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save statistics
    stats = {
        'split': 'train',
        'total': len(train_ids),
        'success': success_count,
        'errors': len(errors),
    }
    stats_path = os.path.join(output_dir, 'train', 'statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save errors
    if errors:
        errors_path = os.path.join(output_dir, 'train', 'errors.json')
        with open(errors_path, 'w') as f:
            json.dump(errors, f, indent=2)
    
    print(f"\nTrain Processing Complete:")
    print(f"  Total: {len(train_ids)}")
    print(f"  Success: {success_count}")
    print(f"  Errors: {len(errors)}")
    
    # Generate summary
    test_stats_path = os.path.join(output_dir, 'test', 'statistics.json')
    with open(test_stats_path, 'r') as f:
        test_stats = json.load(f)
    
    summary = {
        'dataset': 'LJSpeech-1.1',
        'total_samples': 13100,
        'splits': {
            'train': {
                'count': len(train_ids),
                'processed': success_count,
                'errors': len(errors)
            },
            'test': {
                'count': 100,
                'processed': test_stats['success'],
                'errors': test_stats['errors']
            }
        },
        'output_directory': output_dir,
        'updated_metadata': metadata_path
    }
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Dataset: LJSpeech-1.1")
    print(f"Total samples: 13,100")
    print(f"Train: {len(train_ids)} samples, {success_count} processed")
    print(f"Test: 100 samples, {test_stats['success']} processed")
    print("=" * 60)


if __name__ == '__main__':
    main()
