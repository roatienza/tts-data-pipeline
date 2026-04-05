"""
Full dataset preprocessing script for LJSpeech - Optimized Version.
Processes all 13,100 samples through WAV -> MEL -> WAV pipeline.
Allocates: 100 test, 0 val, 13,000 train.
Uses CPU only and single-threaded processing for memory efficiency.
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

# Suppress warnings
warnings.filterwarnings('ignore')

# Set single-threaded operation
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Add src to path
sys.path.insert(0, '/workspace/tts/src')

from audio_pipeline import AudioPipeline
from text_pipeline import TextPipeline
from data_split import load_metadata, split_dataset


def create_full_split(
    metadata_path: str,
    test_samples: int = 100,
    seed: int = 42
) -> dict:
    """Split dataset into train/test with fixed test size and 0 val."""
    data = load_metadata(metadata_path)
    audio_ids = [item[0] for item in data]
    
    random.seed(seed)
    shuffled_ids = audio_ids.copy()
    random.shuffle(shuffled_ids)
    
    test_ids = shuffled_ids[:test_samples]
    train_ids = shuffled_ids[test_samples:]
    
    return {
        'train': train_ids,
        'test': test_ids,
    }


def save_splits(splits: dict, output_dir: str):
    """Save splits to text files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, ids in splits.items():
        output_path = os.path.join(output_dir, f'{split_name}.txt')
        with open(output_path, 'w') as f:
            for audio_id in ids:
                f.write(f"{audio_id}\n")
        print(f"Saved {len(ids)} samples to {output_path}")


def update_metadata_with_split(
    metadata_path: str,
    splits: dict,
    output_path: str
):
    """Update metadata file to include split information."""
    data = load_metadata(metadata_path)
    
    train_set = set(splits['train'])
    test_set = set(splits['test'])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (audio_id, raw_text, norm_text) in enumerate(data):
            if audio_id in train_set:
                split = 'train'
            elif audio_id in test_set:
                split = 'test'
            else:
                split = 'unknown'
            
            f.write(f"{audio_id}|{raw_text}|{norm_text}|{split}\n")
    
    print(f"Updated metadata saved to: {output_path}")


def process_single_file(
    audio_pipeline,
    text_pipeline,
    audio_path: str,
    audio_id: str,
    text: str,
    audio_output_dir: str,
    mel_output_dir: str,
    use_vocos_extractor: bool = False
) -> dict:
    """Process a single audio file."""
    try:
        # Process audio
        audio_result = audio_pipeline.process_file(
            audio_path,
            output_dir=audio_output_dir,
            mel_output_dir=mel_output_dir,
            use_vocos_extractor=use_vocos_extractor
        )
        
        # Process text
        try:
            text_result = text_pipeline.process_text(text)
            audio_result['text_result'] = text_result
        except Exception as e:
            audio_result['text_result'] = {'error': str(e)}
        
        audio_result['status'] = 'success'
        return audio_result
        
    except Exception as e:
        return {
            'audio_id': audio_id,
            'status': 'error',
            'error': str(e)
        }


def process_all_samples(
    wav_dir: str,
    metadata_path: str,
    output_dir: str,
    config_path: str,
    splits: dict,
    split_name: str = 'train',
    max_samples: int = None
):
    """Process samples through the full pipeline."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    mel_extractor = config.get('audio', {}).get('mel_extractor', 'vocos')
    use_vocos_extractor = (mel_extractor == 'vocos')
    
    # Initialize pipelines ONCE
    print("Initializing audio pipeline...")
    audio_pipeline = AudioPipeline(config_path)
    audio_pipeline.load_vocos_model()
    
    print("Initializing text pipeline...")
    text_pipeline = TextPipeline(config_path)
    
    # Get the split to process
    ids_to_process = splits[split_name]
    
    if max_samples:
        ids_to_process = ids_to_process[:max_samples]
    
    print(f"Processing {len(ids_to_process)} {split_name} samples...")
    print(f"Using MEL extractor: {mel_extractor}")
    
    # Create output directories
    split_output_dir = os.path.join(output_dir, split_name)
    audio_output_dir = os.path.join(split_output_dir, 'audio')
    mel_output_dir = os.path.join(split_output_dir, 'mel')
    os.makedirs(split_output_dir, exist_ok=True)
    os.makedirs(audio_output_dir, exist_ok=True)
    os.makedirs(mel_output_dir, exist_ok=True)
    
    # Load metadata once
    data = load_metadata(metadata_path)
    audio_ids = [item[0] for item in data]
    
    # Check for existing processed files
    existing_files = set(os.listdir(audio_output_dir))
    existing_count = len(existing_files)
    print(f"Found {existing_count} existing files, resuming from there...")
    
    # Process samples
    all_results = []
    errors = []
    success_count = 0
    
    # Load existing results if any
    results_path = os.path.join(split_output_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            all_results = json.load(f)
        success_count = len(all_results)
        print(f"Loaded {success_count} existing results")
    
    for i, audio_id in enumerate(tqdm(ids_to_process)):
        # Skip already processed files
        if f"{audio_id}.wav" in existing_files:
            # Add to results if not already there
            if not any(r.get('audio_id') == audio_id for r in all_results):
                all_results.append({
                    'audio_id': audio_id,
                    'status': 'success',
                    'output_path': os.path.join(audio_output_dir, f"{audio_id}.wav"),
                    'mel_path': os.path.join(mel_output_dir, f"{audio_id}.npy")
                })
            continue
        
        audio_path = os.path.join(wav_dir, f"{audio_id}.wav")
        
        if not os.path.exists(audio_path):
            errors.append({'audio_id': audio_id, 'error': 'File not found'})
            continue
        
        # Get text for this audio
        try:
            idx = audio_ids.index(audio_id)
            text = data[idx][1]
        except:
            text = ""
        
        # Process file
        result = process_single_file(
            audio_pipeline=audio_pipeline,
            text_pipeline=text_pipeline,
            audio_path=audio_path,
            audio_id=audio_id,
            text=text,
            audio_output_dir=audio_output_dir,
            mel_output_dir=mel_output_dir,
            use_vocos_extractor=use_vocos_extractor
        )
        
        if result['status'] == 'success':
            success_count += 1
            all_results.append(result)
        else:
            errors.append(result)
        
        # Periodic save and memory cleanup
        if (i + 1) % 100 == 0:
            # Save results
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Progress: {i + 1}/{len(ids_to_process)} - Success: {success_count}")
    
    # Save final results
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")
    
    # Save statistics
    stats = {
        'split': split_name,
        'total': len(ids_to_process),
        'success': success_count,
        'errors': len(errors),
    }
    stats_path = os.path.join(split_output_dir, 'statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_path}")
    
    # Save errors
    if errors:
        errors_path = os.path.join(split_output_dir, 'errors.json')
        with open(errors_path, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"Errors saved to: {errors_path}")
    
    print(f"\n{split_name.upper()} Processing Complete:")
    print(f"  Total: {len(ids_to_process)}")
    print(f"  Success: {success_count}")
    print(f"  Errors: {len(errors)}")
    
    return all_results


def main():
    """Run the full dataset preprocessing."""
    config_path = '/workspace/tts/config/pipeline.yaml'
    wav_dir = '/data/tts/datasets/LJSpeech-1.1/wavs'
    metadata_path = '/data/tts/datasets/LJSpeech-1.1/metadata.csv'
    output_dir = '/data/tts/outputs/full_pipeline'
    splits_dir = '/data/tts/processed/splits'
    
    print("=" * 60)
    print("LJSpeech Full Dataset Preprocessing")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"WAV dir: {wav_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Step 1: Create splits
    print("Step 1: Creating data splits...")
    splits = create_full_split(
        metadata_path=metadata_path,
        test_samples=100,
        seed=42
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    print(f"  Total: {len(splits['train']) + len(splits['test'])} samples")
    
    save_splits(splits, splits_dir)
    
    # Update metadata
    updated_metadata_path = '/data/tts/processed/metadata_with_split.csv'
    update_metadata_with_split(metadata_path, splits, updated_metadata_path)
    
    # Step 2: Process test samples
    print("\n" + "=" * 60)
    print("Step 2: Processing TEST samples (100)...")
    print("=" * 60)
    
    test_results = process_all_samples(
        wav_dir=wav_dir,
        metadata_path=metadata_path,
        output_dir=output_dir,
        config_path=config_path,
        splits=splits,
        split_name='test',
        max_samples=None
    )
    
    # Step 3: Process train samples
    print("\n" + "=" * 60)
    print("Step 3: Processing TRAIN samples (13,000)...")
    print("=" * 60)
    
    train_results = process_all_samples(
        wav_dir=wav_dir,
        metadata_path=metadata_path,
        output_dir=output_dir,
        config_path=config_path,
        splits=splits,
        split_name='train',
        max_samples=None
    )
    
    # Step 4: Generate summary
    print("\n" + "=" * 60)
    print("Step 4: Generating summary...")
    print("=" * 60)
    
    # Load statistics
    test_stats_path = os.path.join(output_dir, 'test', 'statistics.json')
    train_stats_path = os.path.join(output_dir, 'train', 'statistics.json')
    
    with open(test_stats_path, 'r') as f:
        test_stats = json.load(f)
    
    with open(train_stats_path, 'r') as f:
        train_stats = json.load(f)
    
    # Create summary
    summary = {
        'dataset': 'LJSpeech-1.1',
        'total_samples': 13100,
        'splits': {
            'train': {
                'count': len(splits['train']),
                'processed': train_stats['success'],
                'errors': train_stats['errors']
            },
            'test': {
                'count': len(splits['test']),
                'processed': test_stats['success'],
                'errors': test_stats['errors']
            }
        },
        'output_directory': output_dir,
        'updated_metadata': updated_metadata_path
    }
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\nDataset: LJSpeech-1.1")
    print(f"Total samples: 13,100")
    print(f"\nSplits:")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    print(f"\nTrain Statistics:")
    print(f"  Processed: {train_stats['success']}")
    print(f"  Errors: {train_stats['errors']}")
    print(f"\nTest Statistics:")
    print(f"  Processed: {test_stats['success']}")
    print(f"  Errors: {test_stats['errors']}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Updated metadata: {updated_metadata_path}")
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
