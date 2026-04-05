"""
LibriTTS dataset preprocessing script.
Processes LibriTTS through WAV -> MEL -> WAV pipeline.
Adapted from LJSpeech preprocessing for LibriTTS structure.
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
from typing import List, Tuple

# Suppress warnings
warnings.filterwarnings('ignore')

# Set single-threaded operation
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Add src to path
sys.path.insert(0, '/workspace/ljspeech-vocos/src')

from audio_pipeline import AudioPipeline
from text_pipeline import TextPipeline


def scan_libritts_dataset(libritts_root: str, subset_dirs: List[str] = None) -> List[Tuple[str, str, str]]:
    """
    Scan LibriTTS dataset and collect all audio files with their text.
    
    Args:
        libritts_root: Root directory of LibriTTS dataset
        subset_dirs: List of subset directories to process (e.g., ['train-clean-100', 'test-clean'])
        
    Returns:
        List of (audio_id, normalized_text, original_text) tuples
    """
    if subset_dirs is None:
        subset_dirs = ['train-clean-100', 'train-clean-360', 'train-other-500', 
                       'dev-clean', 'dev-other', 'test-clean', 'test-other']
    
    all_data = []
    
    for subset in subset_dirs:
        subset_path = os.path.join(libritts_root, subset)
        if not os.path.exists(subset_path):
            print(f"Subset not found: {subset_path}")
            continue
        
        print(f"Scanning subset: {subset}")
        
        # Walk through speaker/chapter directories
        for speaker_id in os.listdir(subset_path):
            speaker_path = os.path.join(subset_path, speaker_id)
            if not os.path.isdir(speaker_path):
                continue
            
            for chapter_id in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter_id)
                if not os.path.isdir(chapter_path):
                    continue
                
                # Process files in chapter directory
                for filename in os.listdir(chapter_path):
                    if filename.endswith('.wav'):
                        audio_id = os.path.splitext(filename)[0]
                        wav_path = os.path.join(chapter_path, filename)
                        
                        # Get normalized text
                        norm_txt_path = wav_path.replace('.wav', '.normalized.txt')
                        orig_txt_path = wav_path.replace('.wav', '.original.txt')
                        
                        norm_text = ""
                        orig_text = ""
                        
                        if os.path.exists(norm_txt_path):
                            with open(norm_txt_path, 'r', encoding='utf-8') as f:
                                norm_text = f.read().strip()
                        
                        if os.path.exists(orig_txt_path):
                            with open(orig_txt_path, 'r', encoding='utf-8') as f:
                                orig_text = f.read().strip()
                        
                        all_data.append((audio_id, norm_text, orig_text, subset))
        
        print(f"  Found {len(all_data)} total audio files so far")
    
    return all_data


def create_libritts_split(
    data: List[Tuple[str, str, str, str]],
    test_ratio: float = 0.05,
    val_ratio: float = 0.1,
    seed: int = 42
) -> dict:
    """
    Split LibriTTS dataset into train/val/test sets.
    
    Args:
        data: List of (audio_id, norm_text, orig_text, subset) tuples
        test_ratio: Ratio for test set
        val_ratio: Ratio for validation set
        seed: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test' lists
    """
    random.seed(seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    total = len(shuffled_data)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)
    
    test_data = shuffled_data[:test_count]
    val_data = shuffled_data[test_count:test_count + val_count]
    train_data = shuffled_data[test_count + val_count:]
    
    return {
        'train': [(d[0], d[1]) for d in train_data],  # (audio_id, norm_text)
        'val': [(d[0], d[1]) for d in val_data],
        'test': [(d[0], d[1]) for d in test_data],
    }


def save_libritts_splits(splits: dict, output_dir: str):
    """Save splits to text files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, items in splits.items():
        output_path = os.path.join(output_dir, f'{split_name}.txt')
        with open(output_path, 'w') as f:
            for audio_id, text in items:
                f.write(f"{audio_id}|{text}\n")
        print(f"Saved {len(items)} samples to {output_path}")


def find_audio_file(audio_id: str, libritts_root: str) -> str:
    """
    Find the audio file path for a given audio_id in LibriTTS.
    
    Args:
        audio_id: Audio ID (e.g., '103_1241_000000_000001')
        libritts_root: Root directory of LibriTTS
        
    Returns:
        Full path to the audio file, or None if not found
    """
    # Audio ID format: speaker_id_chapter_id_start_end
    parts = audio_id.split('_')
    if len(parts) >= 4:
        speaker_id = parts[0]
        chapter_id = parts[1]
        
        # Search through subsets
        for subset in ['train-clean-100', 'train-clean-360', 'train-other-500',
                       'dev-clean', 'dev-other', 'test-clean', 'test-other']:
            audio_path = os.path.join(libritts_root, subset, speaker_id, chapter_id, f"{audio_id}.wav")
            if os.path.exists(audio_path):
                return audio_path
    
    return None


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


def process_libritts_samples(
    libritts_root: str,
    output_dir: str,
    config_path: str,
    splits: dict,
    split_name: str = 'train',
    max_samples: int = None
):
    """Process LibriTTS samples through the full pipeline."""
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
    items_to_process = splits[split_name]
    
    if max_samples:
        items_to_process = items_to_process[:max_samples]
    
    print(f"Processing {len(items_to_process)} {split_name} samples...")
    print(f"Using MEL extractor: {mel_extractor}")
    
    # Create output directories
    split_output_dir = os.path.join(output_dir, split_name)
    audio_output_dir = os.path.join(split_output_dir, 'audio')
    mel_output_dir = os.path.join(split_output_dir, 'mel')
    os.makedirs(split_output_dir, exist_ok=True)
    os.makedirs(audio_output_dir, exist_ok=True)
    os.makedirs(mel_output_dir, exist_ok=True)
    
    # Check for existing processed files
    existing_files = set(os.listdir(audio_output_dir)) if os.path.exists(audio_output_dir) else set()
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
    
    for i, (audio_id, text) in enumerate(tqdm(items_to_process)):
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
        
        # Find audio file
        audio_path = find_audio_file(audio_id, libritts_root)
        
        if not audio_path:
            errors.append({'audio_id': audio_id, 'error': 'File not found'})
            continue
        
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
            
            print(f"Progress: {i + 1}/{len(items_to_process)} - Success: {success_count}")
    
    # Save final results
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")
    
    # Save statistics
    stats = {
        'split': split_name,
        'total': len(items_to_process),
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
    print(f"  Total: {len(items_to_process)}")
    print(f"  Success: {success_count}")
    print(f"  Errors: {len(errors)}")
    
    return all_results


def main():
    """Run the LibriTTS dataset preprocessing."""
    config_path = '/workspace/ljspeech-vocos/config/pipeline.yaml'
    libritts_root = '/data/tts/datasets/LibriTTS'
    output_dir = '/data/tts/outputs/libritts_pipeline'
    splits_dir = '/data/tts/processed/libritts_splits'
    
    print("=" * 60)
    print("LibriTTS Dataset Preprocessing")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"LibriTTS root: {libritts_root}")
    print(f"Output: {output_dir}")
    print()
    
    # Step 1: Scan dataset
    print("Step 1: Scanning LibriTTS dataset...")
    # For demo, use a small subset first
    all_data = scan_libritts_dataset(
        libritts_root=libritts_root,
        subset_dirs=['train-clean-100']  # Start with smaller subset
    )
    
    print(f"\nTotal audio files found: {len(all_data)}")
    
    # Step 2: Create splits
    print("\nStep 2: Creating data splits...")
    splits = create_libritts_split(
        data=all_data,
        test_ratio=0.05,
        val_ratio=0.1,
        seed=42
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val: {len(splits['val'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    print(f"  Total: {len(splits['train']) + len(splits['val']) + len(splits['test'])} samples")
    
    save_libritts_splits(splits, splits_dir)
    
    # Step 3: Process test samples (small sample for demo)
    print("\n" + "=" * 60)
    print("Step 3: Processing TEST samples...")
    print("=" * 60)
    
    test_results = process_libritts_samples(
        libritts_root=libritts_root,
        output_dir=output_dir,
        config_path=config_path,
        splits=splits,
        split_name='test',
        max_samples=10  # Process only 10 for demo
    )
    
    # Step 4: Process train samples (small sample for demo)
    print("\n" + "=" * 60)
    print("Step 4: Processing TRAIN samples...")
    print("=" * 60)
    
    train_results = process_libritts_samples(
        libritts_root=libritts_root,
        output_dir=output_dir,
        config_path=config_path,
        splits=splits,
        split_name='train',
        max_samples=20  # Process only 20 for demo
    )
    
    # Step 5: Generate summary
    print("\n" + "=" * 60)
    print("Step 5: Generating summary...")
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
        'dataset': 'LibriTTS',
        'subset': 'train-clean-100',
        'total_samples': len(all_data),
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
        'splits_directory': splits_dir
    }
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\nDataset: LibriTTS (train-clean-100)")
    print(f"Total samples: {len(all_data)}")
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
    print(f"Splits directory: {splits_dir}")
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
