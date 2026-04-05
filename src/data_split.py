"""
Data splitting utility for TTS dataset.
Splits LJSpeech dataset into train/val/test with fixed test size.
"""

import os
import random
import yaml
from typing import List, Tuple, Dict
from pathlib import Path


def load_metadata(metadata_path: str) -> List[Tuple[str, str, str]]:
    """
    Load metadata from LJSpeech CSV file.
    
    Args:
        metadata_path: Path to metadata.csv
        
    Returns:
        List of (id, raw_text, norm_text) tuples
    """
    data = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                audio_id, raw_text, norm_text = parts[0], parts[1], parts[2]
                data.append((audio_id, raw_text, norm_text))
    return data


def split_dataset(
    metadata_path: str,
    test_samples: int = 100,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        metadata_path: Path to metadata.csv
        test_samples: Number of samples for test set (fixed)
        val_ratio: Ratio of validation set from remaining data
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' lists of audio IDs
    """
    # Load metadata
    data = load_metadata(metadata_path)
    audio_ids = [item[0] for item in data]
    
    # Shuffle with seed
    random.seed(seed)
    shuffled_ids = audio_ids.copy()
    random.shuffle(shuffled_ids)
    
    # Split: first test_samples go to test
    test_ids = shuffled_ids[:test_samples]
    remaining = shuffled_ids[test_samples:]
    
    # Split remaining into train/val
    val_count = int(len(remaining) * val_ratio)
    val_ids = remaining[:val_count]
    train_ids = remaining[val_count:]
    
    return {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids,
    }


def save_splits(splits: Dict[str, List[str]], output_dir: str):
    """
    Save splits to text files.
    
    Args:
        splits: Dictionary with train/val/test lists
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, ids in splits.items():
        output_path = os.path.join(output_dir, f'{split_name}.txt')
        with open(output_path, 'w') as f:
            for audio_id in ids:
                f.write(f"{audio_id}\n")
        print(f"Saved {len(ids)} samples to {output_path}")


def main():
    """Create data splits for LJSpeech."""
    # Configuration
    metadata_path = '/data/tts/datasets/LJSpeech-1.1/metadata.csv'
    output_dir = '/data/tts/processed/splits'
    
    # Load config if available
    config_path = '/workspace/tts/config/pipeline.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        data_split_config = config.get('data_split', {})
        test_samples = data_split_config.get('test_samples', 100)
        val_ratio = data_split_config.get('val_ratio', 0.1)
    else:
        test_samples = 100
        val_ratio = 0.1
    
    print(f"Loading metadata from: {metadata_path}")
    print(f"Test samples: {test_samples}")
    print(f"Validation ratio: {val_ratio}")
    
    # Split dataset
    splits = split_dataset(
        metadata_path=metadata_path,
        test_samples=test_samples,
        val_ratio=val_ratio,
        seed=42
    )
    
    # Print statistics
    print(f"\nDataset splits:")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val: {len(splits['val'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    print(f"  Total: {len(splits['train']) + len(splits['val']) + len(splits['test'])} samples")
    
    # Save splits
    save_splits(splits, output_dir)
    print(f"\nSplits saved to: {output_dir}")


if __name__ == '__main__':
    main()
