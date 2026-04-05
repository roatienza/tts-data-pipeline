"""
Complete preprocessing - process all remaining train samples.
"""

import os
import sys
import yaml
import json
import gc

os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, '/workspace/tts/src')

from audio_pipeline import AudioPipeline
from text_pipeline import TextPipeline
from tqdm import tqdm

def main():
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
    
    # Load metadata
    data = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 4:
                audio_id, raw_text, norm_text, split = parts[0], parts[1], parts[2], parts[3]
                data.append((audio_id, raw_text, norm_text, split))
    
    train_data = [(d[0], d[1]) for d in data if d[3] == 'train']
    train_ids = [d[0] for d in train_data]
    text_lookup = {d[0]: d[1] for d in train_data}
    
    print(f"Total train samples: {len(train_ids)}")
    
    audio_output_dir = os.path.join(output_dir, 'train', 'audio')
    mel_output_dir = os.path.join(output_dir, 'train', 'mel')
    existing_files = set(os.listdir(audio_output_dir))
    print(f"Existing files: {len(existing_files)}")
    
    # Load existing results
    results_path = os.path.join(output_dir, 'train', 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            all_results = json.load(f)
        existing_ids = set(r.get('audio_id') for r in all_results)
    else:
        all_results = []
        existing_ids = set()
    
    ids_to_process = [id for id in train_ids if f"{id}.wav" not in existing_files and id not in existing_ids]
    print(f"Remaining to process: {len(ids_to_process)}")
    
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
        
        if (i + 1) % 500 == 0:
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            gc.collect()
            print(f"Progress: {i + 1}/{len(ids_to_process)} - Success: {success_count}")
    
    # Save final results
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    stats = {
        'split': 'train',
        'total': len(train_ids),
        'success': success_count,
        'errors': len(errors),
    }
    with open(os.path.join(output_dir, 'train', 'statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    if errors:
        with open(os.path.join(output_dir, 'train', 'errors.json'), 'w') as f:
            json.dump(errors, f, indent=2)
    
    # Load test stats
    with open(os.path.join(output_dir, 'test', 'statistics.json'), 'r') as f:
        test_stats = json.load(f)
    
    # Create summary
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
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTrain Processing Complete:")
    print(f"  Total: {len(train_ids)}")
    print(f"  Success: {success_count}")
    print(f"  Errors: {len(errors)}")
    print(f"\nSummary saved")


if __name__ == '__main__':
    main()
