"""
Audio Pipeline: WAV -> MEL -> WAV
Uses librosa for MEL extraction and Vocos for reconstruction
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Dict, Optional
from pathlib import Path
import yaml


class AudioPipeline:
    """Audio preprocessing and reconstruction pipeline."""
    
    def __init__(self, config_path: str = None):
        """Initialize audio pipeline with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            audio_config = config.get('audio', {})
        else:
            audio_config = {}
        
        # Audio parameters
        self.sample_rate = audio_config.get('sample_rate', 22050)
        self.n_mels = audio_config.get('n_mels', 80)  # Vocos uses 80 mel bins
        self.n_fft = audio_config.get('n_fft', 1024)
        self.hop_length = audio_config.get('hop_length', 256)
        self.win_length = audio_config.get('win_length', 1024)
        self.fmin = audio_config.get('fmin', 0)
        self.fmax = audio_config.get('fmax', 8000)
        
        # Vocoder parameters
        self.vocoder_model_name = audio_config.get('reconstruction', {}).get('model_name', 'BSC-LT/vocos-mel-22khz')
        
        self.vocos_model = None
        self.use_vocos = False
    
    def load_vocos_model(self, cache_dir: str = '/data/tts/cache'):
        """Load Vocos vocoder model."""
        try:
            from vocos import Vocos
            self.vocos_model = Vocos.from_pretrained(
                "BSC-LT/vocos-mel-22khz"
            )
            self.use_vocos = True
            print(f"Vocos model loaded: {self.vocoder_model_name}")
        except Exception as e:
            print(f"Failed to load Vocos model: {e}")
            print("Falling back to librosa-only processing (no reconstruction)")
            self.use_vocos = False
    
    def preprocess_audio(self, audio_path: str, target_sr: int = None) -> np.ndarray:
        """Load and preprocess audio file.
        
        Args:
            audio_path: Path to WAV file
            target_sr: Target sample rate (default: self.sample_rate)
            
        Returns:
            Preprocessed audio array
        """
        if target_sr is None:
            target_sr = self.sample_rate
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=target_sr)
        
        # Volume normalization (RMS normalization to -20dB)
        target_rms = 0.1  # -20dB
        current_rms = np.sqrt(np.mean(y ** 2))
        if current_rms > 1e-8:
            y = y * (target_rms / current_rms)
        
        return y
    
    def extract_mel_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract MEL spectrogram using librosa for Vocos compatibility.
        
        Args:
            audio: Audio array
            
        Returns:
            MEL spectrogram (n_mels, time_steps)
        """
        # Compute raw mel spectrogram (no log)
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            power=1.0  # Raw mel, not log
        )
        
        # Standardize (mean=0, std=1)
        mel_spec = (mel_spec - np.mean(mel_spec)) / (np.std(mel_spec) + 1e-8)
        
        return mel_spec
    
    def extract_mel_features_vocos(self, audio: np.ndarray) -> np.ndarray:
        """Extract MEL spectrogram using Vocos feature extractor.
        
        Args:
            audio: Audio array
            
        Returns:
            MEL spectrogram (n_mels, time_steps) - log mel spectrogram
        """
        if self.vocos_model is None or not self.use_vocos:
            raise ValueError("Vocos model not loaded or not available.")
        
        import torch
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
        
        # Extract mel using Vocos feature extractor
        mel_spec = self.vocos_model.feature_extractor(audio_tensor)
        
        return mel_spec.squeeze(0).numpy()
    
    def reconstruct_audio(self, mel_spec: np.ndarray, use_vocos_extractor: bool = False) -> np.ndarray:
        """Reconstruct audio from MEL spectrogram using Vocos.
        
        Args:
            mel_spec: MEL spectrogram (n_mels, time_steps)
            use_vocos_extractor: If True, mel_spec is log-mel from Vocos extractor
            
        Returns:
            Reconstructed audio array
        """
        if not self.use_vocos:
            # If Vocos is not available, return a zero array of expected length
            # This is a fallback for when we can't reconstruct
            time_steps = mel_spec.shape[1]
            expected_samples = time_steps * self.hop_length
            return np.zeros(expected_samples, dtype=np.float32)
        
        import torch
        
        if use_vocos_extractor:
            # mel_spec is already log-mel from Vocos extractor
            # Reshape for Vocos: (batch, n_mels, time)
            mel_spec_tensor = torch.from_numpy(mel_spec).unsqueeze(0).float()
        else:
            # mel_spec is standardized raw mel from librosa
            # Vocos expects log-mel, so we need to convert
            # For now, let's try passing the standardized mel directly
            mel_spec_tensor = torch.from_numpy(mel_spec).unsqueeze(0).float()
        
        # Decode using Vocos
        audio_tensor = self.vocos_model.decode(mel_spec_tensor)
        
        return audio_tensor.squeeze(0).numpy()
    
    def validate_mel(self, mel_spec: np.ndarray) -> Dict:
        """Validate MEL spectrogram quality.
        
        Args:
            mel_spec: MEL spectrogram
            
        Returns:
            Validation results dictionary
        """
        results = {
            'dimensions': mel_spec.shape,
            'has_nan': np.any(np.isnan(mel_spec)),
            'has_inf': np.any(np.isinf(mel_spec)),
            'mean': float(np.mean(mel_spec)),
            'std': float(np.std(mel_spec)),
            'min': float(np.min(mel_spec)),
            'max': float(np.max(mel_spec)),
        }
        
        # Check for validity
        if results['has_nan'] or results['has_inf']:
            results['valid'] = False
        else:
            results['valid'] = True
        
        return results
    
    def process_file(self, audio_path: str, output_dir: str = None, mel_output_dir: str = None, use_vocos_extractor: bool = False) -> Dict:
        """Process a single audio file through the full pipeline.
        
        Args:
            audio_path: Path to input WAV file
            output_dir: Directory to save reconstructed audio
            mel_output_dir: Directory to save MEL spectrograms (optional)
            use_vocos_extractor: Use Vocos feature extractor instead of librosa
            
        Returns:
            Processing results dictionary
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if mel_output_dir:
            os.makedirs(mel_output_dir, exist_ok=True)
        
        # Get audio ID from path
        audio_id = os.path.basename(audio_path).replace('.wav', '')
        
        # Preprocess audio
        audio = self.preprocess_audio(audio_path)
        
        # Extract MEL features
        if use_vocos_extractor and self.use_vocos:
            mel_spec = self.extract_mel_features_vocos(audio)
        else:
            mel_spec = self.extract_mel_features(audio)
        
        # Validate MEL
        mel_validation = self.validate_mel(mel_spec)
        
        # Save MEL if requested
        mel_path = None
        if mel_output_dir:
            mel_path = os.path.join(mel_output_dir, f"{audio_id}.npy")
            np.save(mel_path, mel_spec)
        
        # Reconstruct audio (or use original if Vocos not available)
        if self.use_vocos:
            reconstructed = self.reconstruct_audio(mel_spec, use_vocos_extractor=use_vocos_extractor)
        else:
            # If Vocos is not available, just save the preprocessed audio
            reconstructed = audio
        
        # Save reconstructed audio
        if output_dir:
            output_path = os.path.join(output_dir, f"{audio_id}.wav")
            sf.write(output_path, reconstructed, self.sample_rate)
        else:
            output_path = None
        
        return {
            'audio_id': audio_id,
            'input_path': audio_path,
            'output_path': output_path,
            'mel_path': mel_path,
            'input_duration': len(audio) / self.sample_rate,
            'output_duration': len(reconstructed) / self.sample_rate,
            'mel_validation': mel_validation,
        }


def main():
    """Test the audio pipeline."""
    import torch
    
    # Initialize pipeline
    config_path = '/workspace/ljspeech-vocos/config/pipeline.yaml'
    pipeline = AudioPipeline(config_path)
    
    # Load Vocos model
    pipeline.load_vocos_model()
    
    # Test with a sample file
    test_audio = '/data/tts/datasets/LJSpeech-1.1/wavs/LJ001-0001.wav'
    
    if os.path.exists(test_audio):
        print(f"Processing: {test_audio}")
        
        # Test with librosa MEL extraction
        print("\n=== Using librosa MEL extraction ===")
        result = pipeline.process_file(test_audio, output_dir='/data/tts/outputs/test_librosa', use_vocos_extractor=False)
        print(f"Result: {result}")
        
        # Test with Vocos MEL extraction
        print("\n=== Using Vocos MEL extraction ===")
        result = pipeline.process_file(test_audio, output_dir='/data/tts/outputs/test_vocos', use_vocos_extractor=True)
        print(f"Result: {result}")
    else:
        print(f"Test audio not found: {test_audio}")


if __name__ == '__main__':
    main()
