"""
Text Pipeline: Text → IPA → Text
Uses phonemizer for IPA conversion and custom normalization
"""

import re
from typing import Tuple, Dict, List, Optional
import yaml
from phonemizer import phonemize
from inflect import engine


class TextPipeline:
    """Text normalization and phonemization pipeline."""
    
    def __init__(self, config_path: str = None):
        """Initialize text pipeline with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            text_config = config.get('text', {})
        else:
            text_config = {}
        
        # Text parameters
        self.language = text_config.get('language', 'en-us')
        self.normalize = text_config.get('normalize', True)
        self.lowercase = text_config.get('lowercase', True)
        self.expand_abbreviations = text_config.get('expand_abbreviations', True)
        self.convert_numbers = text_config.get('convert_numbers', True)
        
        # Phonemizer parameters
        phonemizer_config = text_config.get('phonemizer', {})
        self.backend = phonemizer_config.get('backend', 'espeak')
        self.preserve_punctuation = phonemizer_config.get('preserve_punctuation', True)
        self.with_stress = phonemizer_config.get('with_stress', True)
        self.tie = phonemizer_config.get('tie', True)
        
        # Initialize inflect for number conversion
        self.inflect_engine = engine()
        
        # Abbreviation mappings
        self.abbreviations = {
            'dr.': 'doctor',
            'mr.': 'mister',
            'mrs.': 'missus',
            'ms.': 'miss',
            'prof.': 'professor',
            'sr.': 'senior',
            'jr.': 'junior',
            'vs.': 'versus',
            'etc.': 'etcetera',
            'i.e.': 'that is',
            'e.g.': 'for example',
            'approx.': 'approximately',
            'avg.': 'average',
            'addr.': 'address',
            'art.': 'article',
            'bldg.': 'building',
            'corp.': 'corporation',
            'dept.': 'department',
            'est.': 'established',
            'fig.': 'figure',
            'inc.': 'incorporated',
            'ltd.': 'limited',
            'no.': 'number',
            'phd': 'doctor of philosophy',
            'ph.d.': 'doctor of philosophy',
            'st.': 'saint',
            'ave.': 'avenue',
            'blvd.': 'boulevard',
            'rd.': 'road',
            'st.': 'street',
            'hwy.': 'highway',
            'cnty.': 'county',
            'gov.': 'government',
            'pres.': 'president',
            'rep.': 'representative',
            'sen.': 'senator',
            'rev.': 'reverend',
            'hon.': 'honor',
            'gen.': 'general',
            'col.': 'colonel',
            'maj.': 'major',
            'lt.': 'lieutenant',
            'sgt.': 'sergeant',
            'pvt.': 'private',
            'cpl.': 'corporal',
            'ens.': 'ensign',
            'capt.': 'captain',
            'admiral': 'admiral',
            'govt.': 'government',
            'mon.': 'monday',
            'tue.': 'tuesday',
            'wed.': 'wednesday',
            'thu.': 'thursday',
            'fri.': 'friday',
            'sat.': 'saturday',
            'sun.': 'sunday',
            'jan.': 'january',
            'feb.': 'february',
            'mar.': 'march',
            'apr.': 'april',
            'may': 'may',
            'jun.': 'june',
            'jul.': 'july',
            'aug.': 'august',
            'sep.': 'september',
            'oct.': 'october',
            'nov.': 'november',
            'dec.': 'december',
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for phonemization.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not self.normalize:
            return text
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Expand abbreviations
        if self.expand_abbreviations:
            for abbr, full in self.abbreviations.items():
                # Match abbreviation with period or standalone
                pattern = r'\b' + re.escape(abbr) + r'\b'
                text = re.sub(pattern, full, text)
        
        # Convert numbers to words
        if self.convert_numbers:
            text = self._convert_numbers(text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _convert_numbers(self, text: str) -> str:
        """Convert numbers to words using inflect.
        
        Args:
            text: Input text with numbers
            
        Returns:
            Text with numbers converted to words
        """
        # Find all numbers (integers and decimals)
        def replace_number(match):
            num_str = match.group(0)
            try:
                if '.' in num_str:
                    # Decimal number
                    parts = num_str.split('.')
                    integer_part = self.inflect_engine.number_to_words(int(parts[0]))
                    decimal_part = ' point '.join([str(d) for d in parts[1]])
                    return f"{integer_part} point {decimal_part}"
                else:
                    # Integer
                    return self.inflect_engine.number_to_words(int(num_str))
            except:
                return num_str
        
        # Match integers and decimals
        text = re.sub(r'\b\d+\.?\d*\b', replace_number, text)
        
        return text
    
    def text_to_ipa(self, text: str) -> str:
        """Convert normalized text to IPA using phonemizer.
        
        Args:
            text: Input text (should be normalized)
            
        Returns:
            IPA representation
        """
        # Normalize first
        normalized = self.normalize_text(text)
        
        # Phonemize
        ipa = phonemize(
            normalized,
            language=self.language,
            backend=self.backend,
            preserve_punctuation=self.preserve_punctuation,
            with_stress=self.with_stress,
            tie=self.tie
        )
        
        return ipa.strip()
    
    def validate_ipa(self, ipa: str) -> Dict:
        """Validate IPA output.
        
        Args:
            ipa: IPA string
            
        Returns:
            Validation results dictionary
        """
        # Valid IPA characters (basic set for English)
        valid_ipa_chars = set(
            'aɪəɒɔæɛɜɪʊθðŋʃʒʔˈˌʔɑɒɔæɛɜɪʊθðŋʃʒˈˌ0123456789'
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '.,!?;:\'"()[]{}-—– '
        )
        
        invalid_chars = set(ipa) - valid_ipa_chars
        
        return {
            'length': len(ipa),
            'has_invalid_chars': len(invalid_chars) > 0,
            'invalid_chars': list(invalid_chars) if invalid_chars else [],
            'valid': len(invalid_chars) == 0,
        }
    
    def process_text(self, text: str) -> Dict:
        """Process text through the full pipeline.
        
        Args:
            text: Input text
            
        Returns:
            Processing results dictionary
        """
        # Normalize
        normalized = self.normalize_text(text)
        
        # Convert to IPA
        ipa = self.text_to_ipa(text)
        
        # Validate IPA
        ipa_validation = self.validate_ipa(ipa)
        
        return {
            'original': text,
            'normalized': normalized,
            'ipa': ipa,
            'ipa_validation': ipa_validation,
        }


def main():
    """Test the text pipeline."""
    # Initialize pipeline
    config_path = '/workspace/tts/config/pipeline.yaml'
    pipeline = TextPipeline(config_path)
    
    # Test with sample text
    test_texts = [
        "Hello, world!",
        "The price is $123.45.",
        "Dr. Smith is a professor at MIT.",
        "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition",
    ]
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        result = pipeline.process_text(text)
        print(f"Normalized: {result['normalized']}")
        print(f"IPA: {result['ipa']}")
        print(f"Validation: {result['ipa_validation']}")


if __name__ == '__main__':
    main()
