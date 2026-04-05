"""
Text → IPA → Reconstructed Text Demo
Shows 3 samples of text input, IPA conversion, and reconstructed text from IPA
"""

import re
from phonemizer import phonemize
from inflect import engine


class TextToIPADemo:
    """Text normalization, phonemization, and IPA to text reconstruction."""
    
    def __init__(self, language="en-us"):
        self.language = language
        self.inflect_engine = engine()
        
        # Abbreviation mappings
        self.abbreviations = {
            'dr.': 'doctor', 'mr.': 'mister', 'mrs.': 'missus', 'ms.': 'miss',
            'prof.': 'professor', 'sr.': 'senior', 'jr.': 'junior', 'vs.': 'versus',
            'etc.': 'etcetera', 'i.e.': 'that is', 'e.g.': 'for example',
            'approx.': 'approximately', 'avg.': 'average', 'dept.': 'department',
            'est.': 'established', 'fig.': 'figure', 'inc.': 'incorporated',
            'ltd.': 'limited', 'no.': 'number', 'st.': 'street', 'ave.': 'avenue',
            'blvd.': 'boulevard', 'rd.': 'road', 'hwy.': 'highway',
        }
        
        # IPA to text mapping (simplified inverse mapping)
        # This is a simplified mapping - real IPA-to-text would require more complex alignment
        self.ipa_to_char = {
            # Vowels
            'a': 'a', 'æ': 'a', 'e': 'e', 'ɛ': 'e', 'i': 'i', 'ɪ': 'i',
            'o': 'o', 'ɔ': 'o', 'u': 'u', 'ʊ': 'u', 'ʌ': 'u', 'ɒ': 'o',
            'ɜ': 'er', 'ə': 'e', 'ɝ': 'er', 'ɚ': 'er', 'ʊ': 'u', 'ʒ': 's',
            # Consonants
            'b': 'b', 't': 't', 'd': 'd', 'k': 'k', 'g': 'g',
            'p': 'p', 'f': 'f', 'v': 'v', 'θ': 'th', 'ð': 'th',
            's': 's', 'z': 'z', 'ʃ': 'sh', 'ʒ': 's', 'tʃ': 'ch', 'dʒ': 'j',
            'm': 'm', 'n': 'n', 'ŋ': 'ng', 'l': 'l', 'r': 'r', 'w': 'w',
            'j': 'y', 'h': 'h', 'ʔ': '', 'ɹ': 'r', 'ɾ': 'r',
            # Stress markers and other symbols
            'ˈ': '', 'ˌ': '', 'ː': '', '‿': '', '‿': ' ', ' ': ' ',
            # Punctuation
            '.': '.', ',': ',', '!': '!', '?': '?', "'": "'",
            '"': '"', '(': '(', ')': ')', '-': '-',
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for phonemization."""
        # Lowercase
        text = text.lower()
        
        # Expand abbreviations
        for abbr, full in self.abbreviations.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text = re.sub(pattern, full, text)
        
        # Convert numbers to words
        def replace_number(match):
            num_str = match.group(0)
            try:
                if '.' in num_str:
                    parts = num_str.split('.')
                    integer_part = self.inflect_engine.number_to_words(int(parts[0]))
                    decimal_part = ' point '.join([str(d) for d in parts[1]])
                    return f"{integer_part} point {decimal_part}"
                else:
                    return self.inflect_engine.number_to_words(int(num_str))
            except:
                return num_str
        
        text = re.sub(r'\b\d+\.?\d*\b', replace_number, text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def text_to_ipa(self, text: str) -> str:
        """Convert normalized text to IPA using phonemizer."""
        normalized = self.normalize_text(text)
        
        ipa = phonemize(
            normalized,
            language=self.language,
            backend='espeak',
            preserve_punctuation=True,
            with_stress=True,
            tie=True
        )
        
        return ipa.strip()
    
    def ipa_to_text(self, ipa: str) -> str:
        """
        Reconstruct approximate text from IPA.
        This is a simplified inverse mapping - not perfect but demonstrates the concept.
        """
        reconstructed = []
        i = 0
        
        while i < len(ipa):
            char = ipa[i]
            
            # Check for multi-character IPA symbols (like tʃ, dʒ)
            if i + 1 < len(ipa):
                two_char = char + ipa[i+1]
                if two_char in self.ipa_to_char:
                    reconstructed.append(self.ipa_to_char[two_char])
                    i += 2
                    continue
            
            # Single character mapping
            if char in self.ipa_to_char:
                reconstructed.append(self.ipa_to_char[char])
            else:
                # Keep unknown characters as-is
                reconstructed.append(char)
            
            i += 1
        
        result = ''.join(reconstructed)
        
        # Clean up the result
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def process_sample(self, text: str) -> dict:
        """Process a single text sample through the full pipeline."""
        normalized = self.normalize_text(text)
        ipa = self.text_to_ipa(text)
        reconstructed = self.ipa_to_text(ipa)
        
        return {
            'original': text,
            'normalized': normalized,
            'ipa': ipa,
            'reconstructed': reconstructed,
        }


def main():
    """Demonstrate text → IPA → reconstructed text with 3 samples."""
    demo = TextToIPADemo(language='en-us')
    
    # Select 3 sample texts from LJSpeech metadata
    sample_texts = [
        "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition",
        "The price is $123.45 for this item.",
        "Dr. Smith is a professor at MIT and has published over 50 papers.",
    ]
    
    print("=" * 100)
    print("TEXT → IPA → RECONSTRUCTED TEXT DEMONSTRATION")
    print("=" * 100)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{'='*100}")
        print(f"SAMPLE {i}")
        print(f"{'='*100}")
        
        result = demo.process_sample(text)
        
        print(f"\n📝 ORIGINAL TEXT:")
        print(f"   {result['original']}")
        
        print(f"\n🔄 NORMALIZED TEXT:")
        print(f"   {result['normalized']}")
        
        print(f"\n🔤 IPA (International Phonetic Alphabet):")
        print(f"   {result['ipa']}")
        
        print(f"\n↩️  RECONSTRUCTED TEXT (from IPA):")
        print(f"   {result['reconstructed']}")
        
        print(f"\n📊 COMPARISON:")
        print(f"   Original length:    {len(result['original'])} chars")
        print(f"   Normalized length:  {len(result['normalized'])} chars")
        print(f"   IPA length:         {len(result['ipa'])} chars")
        print(f"   Reconstructed:      {len(result['reconstructed'])} chars")
    
    print(f"\n{'='*100}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
