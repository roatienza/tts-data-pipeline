# Text → IPA → Reconstructed Text Samples

This document shows 3 samples demonstrating the text processing pipeline: **Text → IPA → Reconstructed Text**.

## Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Input Text    │────▶│   IPA (phonemizer) │────▶│ Reconstructed Text │
│  (Normalized)   │     │                   │     │  (from IPA)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Sample 1: Long Sentence from LJSpeech

### Original Text
```
Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition
```

### Normalized Text
```
printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the exhibition
```

### IPA (International Phonetic Alphabet)
```
pɹˈɛntɪŋ, ɪnðɪ ˈoːnli sˈɛns wɪð wɪtːʃ wiː ɑːɹ æt pɹˈɛzənt kənsˈɜːnd, dˈɪfɚz fɹʊm mˈoːst ɪf nɜːt fɹʊm ˈɔːl ði ˈɑːːɹts ænd kɹˈæfts ɹɪbpɹɪˈzɛntɪd ɪnðɪ ɛksɪbˈɪʃən
```

### Reconstructed Text (from IPA)
```
printing, inthi oːunli sens with witːsh wi ɑr at prezent kensernd, diferz frum moːust if nɑt frum ol thi ɑːrts and krafts reprɪzentɪd inthi eksibishen
```

### Statistics
| Metric | Value |
|--------|-------|
| Original length | 151 chars |
| Normalized length | 151 chars |
| IPA length | 162 chars |
| Reconstructed length | 150 chars |

---

## Sample 2: Text with Numbers and Currency

### Original Text
```
The price is $123.45 for this item.
```

### Normalized Text
```
the price is $one hundred and twenty-three point 4 point 5 for this item.
```

### IPA (International Phonetic Alphabet)
```
ðə pɹˈɑːɪs ɑz dˈɑːlɚ wˈʌn hˈʌndɹəd ænd twˈɛntiθɹˈiː pˈɔːɪnt fˈoːːɹ pˈɔːɪnt fˈɑːɪv fɔːːɹ ðəs ˈɑːɪtəm.
```

### Reconstructed Text (from IPA)
```
the praːis iz dɑler wun hundrid and twentithri poːint foːr poːint faːiv foːr this aːirem.
```

### Statistics
| Metric | Value |
|--------|-------|
| Original length | 35 chars |
| Normalized length | 73 chars |
| IPA length | 100 chars |
| Reconstructed length | 89 chars |

---

## Sample 3: Text with Abbreviations and Numbers

### Original Text
```
Dr. Smith is a professor at MIT and has published over 50 papers.
```

### Normalized Text
```
dr. smith is a professor at mit and has published over fifty papers.
```

### IPA (International Phonetic Alphabet)
```
dˈɑːktɚ. smˈɪθ ɑz ʌ pɹɛfˈɛsɚɹ æt mˈɪt ænd hʌz pˈʌblɪʃt ˌoːːvɚ fˈɪfti pˈeːpɚz.
```

### Reconstructed Text (from IPA)
```
dɑkter. smith iz ʌ prefeserr at mit and hʌz publisht oːuver fifti peːiperz.
```

### Statistics
| Metric | Value |
|--------|-------|
| Original length | 65 chars |
| Normalized length | 68 chars |
| IPA length | 78 chars |
| Reconstructed length | 75 chars |

---

## Notes

1. **Text Normalization**: Always applied before phonemization
   - Lowercase conversion
   - Abbreviation expansion (e.g., "Dr." → "doctor")
   - Number conversion (e.g., "123" → "one hundred and twenty-three")

2. **IPA Conversion**: Uses phonemizer with espeak backend
   - Preserves punctuation
   - Includes stress markers (ˈ, ˌ)
   - Uses tie bars (ː) for connected sounds

3. **Reconstruction**: The IPA → Text reconstruction is approximate
   - Uses a simplified inverse mapping
   - Not perfect but demonstrates the concept
   - Real IPA-to-text would require more complex alignment

## Running the Demo

```bash
python src/text_to_ipa_demo.py
```

This will display all 3 samples with their transformations.
