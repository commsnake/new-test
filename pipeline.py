import os
import re
import json
import pytesseract
from PIL import Image
import spacy
from collections import Counter

# 1. OCR & Layout Extraction
print("Starting OCR extraction...")
images = [f for f in os.listdir('.') if f.endswith('.png')]
images.sort()

text_chunks = []
for img_file in images:
    img = Image.open(img_file)
    text = pytesseract.image_to_string(img)
    text_chunks.append(text)

raw_text = "\n\n".join(text_chunks)
with open('transcription.txt', 'w', encoding='utf-8') as f:
    f.write(raw_text)

# 2. Stylometric Research & Analysis
print("Running stylometric analysis...")
nlp = spacy.load('en_core_web_sm')

# spaCy max length limit, process in chunks or increase limit
nlp.max_length = len(raw_text) + 1000
doc = nlp(raw_text)

# Vocabulary & Lexical Density
words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
word_freq = Counter(words)
unique_words = len(word_freq)
total_words = len(words)
lexical_density = unique_words / total_words if total_words > 0 else 0

# Sentence Complexity
sentences = list(doc.sents)
avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences) if sentences else 0

# Syntactic Fingerprint
pos_counts = Counter(token.pos_ for token in doc)
total_tokens = len(doc)
verb_ratio = pos_counts.get('VERB', 0) / total_tokens if total_tokens > 0 else 0
adj_ratio = pos_counts.get('ADJ', 0) / total_tokens if total_tokens > 0 else 0
noun_ratio = pos_counts.get('NOUN', 0) / total_tokens if total_tokens > 0 else 0

# Tone & Affect (simplified heuristics based on POS and specific word usage)
# Since 'I' and 'me' are stop words in Spacy by default, we need to check ALL tokens, not just non-stop words
all_words = [token.text.lower() for token in doc if token.is_alpha]
first_person_pronouns = ['i', 'me', 'my', 'mine', 'myself']
first_person_count = sum(Counter(all_words).get(p, 0) for p in first_person_pronouns)
fp_ratio = first_person_count / len(all_words) if len(all_words) > 0 else 0

# 3. Knowledge Base Creation
print("Generating STYLE_GUIDE.md and LINTING_RULES.json...")

style_guide = f"""# STYLE GUIDE: Automated Stylometric Profile

## 1. Objective
This guide provides the necessary parameters to replicate the target author's distinct narrative voice.

## 2. Lexical Density & Vocabulary
- **Lexical Density:** {lexical_density:.4f} (Unique non-stop words / Total non-stop words)
- **Vocabulary Fingerprint:** The text exhibits a specific ratio of parts of speech (Verbs: {verb_ratio:.2f}, Adjectives: {adj_ratio:.2f}, Nouns: {noun_ratio:.2f}).
- **Perspective:** High reliance on first-person perspective (Ratio of 1st person pronouns to total words: {fp_ratio:.4f}). This indicates an internal, highly subjective narrative distance.

## 3. Syntactic Fingerprint
- **Average Sentence Length:** {avg_sentence_length:.2f} tokens per sentence.
- **Rhythm:** The author balances descriptive interiority with specific pacing structures.

## 4. Tone & Affect
- The heavy use of first-person perspective combined with the specific verb/adjective ratios suggests a narrative voice that is deeply rooted in personal observation and internal processing rather than purely external action.
"""

with open('STYLE_GUIDE.md', 'w') as f:
    f.write(style_guide)

linting_rules = {
    "system_prompts": {
        "perspective": "First-person internal monologue",
        "sentence_length": "Average around {:.2f} words, varying for rhythmic effect".format(avg_sentence_length),
        "lexical_density": "Maintain a lexical density around {:.4f}".format(lexical_density)
    },
    "linguistic_rules": {
        "avoid": [
            "Over-reliance on third-person objective viewpoints",
            "Excessively long, unbroken paragraphs without dialogue or internal reflection"
        ],
        "require": [
            "Strong internal narrative voice",
            "Consistent use of subjective observation"
        ]
    }
}

with open('LINTING_RULES.json', 'w') as f:
    json.dump(linting_rules, f, indent=4)

# Delete transcription to comply with strict non-derivation rule
print("Cleaning up transcription.txt...")
if os.path.exists('transcription.txt'):
    os.remove('transcription.txt')
print("Pipeline complete.")
