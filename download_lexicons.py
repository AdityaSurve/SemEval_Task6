"""
Download professional NLP lexicons for feature extraction.

Lexicons included:
1. AFINN - Sentiment lexicon (via pip)
2. NRC EmoLex - Emotion lexicon (manual download required)
3. Brysbaert Concreteness Ratings (manual download required)
4. MPQA Subjectivity Lexicon (manual download required)

Run this script once to set up the lexicons.
"""

import os
import subprocess
import sys

LEXICON_DIR = "external_datasets"
os.makedirs(LEXICON_DIR, exist_ok=True)

print("=" * 60)
print("PROFESSIONAL NLP LEXICON SETUP")
print("=" * 60)

# 1. Install AFINN
print("\n[1/4] AFINN Sentiment Lexicon (Finn Årup Nielsen, 2011)")
print("-" * 60)
try:
    from afinn import Afinn
    print("✓ Already installed")
except ImportError:
    print("Installing via pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "afinn", "-q"])
    print("✓ Installed successfully")

# 2. NRC EmoLex
print("\n[2/4] NRC Emotion Lexicon (Mohammad & Turney, 2013)")
print("-" * 60)
nrc_path = f"{LEXICON_DIR}/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
if os.path.exists(nrc_path):
    print(f"✓ Found at: {nrc_path}")
else:
    print("⚠ Manual download required:")
    print("  1. Go to: https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm")
    print("  2. Fill out the form to request access")
    print("  3. Download and extract to: external_datasets/")
    print(f"  4. Expected file: {nrc_path}")

# 3. Brysbaert Concreteness
print("\n[3/4] Brysbaert Concreteness Ratings (Brysbaert et al., 2014)")
print("-" * 60)
conc_path = f"{LEXICON_DIR}/Concreteness_ratings_Brysbaert_et_al_BRM.txt"
if os.path.exists(conc_path):
    print(f"✓ Found at: {conc_path}")
else:
    print("⚠ Manual download required:")
    print("  1. Go to: https://link.springer.com/article/10.3758/s13428-013-0403-5")
    print("  2. Or direct: https://osf.io/j5ehb/ (Supplementary materials)")
    print(f"  3. Save as: {conc_path}")

# 4. MPQA Subjectivity Lexicon
print("\n[4/4] MPQA Subjectivity Lexicon (Wilson et al., 2005)")
print("-" * 60)
mpqa_path = f"{LEXICON_DIR}/subjclueslen1-HLTEMNLP05.tff"
if os.path.exists(mpqa_path):
    print(f"✓ Found at: {mpqa_path}")
else:
    print("⚠ Manual download required:")
    print("  1. Go to: https://mpqa.cs.pitt.edu/lexicons/subj_lexicon/")
    print("  2. Download 'subjectivity_clues_hltemnlp05'")
    print(f"  3. Extract and place: {mpqa_path}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

lexicons = [
    ("AFINN", True),  # Always available via pip
    ("NRC EmoLex", os.path.exists(nrc_path)),
    ("Brysbaert Concreteness", os.path.exists(conc_path)),
    ("MPQA Subjectivity", os.path.exists(mpqa_path)),
    ("Hedge Words (hedges.txt)", os.path.exists(f"{LEXICON_DIR}/hedges.txt")),
]

for name, available in lexicons:
    status = "✓" if available else "⚠ Missing"
    print(f"  {status} {name}")

print("\n" + "=" * 60)
