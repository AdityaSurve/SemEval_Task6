"""
Professional NLP Lexicon Loader

Loads academic/professional lexicons for linguistic feature extraction:
- AFINN: Sentiment scores
- NRC EmoLex: Emotion associations  
- Brysbaert: Concreteness ratings
- MPQA: Subjectivity classification
- Hedge words: Uncertainty markers
"""

import os
from typing import Dict, Set, List, Optional

LEXICON_DIR = "external_datasets"


class LexiconLoader:
    """Load and manage professional NLP lexicons."""
    
    def __init__(self, lexicon_dir: str = LEXICON_DIR):
        self.lexicon_dir = lexicon_dir
        self._afinn = None
        self._nrc_emotions = None
        self._nrc_sentiments = None
        self._concreteness = None
        self._mpqa_strong = None
        self._mpqa_weak = None
        self._hedges = None
        
    # ==========================================
    # AFINN Sentiment Lexicon
    # ==========================================
    @property
    def afinn(self):
        """AFINN sentiment analyzer (Finn Årup Nielsen, 2011)."""
        if self._afinn is None:
            try:
                from afinn import Afinn
                self._afinn = Afinn()
            except ImportError:
                print("Install AFINN: pip install afinn")
                self._afinn = None
        return self._afinn
    
    def get_afinn_score(self, text: str) -> float:
        """Get AFINN sentiment score for text."""
        if self.afinn:
            return self.afinn.score(text)
        return 0.0
    
    # ==========================================
    # NRC Emotion Lexicon (EmoLex)
    # ==========================================
    def _load_nrc(self):
        """Load NRC Emotion Lexicon."""
        if self._nrc_emotions is not None:
            return
            
        path = f"{self.lexicon_dir}/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
        emotions = ['anger', 'fear', 'anticipation', 'trust', 'surprise', 'sadness', 'joy', 'disgust']
        self._nrc_emotions = {e: set() for e in emotions}
        self._nrc_sentiments = {'positive': set(), 'negative': set()}
        
        if not os.path.exists(path):
            return
            
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3 and parts[2] == '1':
                    word, category = parts[0].lower(), parts[1]
                    if category in self._nrc_emotions:
                        self._nrc_emotions[category].add(word)
                    elif category in self._nrc_sentiments:
                        self._nrc_sentiments[category].add(word)
    
    @property
    def nrc_emotions(self) -> Dict[str, Set[str]]:
        """NRC emotion word sets."""
        self._load_nrc()
        return self._nrc_emotions
    
    @property
    def nrc_sentiments(self) -> Dict[str, Set[str]]:
        """NRC positive/negative word sets."""
        self._load_nrc()
        return self._nrc_sentiments
    
    def get_emotion_counts(self, tokens: List[str]) -> Dict[str, int]:
        """Count emotion words in token list."""
        counts = {e: 0 for e in self.nrc_emotions}
        for token in tokens:
            for emotion, words in self.nrc_emotions.items():
                if token.lower() in words:
                    counts[emotion] += 1
        return counts
    
    # ==========================================
    # Brysbaert Concreteness Ratings
    # ==========================================
    @property
    def concreteness(self) -> Dict[str, float]:
        """Brysbaert concreteness ratings (1=abstract, 5=concrete)."""
        if self._concreteness is None:
            self._concreteness = {}
            path = f"{self.lexicon_dir}/Concreteness_ratings_Brysbaert_et_al_BRM.txt"
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    next(f)  # Skip header
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            try:
                                self._concreteness[parts[0].lower()] = float(parts[2])
                            except:
                                pass
        return self._concreteness
    
    def get_avg_concreteness(self, tokens: List[str]) -> float:
        """Get average concreteness score for tokens."""
        scores = [self.concreteness.get(t.lower(), 0) for t in tokens]
        valid = [s for s in scores if s > 0]
        return sum(valid) / len(valid) if valid else 0.0
    
    # ==========================================
    # MPQA Subjectivity Lexicon
    # ==========================================
    def _load_mpqa(self):
        """Load MPQA subjectivity lexicon."""
        if self._mpqa_strong is not None:
            return
            
        self._mpqa_strong = set()
        self._mpqa_weak = set()
        path = f"{self.lexicon_dir}/subjclueslen1-HLTEMNLP05.tff"
        
        if not os.path.exists(path):
            return
            
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = dict(item.split('=') for item in line.strip().split() if '=' in item)
                word = parts.get('word1', '').lower()
                if parts.get('type') == 'strongsubj':
                    self._mpqa_strong.add(word)
                elif parts.get('type') == 'weaksubj':
                    self._mpqa_weak.add(word)
    
    @property
    def mpqa_strong_subjective(self) -> Set[str]:
        """Strongly subjective words."""
        self._load_mpqa()
        return self._mpqa_strong
    
    @property
    def mpqa_weak_subjective(self) -> Set[str]:
        """Weakly subjective words."""
        self._load_mpqa()
        return self._mpqa_weak
    
    # ==========================================
    # Hedge Words
    # ==========================================
    @property
    def hedge_words(self) -> Set[str]:
        """Hedge/uncertainty words (Hyland-style)."""
        if self._hedges is None:
            path = f"{self.lexicon_dir}/hedges.txt"
            self._hedges = set()
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self._hedges = {line.strip().lower() for line in f if line.strip()}
        return self._hedges
    
    # ==========================================
    # Standard Word Lists
    # ==========================================
    @property
    def modal_verbs(self) -> Set[str]:
        return {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would', 'ought'}
    
    @property
    def negation_words(self) -> Set[str]:
        return {'no', 'not', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'never',
                "n't", 'cannot', "can't", "won't", "don't", "doesn't", "didn't",
                "haven't", "hasn't", "hadn't", "isn't", "aren't", "wasn't", "weren't"}
    
    @property
    def filler_words(self) -> Set[str]:
        return {'uh', 'um', 'er', 'ah', 'like', 'well', 'so', 'basically', 'actually', 'literally'}
    
    @property
    def filler_phrases(self) -> List[str]:
        return ['you know', 'i mean', 'kind of', 'sort of']
    
    @property
    def vague_words(self) -> Set[str]:
        return {'some', 'many', 'few', 'several', 'various', 'certain', 'thing', 'things',
                'stuff', 'something', 'anything', 'people', 'someone', 'often', 'sometimes'}
    
    @property
    def vague_phrases(self) -> List[str]:
        return ['a lot', 'a bit', 'a little']
    
    @property
    def pivot_phrases(self) -> List[str]:
        return ['what i want to say', 'let me be clear', 'the fact is', 'the truth is',
                'the real question', 'what matters is', 'the important thing', 
                'the bottom line', 'at the end of the day']
    
    @property
    def thanks_starters(self) -> List[str]:
        return ['thank you', 'thanks for', 'great question', 'good question', 'i appreciate']
    
    def print_status(self):
        """Print lexicon loading status."""
        print("=" * 50)
        print("LEXICON STATUS")
        print("=" * 50)
        
        checks = [
            ("AFINN", self.afinn is not None),
            ("NRC EmoLex", bool(self.nrc_emotions.get('anger'))),
            ("Brysbaert Concreteness", bool(self.concreteness)),
            ("MPQA Subjectivity", bool(self.mpqa_strong_subjective)),
            ("Hedge Words", bool(self.hedge_words)),
        ]
        
        for name, loaded in checks:
            status = "✓" if loaded else "✗"
            print(f"  {status} {name}")
        
        print("=" * 50)


# Global instance
lexicons = LexiconLoader()


if __name__ == "__main__":
    lexicons.print_status()
