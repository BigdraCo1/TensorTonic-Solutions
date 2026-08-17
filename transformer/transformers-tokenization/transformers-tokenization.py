import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id["<PAD>"] = 0
        self.word_to_id["<UNK>"] = 1
        self.word_to_id["<BOS>"] = 2
        self.word_to_id["<EOS>"] = 3

        self.id_to_word[0] = "<PAD>"
        self.id_to_word[1] = "<UNK>"
        self.id_to_word[2] = "<BOS>"
        self.id_to_word[3] = "<EOS>"

        unique_words = set()
        size = 0
        for text in texts:
            words = text.lower().split()
            unique_words.update(words)
        sorted_w = sorted(unique_words)
        size += len(sorted_w)
        self.vocab_size = size + 4
        
        for i in range(size):
            word = sorted_w[i];
            self.word_to_id[word] = i + 4;
            self.id_to_word[i + 4] = word;
            
            
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        encoded = []
        tokens = text.lower().split()

        for token in tokens:
            token_id = self.word_to_id.get(token, self.word_to_id[self.unk_token])
            encoded.append(token_id)
        return encoded
        
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        decoded = []
        for id in ids:
            word = self.id_to_word.get(id, self.unk_token)
            decoded.append(word)
        return " ".join(decoded)
            
