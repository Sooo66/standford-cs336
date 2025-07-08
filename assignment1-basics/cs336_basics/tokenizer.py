from typing import Iterable, Iterator
import json
import base64
from bpe_tokenizer_trainer import b64_encode, b64_decode, Word
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self._rvocab = {v: k for k, v in self.vocab.items()}
        self._merge_rank = {pair: rk for rk, pair in enumerate(self.merges)}
        self._id = len(self.vocab)
        for token in special_tokens:
            self._add_token(token)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens: list[str] | None=None):
        json_vocab = json.load(vocab_filepath)
        json_merges = json.load(merges_filepath)
        
        cls.vocab = {int(k): b64_decode(v) for k, v in json_vocab.items()}
        cls.merges = [(b64_decode(x), b64_decode(y)) for x, y in json_merges]

        for token in special_tokens:
            cls._add_token(token) 


    def encode(self, text: str) -> list[int]:
        words: list[list[bytes]] = self._pre_tokenize(text)

        token_ids: list[int] = []
        for wd in words:
            ids = self._encode_word(wd)
            token_ids.extend(ids)
        
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass

    def _add_token(self, token: str):
        self.vocab[self._id] = token
        pass

    def _pre_tokenize(self, text: str) -> list[list[bytes]]:
        pattern = "|".join(f"({re.escape(tok)})" for tok in self.special_tokens)
        chunks = re.split(pattern, text)
        chunks = [chunk for chunk in chunks if chunk]

        words: list[list[bytes]] = []

        for chunk in chunks:
            if chunk in self.special_tokens:
                words.append([chunk.encode('utf-8')])
            else:
                for match in re.finditer(PAT, chunk):
                    token = match.group(0)
                    words.append([bytes([c]) for c in token.encode('utf-8')])
        
        return words
    
    def _encode_word(self, word: list[bytes]):
        tokens = word.copy()
        while True:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

            min_rank = float('inf')
            merge_pair = None
            for pair in pairs:
                if pair in self._merge_ranks and self._merge_ranks[pair] < min_rank:
                    min_rank = self._merge_ranks[pair]
                    merge_pair = pair

            if merge_pair is None:
                break

            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == merge_pair:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [self.vocab[token] for token in tokens]