from .pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool, cpu_count
import regex as re
from collections import Counter, defaultdict
from loguru import logger


class Chunker:
    def __init__(self, input_path: str, special_token: str, num_processes: int):
        self.input_path = input_path
        self.special_token = special_token.encode("utf-8")
        self.num_processes = num_processes

    def find_boundaries(self) -> list[int]:
        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, self.num_processes, self.special_token
            )
        return boundaries


def pre_tokenize(
    input_path: str, special_tokens: list[str], start: int, end: int, pat: str
) -> Counter:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    counts = Counter()
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    sub_chunks = re.split(pattern, chunk)

    for sub_chunk in sub_chunks:
        for match in re.finditer(pat, sub_chunk):
            token = match.group(0)
            counts[tuple([bytes([b]) for b in token.encode("utf-8")])] += 1

    return counts


class Word:
    def __init__(self, tokens: tuple[bytes], count: int):
        self.tokens = list(tokens)
        self.count = count
        self.pairs = Counter()
        self._update_pairs()

    def _update_pairs(self):
        self.pairs.clear()
        for x, y in zip(self.tokens[:-1], self.tokens[1:]):
            self.pairs[(x, y)] += 1

    def get_pairs(self) -> Counter:
        return self.pairs

    def get_count(self) -> int:
        return self.count

    def merge(self, pair: tuple[bytes, bytes]):
        idx = 0
        new_tokens = []
        while idx < len(self.tokens):
            if (
                idx < len(self.tokens) - 1
                and self.tokens[idx] == pair[0]
                and self.tokens[idx + 1] == pair[1]
            ):
                new_tokens.append(pair[0] + pair[1])
                idx += 2
            else:
                new_tokens.append(self.tokens[idx])
                idx += 1

        self.tokens = new_tokens
        self._update_pairs()


class BPETrainer:
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab_size = vocab_size
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.next_id = 256
        for idx, token in enumerate(special_tokens):
            self.vocab[self.next_id] = token.encode("utf-8")
            self.next_id += 1
        self.merges: list[tuple[bytes, bytes]] = []

    def train(
        self, pre_tokens: Counter
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        words = [Word(token, cnt) for token, cnt in pre_tokens.items()]
        pair_counter = Counter()
        pair_to_word_idx: dict[tuple[bytes, bytes], set] = defaultdict(set)

        for idx, word in enumerate(words):
            wd_pairs = word.get_pairs()  # dict[tuple[bytes, bytes], int]
            for pair, cnt in wd_pairs.items():
                pair_to_word_idx[pair].add(idx)
                pair_counter[pair] += (word.get_count() * cnt)

        while len(self.vocab) < self.vocab_size:
            if not pair_counter:
                break

            max_freq = max(pair_counter.values())
            most_common_pairs = [k for k, v in pair_counter.items() if v == max_freq]
            most_common_pairs.sort()
            most_common_pairs_tuple = most_common_pairs[-1]

            self.merges.append(most_common_pairs_tuple)

            most_common_pairs_bytes = most_common_pairs_tuple[0] + most_common_pairs_tuple[1]
            self.vocab[self.next_id] = most_common_pairs_bytes
            self.next_id += 1

            affected_idx = pair_to_word_idx[most_common_pairs_tuple].copy()  # set
            for idx in affected_idx:
                word = words[idx]
                wd_pairs = word.get_pairs()

                for pair, cnt in wd_pairs.items():
                    pair_counter[pair] -= word.get_count() * cnt
                    pair_to_word_idx[pair].discard(idx)
                    if pair_counter[pair] <= 0:
                        del pair_counter[pair]

                word.merge(most_common_pairs_tuple)

                wd_pairs = word.get_pairs()
                for pair, cnt in wd_pairs.items():
                    pair_counter[pair] += word.get_count() * cnt
                    pair_to_word_idx[pair].add(idx)


        return self.vocab, self.merges


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe_main(input_path: str, vocab_size: int, special_tokens: list[str]):
    num_proc = cpu_count() - 2
    chunker = Chunker(input_path, "<|endoftext|>", num_proc)
    boundaries = chunker.find_boundaries()

    chunk_args = [
        (input_path, special_tokens, start, end, PAT)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(num_proc) as pool:
        results = pool.starmap(pre_tokenize, chunk_args)

    pre_tokens = Counter()
    for res in results:
        pre_tokens.update(res)

    trainer = BPETrainer(vocab_size, special_tokens)
    vocab, merges = trainer.train(pre_tokens)

    return vocab, merges