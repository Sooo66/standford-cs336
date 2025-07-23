from typing import Iterable, Iterator
import json
import base64
from cs336_basics.bpe_tokenizer_trainer import b64_decode
import regex as re
from loguru import logger
from timeit import timeit
import numpy as np
from tqdm import tqdm

logger = logger.bind(name='tokenizer')

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self._rvocab = {v: k for k, v in self.vocab.items()}
        self._merge_rank = {pair: rk for rk, pair in enumerate(self.merges)}
        self._id = len(self.vocab)
        if special_tokens is not None:
            self.special_tokens.sort(key=len, reverse=True)
            for token in self.special_tokens:
                btoken = token.encode("utf-8")
                if btoken not in self._rvocab:
                    logger.debug(f"Adding special token: {token}")
                    self._add_token(token)


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens: list[str] | None=None):
        instance = cls({}, [], special_tokens)
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            json_vocab = json.load(f)
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            json_merges = json.load(f)
        
        instance.vocab = {int(k): b64_decode(v) for k, v in json_vocab.items()}
        instance.merges = [(b64_decode(x), b64_decode(y)) for x, y in json_merges]
        instance._rvocab = {v: k for k, v in instance.vocab.items()}
        instance._merge_rank = {pair: rk for rk, pair in enumerate(instance.merges)}
        instance._id = len(instance.vocab)
        if instance.special_tokens is not None:
            instance.special_tokens.sort(key=len, reverse=True)
            for token in instance.special_tokens:
                btoken = token.encode("utf-8")
                if btoken not in instance._rvocab:   # 同样判断
                    logger.debug(f"Adding special token: {token}")
                    instance._add_token(token)
        
        return instance


    def encode(self, text: str) -> list[int]:
        words: list[list[bytes]] = self._pre_tokenize(text)

        token_ids: list[int] = []
        for wd in words:
            ids = self._encode_word(wd)
            token_ids.extend(ids)
        
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        byte_chunks = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_chunks.append(self.vocab[token_id])
            else:
                byte_chunks.append(b'\xff')
        
        concatenated_bytes = b''.join(byte_chunks)
        decoded_text = concatenated_bytes.decode('utf-8', errors='replace')
        return decoded_text

    def _add_token(self, token: str):
        self.vocab[self._id] = token.encode('utf-8')
        self._rvocab[token.encode('utf-8')] = self._id
        self._id += 1

    def _pre_tokenize(self, text: str) -> list[list[bytes]]:
        if not self.special_tokens:
            words: list[list[bytes]] = []
            for match in re.finditer(PAT, text):
                token = match.group(0)
                words.append([bytes([c]) for c in token.encode('utf-8')])
            return words

        special_pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
        split_pattern = f"({special_pattern})"
        chunks = re.split(split_pattern, text)
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
                if pair in self._merge_rank and self._merge_rank[pair] < min_rank:
                    min_rank = self._merge_rank[pair]
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

        return [self._rvocab[token] for token in tokens]
    

import random

def tokenizer_experiments():
    # Load the tokenizer
    ts_tokenizer = tokenizer.from_files(
        '/home/yangmingxuan/standford-cs336/data/ts_train_vocab.json',
        '/home/yangmingxuan/standford-cs336/data/ts_train_merges.json',
        special_tokens=['<|endoftext|>']
    )

    owt_tokenizer = tokenizer.from_files(
        '/home/yangmingxuan/standford-cs336/data/owt_train_vocab.json',
        '/home/yangmingxuan/standford-cs336/data/owt_train_merges.json',
        special_tokens=['<|endoftext|>']
    )

    ts_tokenizer.encode('<|endoftext|>')  # warm up the tokenizer
    owt_tokenizer.encode('<|endoftext|>')  # warm up the tokenizer

    # sample 10 docs from file, <eot> is the split token
    with open('/home/yangmingxuan/standford-cs336/data/TinyStoriesV2-GPT4-valid.txt', 'r', encoding='utf-8') as f:
        ts_docs = random.sample(f.read().split('<|endoftext|>'), 10)
    with open('/home/yangmingxuan/standford-cs336/data/owt_valid.txt', 'r', encoding='utf-8') as f:
        owt_docs = random.sample(f.read().split('<|endoftext|>'), 10)

    # Encode the documents
    ts_tokens = [ts_tokenizer.encode(doc) for doc in ts_docs]
    owt_tokens = [owt_tokenizer.encode(doc) for doc in owt_docs]

    # (a)
    # Calculate compression ratios
    ts_compression = get_compression_ratio(ts_docs, ts_tokens)
    owt_compression = get_compression_ratio(owt_docs, owt_tokens)

    logger.info(f"TS Compression Ratio: {ts_compression:.2f} bytes/token")
    logger.info(f"OWT Compression Ratio: {owt_compression:.2f} bytes/token")

    # (b)
    owt_tokens_with_ts_tokenizer = [ts_tokenizer.encode(s) for s in owt_docs]
    _ = get_compression_ratio(owt_docs, owt_tokens_with_ts_tokenizer)
    logger.info(f"OWT Tokens with TS Tokenizer: {get_compression_ratio(owt_docs, owt_tokens_with_ts_tokenizer):.2f} bytes/token")

    # (c)
    time_taken = timeit(lambda: list(map(owt_tokenizer.encode, owt_docs)), number=100)
    total_bytes = sum(len(doc.encode('utf-8')) for doc in owt_docs)
    logger.info(f"OWT Tokenization Time: {time_taken:.2f} seconds")
    logger.info(f"OWT Tokenization Throughput: {total_bytes * 100 / time_taken:.2f} bytes/second")

    # (d)
    # save_tokenized_dataset(
    #     '/home/yangmingxuan/standford-cs336/data/owt_train.txt',
    #     '/home/yangmingxuan/standford-cs336/data/owt_train_tokenized.npy'
    # )
    save_tokenized_dataset(
        '/home/yangmingxuan/standford-cs336/data/TinyStoriesV2-GPT4-train.txt',
        '/home/yangmingxuan/standford-cs336/data/ts_train_tokenized.npy'
    )
    save_tokenized_dataset(
        '/home/yangmingxuan/standford-cs336/data/owt_valid.txt',
        '/home/yangmingxuan/standford-cs336/data/owt_valid_tokenized.npy'
    )
    save_tokenized_dataset(
        '/home/yangmingxuan/standford-cs336/data/TinyStoriesV2-GPT4-valid.txt',
        '/home/yangmingxuan/standford-cs336/data/ts_valid_tokenized.npy'
    )
    
def get_compression_ratio(text: list[str], token: list[int]) -> float:
    # bytes / tokens
    text_bytes = sum(len(t.encode('utf-8')) for t in text)
    token_lens = sum(len(t) for t in token)
    return text_bytes / token_lens if token_lens > 0 else float('inf')

# from pretokenization_example import find_chunk_boundaries
from cs336_basics.pretokenization_example import find_chunk_boundaries

from tqdm import tqdm

def tokenize_chunk(input_path: str, start: int, end: int, rank: int) -> list[int]:
    tokens = []
    mini_chunk_size = 64 * 1024 * 16
    special_token = '<|endoftext|>'.encode('utf-8')
    buffer = b''

    total_bytes = end - start
    current_bytes = 0

    with open(input_path, 'rb') as f:
        f.seek(start)

        # 如果 rank==1 就创建进度条，否则 dummy
        pbar = None
        if rank == 1:
            pbar = tqdm(total=total_bytes, desc=f"Rank {rank} processing", unit="B", unit_scale=True)

        while start < end:
            read_size = min(mini_chunk_size, end - start)
            chunk = f.read(read_size)
            if not chunk:
                break
            buffer += chunk
            start += len(chunk)

            current_bytes += len(chunk)
            if pbar:
                pbar.update(len(chunk))

            while True:
                idx = buffer.find(special_token)
                if idx == -1:
                    break
                part = buffer[:idx + len(special_token)]
                text = part.decode('utf-8', errors='ignore')
                tokens.extend(tokenizer_instance.encode(text))

                buffer = buffer[idx + len(special_token):]

        if pbar:
            pbar.close()

    if buffer:
        text = buffer.decode('utf-8', errors='ignore')
        tokens.extend(tokenizer_instance.encode(text))

    return tokens

from multiprocessing import Pool, cpu_count
tokenizer_instance = None
def init_tokenizer(tk_name: str):
    global tokenizer_instance
    if tk_name == 'owt':
        tokenizer_instance = tokenizer.from_files(
            '/home/yangmingxuan/standford-cs336/data/owt_train_vocab.json',
            '/home/yangmingxuan/standford-cs336/data/owt_train_merges.json',
            special_tokens=['<|endoftext|>']
        )
    else:
        tokenizer_instance = tokenizer.from_files(
            '/home/yangmingxuan/standford-cs336/data/ts_train_vocab.json',
            '/home/yangmingxuan/standford-cs336/data/ts_train_merges.json',
            special_tokens=['<|endoftext|>']
        )

def save_tokenized_dataset(input_path, output_path):
    special_tokens = ['<|endoftext|>']
    token_ids_list = []
    num_proc = cpu_count() - 2
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks=num_proc, split_special_token='<|endoftext|>'.encode('utf-8'))
    
    chunk_args = [(input_path, start, end, rank) for rank, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))]

    if 'owt' in input_path:
        tk_name = 'owt'
    else:
        tk_name = 'ts'
    
    with Pool(num_proc, initializer=init_tokenizer(tk_name)) as pool:
        results = pool.starmap(tokenize_chunk, chunk_args)
    
    for tokens in results:
        token_ids_list.extend(tokens)
    
    np.save(output_path, np.array(token_ids_list, dtype=np.uint16))


if __name__ == '__main__':
    tokenizer_experiments()
