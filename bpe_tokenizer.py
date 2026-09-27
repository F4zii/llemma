import os, sys
from typing import BinaryIO
from multiprocessing import Pool
import regex as re
import itertools
from collections import Counter, defaultdict

from cs336_basics.helpers import pre_tokenize_and_count
from cs336_basics.word import Word

END_OF_TEXT_TOKEN = '<|endoftext|>'
INITIAL_VOCAB_LEN = 256

def get_pair_to_merge(word_frequency_table: dict[Word, int]):
    pair_freq_table = defaultdict(int)
    pair_to_word_table = defaultdict(list)

    # Local variable bindings for bytecode optimization (LOAD_FAST)
    p_freq = pair_freq_table
    p_words = pair_to_word_table

    for word, frequency in word_frequency_table.items():
        # Do not merge across special tokens
        if word.is_special or len(word) < 2:
            continue

        # Extract adjacent pairs in the tuple
        for pair in zip(word[:-1], word[1:]):
            # Weight by dataset frequency
            p_freq[pair] += frequency

            # Single dictionary lookup optimization
            p_words[pair].append(word)

    if not pair_freq_table:
        return None

    pair_to_merge = max(
        pair_freq_table,
        key=lambda pair: (pair_freq_table[pair], pair)
    )
    if pair_to_merge in [(b' ', b'.'), (b'a', b't')]:
        print(f'{pair_to_merge=} freq={pair_freq_table[pair_to_merge]}')

    return pair_to_merge, pair_to_word_table

def apply_merge(
    word_freq_table: dict[Word, int], pair: tuple[bytes, bytes]) -> dict[Word, int]:
    new_table = {}
    for word, freq in word_freq_table.items():
        merged_word = word.merge(pair)
        new_table[merged_word] = new_table.get(merged_word, 0) + freq
    return new_table

def bpe_shit(data: list[Word], special_tokens: list[bytes], vocab_size: int):
    # Initialize base vocabulary (0-255 bytes + special tokens)
    merges = []
    vocab = {i: bytes([i]) for i in range(INITIAL_VOCAB_LEN)}
    vocab.update({
        INITIAL_VOCAB_LEN + i: special_tokens[i]
        for i in range(len(special_tokens))
    })

    # O(N) frequency table construction
    word_frequency_table = Counter(data)
    while (len(vocab) < vocab_size):
        pair_to_merge, pair_to_word_table = get_pair_to_merge(word_frequency_table)
        # Update new pair in vocab and merge it across data.
        vocab[len(vocab)] = b''.join(pair_to_merge)

        matching_words = set(pair_to_word_table[pair_to_merge])
        # Go over words that need merging
        for word in matching_words:
            merged_word = word.merge(pair_to_merge)
            word_frequency_table[merged_word] = word_frequency_table[word]
            del word_frequency_table[word]
        merges.append(pair_to_merge)

    # print(vocab, merges)
    return vocab, merges

def train_bpe(filepath: str, vocab_size: int, special_tokens: list[str]):
    with open(filepath, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(
            f, num_processes, END_OF_TEXT_TOKEN.encode()
        )
        chunks = []

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

        # Create a pool and return Counter dictionaries instead of raw lists
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(
                pre_tokenize_and_count,
                [(chunk, special_tokens) for chunk in chunks],
            )

        # Merge partial Counters in the main process
        word_freq_table = Counter()
        for partial_counter in results:
            word_freq_table.update(partial_counter)

        special_tokens_encoded = [tok.encode() for tok in special_tokens]
        return bpe_shit(word_freq_table, special_tokens_encoded, vocab_size)

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file_size = os.fstat(file.fileno()).st_size

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

if __name__ == "__main__":
    # print(train_bpe('test.txt', 500, [END_OF_TEXT_TOKEN]))
    print(train_bpe('tests/fixtures/corpus.en', 500, [END_OF_TEXT_TOKEN]))
