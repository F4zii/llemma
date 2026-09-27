import regex
from collections import Counter
from cs336_basics.word import Word

PAT = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def pre_tokenize(chunk: str, special_tokens: list[str]) -> list[Word]:
    """Splits chunk into protected special tokens and byte-ID sequences."""
    if not chunk:
        return []

    if not special_tokens:
        words = PAT.findall(chunk)
        return [
            Word(tuple(bytes([b]) for b in word.encode("utf-8")))
            for word in words
        ]

    special_pat = regex.compile(
        "|".join(regex.escape(tok) for tok in special_tokens)
    )

    pre_tokenized_chunks = []
    last_idx = 0

    for match in special_pat.finditer(chunk):
        start, end = match.span()

        # Normal text BEFORE the special token
        if start > last_idx:
            segment = chunk[last_idx:start]
            for word in PAT.findall(segment):
                byte_ids = tuple(bytes([b]) for b in word.encode("utf-8"))
                pre_tokenized_chunks.append(Word(byte_ids))

        # Special Token itself
        special_tok_bytes = match.group(0).encode("utf-8")
        pre_tokenized_chunks.append(
            Word((special_tok_bytes,), is_special=True)
        )
        last_idx = end

    # Remaining normal text AFTER the last special token
    if last_idx < len(chunk):
        segment = chunk[last_idx:]
        for word in PAT.findall(segment):
            byte_ids = tuple(bytes([b]) for b in word.encode("utf-8"))
            pre_tokenized_chunks.append(Word(byte_ids))

    return pre_tokenized_chunks

def pre_tokenize_and_count(chunk: str, special_tokens: list[str]) -> Counter:
    """Pre-tokenizes a text chunk and returns local Word counts."""
    words = pre_tokenize(chunk, special_tokens)
    return Counter(words)
