from collections import defaultdict
import os, re, regex
import multiprocessing as mp


VOCAB_SIZE = 256


def build_word_freq_chunk(
    file_path: str, start: int, end: int, special_tokens: list[str]
) -> dict[tuple[int, ...], int]:
    """
    Build a word frequency dictionary for a chunk of the file.
    """
    word_freq = defaultdict(int)

    # 3. read the chunk of the file
    with open(file_path, "r") as f:
        f.seek(start)
        text = f.read(end - start)

    # 4. split around special tokens
    if special_tokens:
        pattern = '|'.join(re.escape(token) for token in special_tokens)
        pieces = re.split(pattern, text)
    else:
        pieces = [text]
    
    # 5. GPT-2 pre-tokenization
        GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenized_pieces = []
    for piece in pieces:
        tokens = regex.findall(GPT2_PATTERN, piece)
        pre_tokenized_pieces.extend(tokens)
    
    # 6. convert pieces to sequences of byte token IDs
    for token in pre_tokenized_pieces:
        token_ids = tuple(token.encode("utf-8"))
        word_freq[token_ids] += 1

    return word_freq


def build_word_freq_parallel(
    file_path: str,
    special_tokens: list[str],
    num_processes: int = 4
) -> dict[tuple[int, ...], int]:
    """
    Build a word frequency dictionary for the entire file using parallel processing.
    """
    file_size = os.path.getsize(file_path)

    # edge case
    if num_processes < 1 or num_processes > mp.cpu_count():
        raise ValueError(f"num_processes must be between 1 and {mp.cpu_count()}")
    
    # multiprocessing tool
    chunk_size = file_size // num_processes
    
    args = []
    for chunk_idx in range(num_processes):
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size if chunk_idx < num_processes - 1 else file_size
        args.append((file_path, start_idx, end_idx, special_tokens))

    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            build_word_freq_chunk, 
            args
        )
        
    # process the multiprocessed results to build the final word frequency dictionary
    word_freq = defaultdict(int)
    for local_word_freq in results:
        for k, v in local_word_freq.items():
            word_freq[k] += v
    
    return word_freq


def apply_merge(token_ids: tuple[int, ...], prev: int, next: int, new_id: int) -> tuple[int, ...]:
    """
    Given a sequence of token IDs, replace all occurrences of the pair (prev, next) with new_id.
    """
    result = []
    idx = 0
    while idx < len(token_ids):
        if idx + 1 < len(token_ids) and token_ids[idx] == prev and token_ids[idx + 1] == next:
            result.append(new_id)
            idx += 2
        else:
            result.append(token_ids[idx])
            idx += 1
    return tuple(result)

def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input file and return the vocabulary and merges.
    """
    # 0. edge case
    if vocab_size < VOCAB_SIZE + len(special_tokens):
        raise ValueError(f"vocab_size must be greater than {VOCAB_SIZE + len(special_tokens)}")
    
    # 1. initialized vocabulary
    vocab = {i: bytes([i]) for i in range(VOCAB_SIZE)}
    
    # 2. add special tokens to the vocabulary
    next_id = VOCAB_SIZE
    for token in special_tokens:
        vocab[next_id] = token.encode('utf-8')
        next_id += 1
    
    # 3. read training corpus
    # with open(input_path, 'r') as f:
    #     text = f.read()
    
    # 4. split around special tokens
    # if special_tokens:
    #     pattern = '|'.join(re.escape(token) for token in special_tokens)
    #     pieces = re.split(pattern, text)
    # else:
    #     pieces = [text]
    # pieces = [piece for piece in pieces if piece]

    # 5. GPT-2 pre-tokenization
    # GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # pre_tokenized_pieces = []
    # for piece in pieces:
    #     tokens = regex.findall(GPT2_PATTERN, piece)
    #     pre_tokenized_pieces.extend(tokens)

    # 6. convert pieces to sequences of byte token IDs
    # tokenized_corpus = []
    # for token in pre_tokenized_pieces:
    #     token_ids = list(token.encode("utf-8"))
    #     tokenized_corpus.append(token_ids)
    # Optimized version using word frequency dict
    # word_freq = defaultdict(int)
    # for token in pre_tokenized_pieces:
    #     token_ids = tuple(token.encode("utf-8"))
    #     word_freq[token_ids] += 1

    # 3 to 6. Build word frequency dictionary in parallel using multiprocessing
    word_freq = build_word_freq_parallel(input_path, special_tokens, num_processes=mp.cpu_count())

    # 7. repeatedly: count pairs, choose best pair, create new vocabulary entry, replace pair
    merges = []
    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        # Count adjacent token-ID pairs
        pair_counts = defaultdict(int)

        # for ids in tokenized_corpus:
        #     for i in range(len(ids) - 1):
        #         pair = (ids[i], ids[i + 1])
        #         pair_counts[pair] = pair_counts.get(pair, 0) + 1
        # Optimized version using word frequency dict
        for k_ids, v_freq in word_freq.items():
            for idx in range(len(k_ids) - 1):
                pair = (k_ids[idx], k_ids[idx + 1])
                pair_counts[pair] += v_freq


        if not pair_counts:
            break
        
        # Most frequent pair
        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair], 
                vocab[pair[0]],
                vocab[pair[1]]
            )
        )

        # Create new token
        new_id = next_id

        vocab[new_id] = (
            vocab[best_pair[0]]
            + vocab[best_pair[1]]
        )

        # Record merge as bytes
        merges.append(
            (
                vocab[best_pair[0]],
                vocab[best_pair[1]],
            )
        )

        # Replace pair
        # for ids in tokenized_corpus:
        #     i = 0
        #     while i < len(ids) - 1:
        #         if (ids[i], ids[i + 1]) == best_pair:
        #             ids[i:i + 2] = [new_id]
        #             i += 1
        #         else:
        #             i += 1
        # refactor to use standalone function for merge
        # for i in range(len(tokenized_corpus)):
        #     tokenized_corpus[i] = apply_merge(tokenized_corpus[i], best_pair[0], best_pair[1], new_id)
        # Optimized version using word frequency dict
        new_word_freq = defaultdict(int)
        for k_ids, v_freq in word_freq.items():
            new_ids = apply_merge(k_ids, best_pair[0], best_pair[1], new_id)
            new_word_freq[new_ids] += v_freq
        word_freq = new_word_freq
        next_id += 1

    # 8. return vocab, merges
    return vocab, merges


if __name__ == "__main__":
    input_path = "./sample.txt"
    vocab_size = 264
    special_tokens = ["<|endoftext|>", "<|pad|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    print("Vocabulary:")
    for token_id, token_bytes in vocab.items():
        print(f"{token_id}: {token_bytes}")
    print("\nMerges:")
    for merge in merges:
        print(merge)