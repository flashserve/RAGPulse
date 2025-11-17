"""
Load synthetic traces and hash-id mappings, then simulate token sequences
for each hash-id.
"""

import json          # For parsing .jsonl lines
import os            # For path joining
from typing import Dict, List, Tuple


def simulate_token_lists(token_length: int, vocab_size: int) -> List[int]:
    """
    Generate a random token-id list of given length.

    Args:
        token_length: Desired number of tokens.
        vocab_size: Vocabulary size (exclusive upper bound).

    Returns:
        List of token ids.
    """
    import random                  # Deferred import for randomness
    # Create a list of 'token_length' integers, each in [0, vocab_size-1]
    return [random.randint(0, vocab_size - 1) for _ in range(token_length)]


def simulate_traces(trace_dir: str, model_dir: str) -> Tuple[List[dict], Dict[int, List[int]]]:
    """
    Load trace file and hash-id files, then create a mapping
    from hash-id to simulated token lists.

    Args:
        trace_dir: Directory containing trace and hash-id files.
        model_dir: Directory containing vocab.json to obtain vocab size.

    Returns:
        traces: List of trace dictionaries.
        hash_id_dict: Dict mapping hash-id -> List[token_ids].
    """
    #  ----------------  LOAD TRACE FILE  ----------------
    traces = []                      # Will hold every trace record
    trace_name = "0_trace.jsonl"     # Fixed file name for traces
    # Build full path and open in text mode with UTF-8
    with open(os.path.join(trace_dir, trace_name), "r", encoding="utf-8") as f:
        for line in f:               # .jsonl => one JSON object per line
            traces.append(json.loads(line))  # Parse JSON -> dict
    print(f"Loaded {len(traces)} traces from {trace_name}.")

    #  ----------------  PREPARE HASH-ID FILES  ----------------
    # These files store {hash_id: token_length} pairs for different contexts
    hash_id_dir_names = [
        "1_sys_prompt.jsonl",
        "2_passages.jsonl",
        "3_history.jsonl",
        "4_user_input.jsonl",
        "5_web_search.jsonl",
    ]
    hash_id_dict: Dict[int, List[int]] = {}   # hash_id -> simulated tokens

    #  ----------------  GET VOCAB SIZE  ----------------
    vocab_file = os.path.join(model_dir, "vocab.json")  # Standard HF file
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab_size = len(json.load(f))           # Number of entries = vocab size
    print(f"Loaded vocab size: {vocab_size} from {vocab_file}.")

    #  ----------------  LOAD HASH-ID -> TOKEN_LENGTH  ----------------
    for hash_id_dir_name in hash_id_dir_names:   # Iterate over each context file
        # Build full path for current context file
        with open(os.path.join(trace_dir, hash_id_dir_name), "r", encoding="utf-8") as f:
            for line in f:                       # Each line is a single JSON object
                hash_id = json.loads(line)       # E.g. {"42": 128}
                k, v = list(hash_id.values())    # k=hash_id (int), v=token_length (int)
                # Simulate a random token list of length 'v'
                hash_id_dict[k] = simulate_token_lists(v, vocab_size)
    print(f"Loaded {len(hash_id_dict)} hash_ids from {hash_id_dir_names}.")
    return traces, hash_id_dict