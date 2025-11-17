"""preprocess_data

Refactor the previous top-level functions into a `PreprocessData` class,
providing a clearer interface and more robust hash_id parsing.

The script preserves original behavior:
- Read trace records from `0_trace.jsonl`
- Read several hash_id files (one JSON object per line, e.g. {"42": 128}),
  where the value indicates token length
- Obtain vocab size from model directory's `vocab.json`
- Generate a random token id list for each hash_id (length = token_length)
"""

import json
import os
import random
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer
from logger import RAGPulseLogger
class PreprocessData:
    """Preprocessor that reads traces and hash_id files and simulates token id lists for each hash_id.

    Usage example:
        p = PreprocessData(trace_dir, tokenizer_dir)
        traces, hash_id_dict = p.preprocess()
    """

    DEFAULT_TRACE_NAME = "0_trace.jsonl"
    DEFAULT_HASH_ID_FILES = [
        "1_sys_prompt.jsonl",
        "2_passages.jsonl",
        "3_history.jsonl",
        "4_user_input.jsonl",
        "5_web_search.jsonl",
    ]

    def __init__(self, trace_dir: str, tokenizer_dir: str, use_trace_num: int = 5,logger: RAGPulseLogger = None, trace_name: str = None, hash_id_files: Optional[List[str]] = None):
        """
        Initialize PreprocessData instance.
        Args:
            trace_dir: Directory containing trace and hash_id files.
            tokenizer_dir: Directory of the tokenizer (for vocab size and decoding).
            use_trace_num: Number of traces to use from the front.
            logger: Optional RAGPulseLogger for logging.
            trace_name: Name of the trace file (default "0_trace.jsonl").
            hash_id_files: List of hash_id file names (default predefined list).
        """
        self.trace_dir = trace_dir
        self.tokenizer_dir = tokenizer_dir
        self.use_trace_num = use_trace_num
        self.logger = logger
        self.trace_name = trace_name or self.DEFAULT_TRACE_NAME
        self.hash_id_files = hash_id_files or list(self.DEFAULT_HASH_ID_FILES)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)

    @staticmethod
    def simulate_token_lists(token_length: int, vocab_size: int) -> List[int]:
        """Generate a random token id list of length token_length.

        Token ids are in the range [0, vocab_size-1].
        """
        if token_length <= 0 or vocab_size <= 0:
            return []
        return [random.randint(0, vocab_size - 1) for _ in range(token_length)]
    
    @staticmethod
    def simulate_token_lists_to_input_text(hash_id_lists: List[List[int]], hash_id_dict:Dict[int, List[int]],tokenizer) -> str:
        """Convert lists of hash_id lists to decoded input text using the tokenizer."""
        input_token_list = []
        for hash_id_list in hash_id_lists:
            for hash_id in hash_id_list:
                input_token_list.extend(hash_id_dict[hash_id])
        input_text = tokenizer.decode(input_token_list)
        return input_text

    def _load_traces(self) -> List[dict]:
        """Read trace records from the trace file (one JSON object per line)."""
        traces: List[dict] = []
        trace_path = os.path.join(self.trace_dir, self.trace_name)
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                traces.append(json.loads(line))
        if self.logger:
            self.logger.info(f"Loaded {len(traces)} traces from {self.trace_name}.")
        else:
            print(f"Loaded {len(traces)} traces from {self.trace_name}.")
        return traces

    def _load_vocab_size(self) -> int:
        """Read vocab size from tokenizer directory's vocab.json file."""
        vocab_file = os.path.join(self.tokenizer_dir, "vocab.json")
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
        if self.logger:
            self.logger.info(f"Loaded vocab size: {vocab_size} from {vocab_file}.")
        else:
            print(f"Loaded vocab size: {vocab_size} from {vocab_file}.")
        return vocab_size

    def _load_hash_id_dict(self, vocab_size: int) -> Dict[int, List[int]]:
        """Read hash_id files (one JSON object per line) and simulate token lists for each hash_id.

        This method is more robust: keys can be strings or numbers, and an object with multiple keys
        will generate entries for each key. Malformed lines are skipped. Missing files are warned and skipped.
        """
        hash_id_dict: Dict[int, List[int]] = {}
        for fname in self.hash_id_files:
            path = os.path.join(self.trace_dir, fname)
            if not os.path.isfile(path):
                # Skip missing files but print a warning
                self.logger.warning(f"Hash_id file not found, skipping: {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    hash_id = json.loads(line)
                    hash_id_values = list(hash_id.values())
                    assert len(hash_id_values)==2, "Each line must contain exactly two key-value pairs."
                    hash_id_dict[hash_id_values[0]] = self.simulate_token_lists(hash_id_values[1],vocab_size)
        if self.logger:
            self.logger.info(f"Loaded {len(hash_id_dict)} hash_ids from {self.hash_id_files}.")
        else:
            print(f"Loaded {len(hash_id_dict)} hash_ids from {self.hash_id_files}.")
        return hash_id_dict

    def preprocess(self) -> List[dict]:
        """Run the full preprocessing pipeline: load traces, determine vocab size,
        load hash_id token lengths and simulate token lists. Returns (traces, hash_id_dict)."""
        traces = self._load_traces()
        vocab_size = self._load_vocab_size()
        hash_id_dict = self._load_hash_id_dict(vocab_size)
        for trace in traces:
            hash_id_lists = list(trace["hash_ids"].values())
            trace["input_text"] = self.simulate_token_lists_to_input_text(hash_id_lists, hash_id_dict,self.tokenizer)
        return traces[:self.use_trace_num]

# Unit-Test
def preprocess_traces(trace_dir: str, tokenizer_dir: str) -> Tuple[List[dict], Dict[int, List[int]]]:
    """Backward-compatible convenience wrapper: construct PreprocessData and run preprocess()."""
    p = PreprocessData(trace_dir, tokenizer_dir)
    return p.preprocess()