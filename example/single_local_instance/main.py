"""
Entry point for the real-time LLM-inference benchmark.
"""

import argparse
import os
import torch

# Import helper modules
from preprocess_data import simulate_traces
from model_pool import build_llm
from real_time_runner import run_real_time
from metrics import aggregate_metrics


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace_dir",
        type=str,
        default=r"../../data",
        help="Path to the trace directory.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=r"/home/share/models/Qwen2.5-14B-Instruct",
        help="Path to the local model directory.",
    )
    return parser.parse_args()


def main():
    """Main workflow:
    1. Load traces and hash-id mappings.
    2. Build text prompts from token ids (only first 5 traces).
    3. Instantiate the vLLM model.
    4. Run real-time simulation and print average TTFT/TPOT.
    """
    print("CUDA available:", torch.cuda.is_available())
    args = get_args()

    # Load traces and hash-id dictionary
    traces, hash_id_dict = simulate_traces(args.trace_dir, args.model_dir)

    # Sort by timestamp and keep the first 5 traces only
    sorted_trace = sorted(traces, key=lambda x: int(x["timestamp"]))[:5]

    # Reconstruct input text from token ids
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    for tr in sorted_trace:
        token_ids = []
        for hid_list in tr["hash_ids"].values():
            for hid in hid_list:
                token_ids.extend(hash_id_dict[hid])
        tr["input_text"] = tokenizer.decode(token_ids)

    # Build LLM instance
    llm, _ = build_llm(args.model_dir)

    # Run real-time simulation
    ttfts, tpots = run_real_time(sorted_trace, llm)

    # Aggregate and print metrics
    aggregate_metrics(ttfts, tpots)


if __name__ == "__main__":
    main()