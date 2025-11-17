"""
Factory function to build a vLLM model instance and its tokenizer.
"""

from vllm import LLM
from transformers import AutoTokenizer


def build_llm(model_dir: str):
    """
    Build a vLLM LLM object with prefix-caching enabled.

    Args:
        model_dir: Local path to the HuggingFace model.(set by main.py)

    Returns:
        Tuple of (llm_instance, tokenizer_instance).
    """
    llm = LLM(
        model=model_dir,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        max_model_len=20000,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return llm, tokenizer