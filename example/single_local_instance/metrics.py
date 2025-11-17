"""
Simple metric utilities for TTFT (Time-To-First-Token) and
TPOT (Time-Per-Output-Token).
"""

from typing import List


def aggregate_metrics(ttfts: List[float], tpots: List[float]):
    """
    Compute and print average TTFT and TPOT.

    Args:
        ttfts: List of Time-To-First-Token values (seconds).
        tpots: List of Time-Per-Output-Token values (seconds).

    Returns:
        Tuple of (avg_ttft, avg_tpot).
    """
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    avg_tpot = sum(tpots) / len(tpots) if tpots else 0.0
    print(f"Average TTFT: {avg_ttft:.4f}s, Average TPOT: {avg_tpot:.4f}s")
    return avg_ttft, avg_tpot