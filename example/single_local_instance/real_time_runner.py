"""
Real-time simulation of request arrival and LLM inference.
Two concurrent threads:
  1. Request injector – sends requests according to trace timestamps.
  2. Response handler – runs vLLM.generate and records TTFT/TPOT.
"""

import time
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pipe
from typing import List, Tuple, Any, Dict
from vllm import LLM, SamplingParams

# Sentinel value to signal end of request stream
IS_END = -1


def _simulate_real_time_request(
    traces: List[Dict],
    start_time: float,
    real_time_requests: List,
    conn: Any,
):
    """
    Inject requests into the shared queue when the elapsed time
    matches the trace timestamp (scaled by 10× for faster demo).

    Args:
        traces: List of trace dictionaries containing 'timestamp' and 'input_text'.
        start_time: Reference time (seconds since epoch).
        real_time_requests: Shared list acting as a FIFO queue.
        conn: Multiprocessing connection to notify the consumer when finished.
    """
    idx = 0
    n_traces = len(traces)
    while idx < n_traces:
        current_time = time.time()
        if current_time - start_time >= int(traces[idx]["timestamp"]) // 10:
            trace = traces[idx]
            real_time_requests.append((current_time, trace["input_text"]))
            print(f"Sent request {idx + 1}/{n_traces} at time {current_time - start_time:.2f}s")
            idx += 1
        else:
            time.sleep(0.0001)
    conn.send(IS_END)


def _simulate_real_time_response(
    real_time_requests: List,
    llm: LLM,
    conn: Any,
):
    """
    Dequeue requests, run vLLM.generate, and collect TTFT/TPOT metrics.

    Args:
        real_time_requests: Shared FIFO queue of (receive_time, input_text).
        llm: vLLM instance.
        conn: Connection to check for end-of-stream signal.

    Returns:
        TTFTs: List of Time-To-First-Token values.
        TPOTs: List of Time-Per-Output-Token values.
    """
    TTFTs, TPOTs = [], []
    is_sending_end = False
    sampling_params = SamplingParams(temperature=0.0, max_tokens=300)

    while True:
        if real_time_requests:
            receive_time, input_text = real_time_requests.pop(0)
            outputs = llm.generate([input_text], sampling_params=sampling_params)
            ed_time = time.time()

            # Calculate TTFT
            ttft = outputs[0].metrics.first_token_time - receive_time
            gen_tokens = outputs[0].outputs[0].token_ids

            # Calculate TPOT
            tpot = (ed_time - outputs[0].metrics.arrival_time) / len(gen_tokens) if gen_tokens else 0.0

            TTFTs.append(ttft)
            TPOTs.append(tpot)
        else:
            # Check if producer has finished
            if conn.poll():
                if conn.recv() == IS_END:
                    is_sending_end = True
            # Exit when no more requests and producer is done
            if is_sending_end and not real_time_requests:
                break
            time.sleep(0.0001)
    return TTFTs, TPOTs


def run_real_time(traces: List[Dict], llm: LLM):
    """
    Launch producer-consumer threads for real-time simulation.

    Args:
        traces: List of trace dictionaries.
        llm: vLLM instance.

    Returns:
        TTFTs, TPOTs: Lists of metrics collected during simulation.
    """
    start_time = time.time()
    conn1, conn2 = Pipe()
    real_time_requests = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_simulate_real_time_request, traces, start_time, real_time_requests, conn1)
        f2 = executor.submit(_simulate_real_time_response, real_time_requests, llm, conn2)
        f1.result()          # Wait for producer to finish
        ttfts, tpots = f2.result()
    return ttfts, tpots