# RAGPulse Demo

<!-- ![alt text](logo.webp.png)
# Demo Use of RAGPulse -->

This demo use of RAGPulse provides a simple request generator for vLLM serving framework.

## Project Structure

| Path / File | Purpose | Notes |
|---|---|---|
| `main.py` | Entry point: runs the full example workflow | Sequentially invokes preprocessing, replay and metrics collection |
| `preprocess_data.py` | Load traces and map hash_ids to token lists | Produces `{hash_id: token_list}` mappings and a chunk-level index |
| `model_pool.py` | Model initialization and orchestration | Builds `vllm.LLM` and tokenizer; supports real and synthetic generation modes |
| `real_time_runner.py` | Trace replay and execution | `run_real_time()` replays requests (multi-threaded) and captures TTFT/TPOT |
| `metrics.py` | Metrics aggregation and reporting | `aggregate_metrics()` computes latency/throughput summaries |
| `data/0_trace.jsonl` | Per-request trace metadata | Key fields: `timestamp`, `hash_ids` |
| `data/1_sys_prompt.jsonl` … `data/5_web_search.jsonl` | Context corpora mapping hash_id -> token_length | One `{hash_id: length}` entry per line; used for reconstruction and analysis |


## Prepare Environment

- Clone this repo
```sh
git clone https://github.com/flashserve/RAGPulse.git
```

- Install requirments in a new venv, just run

```sh
conda create -n myenv python=3.12 -y
conda activate myenv
pip install vllm==0.6.2
```

```sh
cd example
```
## Path Configuration
Configure the trace and model paths in the get_args() function within the main.py file.

## Execution
```sh
cd example
python main.py
```


##  Evaluation Metrics

We adopt the two metrics widely used in the LLM-serving community:

1. **TTFT** (Time-To-First-Token)  
   Latency from HTTP request sent to first token received.
2. **TPOT** (Time-Per-Output-Token)  
   Average time to generate each subsequent token, computed as  
   `(last_token_time ? first_token_time) / generated_token_count`.

Auxiliary metrics:
* **Throughput** = `total_output_tokens / total_duration`  
* **Goodput** = fraction of requests that meet the SLO (default SLO: TTFT ≤ 200 ms and TPOT ≤ 20 ms)

## Expected / Sample Run Output

Below is a **single-run log** captured from the console (abridged for clarity).  
Your numbers will differ slightly depending on GPU model, driver, and system load.

```
True
Loaded 7106 traces from 0_trace.jsonl.
Loaded vocab size: 151643 from /root/share/models/Qwen2.5-7B-Instruct/vocab.json.
Loaded 26924 hash_ids from ['1_sys_prompt.jsonl', '2_passages.jsonl', '3_history.jsonl', '4_user_input.jsonl', '5_web_search.jsonl'].
INFO Initializing an LLM engine (v0.6.1.dev238+ge2c6e0a82) with config: model='/root/share/models/Qwen2.5-7B-Instruct', max_seq_len=20000, enforce_eager=True, enable_prefix_caching=True
INFO Loading model weights took 14.25 GB
INFO # GPU blocks: 2055, # CPU blocks: 4681
INFO Automatic prefix caching is enabled.
Sent request 1/5 at time 0.00s
Sent request 2/5 at time 2.00s
Sent request 3/5 at time 6.00s
Sent request 4/5 at time 6.00s
Sent request 5/5 at time 8.00s
Processed prompts: 100%|█| 1/1 [00:07<00:00, 7.05s/it, est. speed input: 434.76 toks/s, output: 42.58 toks/s]
```

After the last request finishes the benchmark prints:
```
Average TTFT: 10.484 s
Average TPOT: 0.024 s
```

---

<!-- <details> -->
<summary style="padding: 8px 0;">
  <h2 id="overview" style="font-size: 2em; margin: 0; display: inline-block; border-bottom: none;">Main Characters</h2>
</summary>

<div style="margin-top: -18px;">


 - **Hash -> Document**
   - Goal: rebuild readable text from privacy-preserving hash_ids for analysis and replay.
   - Approach: load hash->metadata, fetch referenced chunks in request order, then merge/trim boundaries to form coherent passages while keeping provenance.

 - **Trace time-scaling**
   - Goal: generate time-scaled traces for replay or simulation to test different load patterns.
   - Approach: treat events as a time series and apply simple transforms (global scaling, segment stretch/compress, rate resampling) plus optional jitter, keeping relative ordering and dependencies.

 - **Model generation**
   - Goal: emulate the generation step to measure latency and output characteristics.
   - Approach: either run the real model to collect exact metrics, or use a lightweight statistical proxy sampled from observed output-length and token-rate distributions to estimate generation cost.
  
<!-- <details> -->

