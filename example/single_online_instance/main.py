"""
Main script to run RAG Pulse workload replay against a single online instance.
"""
import argparse
from logger import RAGPulseLogger
from preprocess_data import PreprocessData
from online_server import OnlineServer
from metrics import RAGPulseMetrics

def get_args()->argparse.Namespace:
    '''
    Get the command line arguments.
    1. api_key: The OpenAI API key for authentication.
    2. api_base: The base URL of the online instance.
    3. model_name: The model name to use for generation.
    4. trace_dir: The directory of trace files.
    5. log_dir: The directory to save logs.
    6. time_scale_factor: The time scale factor to speed up the workload replay.
    7. use_trace_num: The number of traces to use for workload replay,only
    use the front of traces.
    8. tokenizer_path: The path of tokenizer.
    9. streaming: Whether to use streaming generation.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="dummy_key",
                        help="The OpenAI API key for authentication.")
    parser.add_argument("--api_base", type=str, default="http://0.0.0.0:8000/v1",
                        help="The base URL of the online instance.")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-14B-Instruct",
                        help="The model name to use for generation.")
    parser.add_argument("--trace_dir", type=str, default="../../data",
                        help="The directory of trace files.")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="The directory to save logs.")
    parser.add_argument("--time_scale_factor",type=int,default=10,
                        help="The time scale factor to speed up the workload replay.")
    parser.add_argument("--use_trace_num",type=int,default=5,
                        help="The number of traces to use for workload replay,only use the front of traces.")
    parser.add_argument("--tokenizer_path", type=str, default="/home/share/models/Qwen2.5-14B-Instruct",
                        help="The path of tokenizer.")
    parser.add_argument("--streaming", type=bool, default=True, help="Whether to use streaming generation.")
    args = parser.parse_args()
    return args
    

def main(args:argparse.Namespace):
    # Initialize logger
    rag_pulse_logger = RAGPulseLogger(log_level="DEBUG")
    # Initialize metrics collector
    metrics = RAGPulseMetrics(logger=rag_pulse_logger, args=vars(args))

    rag_pulse_logger.info(f"Arguments: {args}")
    rag_pulse_logger.info("RAG Pulse Online Instance Logger initialized.")
    # Preprocess traces and hash_ids
    rag_pulse_logger.info("Preprocessing traces and hash_ids...")
    # Initialize PreprocessData
    p = PreprocessData(
        args.trace_dir, 
        args.tokenizer_path,
        args.use_trace_num,
        rag_pulse_logger
    )
    # Preprocess data
    traces = p.preprocess()
    rag_pulse_logger.info(f"Preprocessed {len(traces)} traces.")
    server = OnlineServer(
        api_key=args.api_key,
        api_base=args.api_base,
        model_name=args.model_name,
        streaming=args.streaming,
        logger=rag_pulse_logger,
        metrics=metrics
    )
    rag_pulse_logger.info("Starting workload replay...")
    # Run the online server to process traces and save metrics
    server.run(traces, args.time_scale_factor)
    rag_pulse_logger.info("Workload replay finished.")
    server.close()

if __name__ == "__main__":
    args = get_args()
    main(args)