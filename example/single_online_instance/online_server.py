"""
Online server implementation for handling requests to an online instance.

Usage:
    server = OnlineServer(api_key, api_base, model_name, streaming, logger)
    server.run(traces, time_scale_factor)
    server.close()
"""
import time
import queue
from threading import Thread
from openai import OpenAI
from logger import RAGPulseLogger
from typing import List,Tuple
from metrics import RAGPulseMetrics

class OnlineServerEndSignal:
    pass

class OnlineServer:
    def __init__(self, api_key: str, api_base: str, model_name:str,streaming: bool, 
                 logger: RAGPulseLogger, metrics: RAGPulseMetrics):
        """
        Initialize OnlineServer instance.
        Args:
            api_key: API key for authentication.
            api_base: Base URL of the online instance.
            model_name: Model name to use for generation.
            streaming: Whether to use streaming generation.
            logger: RAGPulseLogger instance for logging.
        """
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.streaming = streaming
        self.logger = logger
        self.metrics = metrics
        self.logger.info("Initializing OnlineServer...")
        self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        self.request_queue:queue.Queue[Tuple[str,int]] = queue.Queue()
        self.request_thread=Thread(target=self.request_worker,args=())
        self.request_thread.start()
        self.logger.info("OnlineServer initialized.")

    def request_worker(self, ):
        """
        Worker thread to process requests from the queue.
        """
        while True:
            item = self.request_queue.get()
            if isinstance(item, OnlineServerEndSignal):
                self.logger.info("Request worker received end signal. Exiting...")
                break
            input_text, receive_time = item
            self.response(input_text,receive_time)

    def response(self, input_text: str, receive_time: int): 
        """
        Send request to the online instance and handle response.
        Args:
            input_text: The input text for generation.
            receive_time: The time when the request was received.
        """
        TTFT = None
        TPOT = None
        if self.streaming:
            chunk_count = 0
            response_chunks = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": input_text}
                ],
                temperature=0.0,
                max_tokens=300,
                stream=True
            )
            self.logger.info("Started streaming response...")
            self.logger.info("Time is {}".format(time.time()-receive_time))
            full_response = ""
            for chunk in response_chunks:
                cur_time = time.time()
                if TTFT is None:
                    TTFT = cur_time - receive_time
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    chunk_count += 1
            ed_time = time.time()
            TPOT = (ed_time - receive_time) / chunk_count if chunk_count > 0 else 0.0
            self.logger.info(f"Response: {full_response}")
            self.logger.info(f"Streaming response completed. TTFT: {TTFT:.2f}s, TPOT: {TPOT:.4f}s")
            # Record metrics
            self.metrics.add_metrics({"TTFT": TTFT, "TPOT": TPOT})
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": input_text}
                ],
                temperature=0.0,
                max_tokens=300
            )
            self.logger.info(f"In this way,we can not calculate TTFT and TPOT without streaming temporarily.")
            self.logger.info(f"Response: {response}")

    def run(self,traces:List[dict], time_scale_factor:int):
        """
        Simulate real-time requests based on traces.
        Args:
            traces: A list of trace dictionaries containing input_text and timestamp.
            time_scale_factor: A factor to scale the timing of requests.
        """
        self.logger.info("Starting simulation of real-time requests...")
        # Simulate real-time requests based on traces
        start_time = time.time()
        trace_idx = 0
        n_traces = len(traces)
        while trace_idx < n_traces:
            current_time = time.time()
            if current_time - start_time >= int(traces[trace_idx]["timestamp"]) / time_scale_factor:
                self.request_queue.put((traces[trace_idx]["input_text"], current_time))
                self.logger.info(f"Sent request at time {current_time - start_time:.2f}s with input_text length {len(traces[trace_idx]['input_text'])} and orginal scaling timestamp {int(traces[trace_idx]["timestamp"]) / time_scale_factor}.")
                trace_idx += 1
            else:
                time.sleep(0.0001)  # sleep for 0.1ms to avoid busy waiting
        
    def close(self):
        """
        Close the online server and clean up resources.
        """
        if self.request_thread is not None:
            self.request_queue.put(OnlineServerEndSignal())
            self.request_thread.join()
            self.logger.info("OnlineServer has closed.")
            self.metrics.save_metrics()
            self.logger.info("Metrics have been saved.")
    
    def __del__(self):
        """
        Destructor to ensure resources are cleaned up.
        """
        self.close()