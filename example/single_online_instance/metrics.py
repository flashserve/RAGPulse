"""
RAGPulseMetrics: A class to collect and save metrics for RAG Pulse workload replay.
Metrics include TTFT (Time to First Token) and TPOT (Time per Output Token).
"""
import json
import os
from typing import Dict
from datetime import datetime

class RAGPulseMetrics:
    DEFAULT_METRIC_NAMES = ["TTFT","TPOT","Average_TTFT","Average_TPOT"]
    def __init__(self,metric_dir:str=None,metric_name:str=None,logger=None,args:Dict=None):
        """
        Initialize RAGPulseMetrics instance.
        Args:
            metric_dir: Directory to save metrics file. If None, defaults to "./metrics".
            metric_name: Name of the metrics file. If None, defaults to "rag_pulse_metrics_<timestamp>.json".
            logger: Optional RAGPulseLogger for logging.
        """
        # Currently, metrics are only TTFT,TPOT.
        self.TTFTs = []
        self.TPOTs = []
        if metric_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            metric_dir = os.path.join(current_dir, "metrics")
            os.makedirs(metric_dir, exist_ok=True)
        self.metric_dir = metric_dir
        if metric_name is None:
            today = datetime.now().strftime("%Y-%m-%d-%H-%M")
            self.metric_name = f"rag_pulse_metrics_{today}.json"
        self.logger = logger
        # Store args
        self.args = args

    def add_metrics(self,metrics:Dict[str, float]):
        """Add metrics to the current metrics list.
        Args:
            metrics: A dictionary containing metric names and their float values.
        """
        for metric_name in self.DEFAULT_METRIC_NAMES:
            if metric_name in metrics:
                if metric_name == "TTFT":
                    if metrics[metric_name] is not None:
                        self.TTFTs.append(metrics[metric_name])
                elif metric_name == "TPOT":
                    if metrics[metric_name] is not None:
                        self.TPOTs.append(metrics[metric_name])
    
    def save_metrics(self):
        """Save the collected metrics to a JSON file."""
        metric_path = os.path.join(self.metric_dir,self.metric_name)
        metrics_summary = {
            "TTFTs": self.TTFTs,
            "TPOTs": self.TPOTs,
            "Average_TTFT": sum(self.TTFTs)/len(self.TTFTs) if len(self.TTFTs)>0 else None,
            "Average_TPOT": sum(self.TPOTs)/len(self.TPOTs) if len(self.TPOTs)>0 else None,
            "Args": self.args
        }
        with open(metric_path,"w",encoding="utf-8") as f:
            json.dump(metrics_summary,f,indent=4)
        if self.logger:
            self.logger.info(f"Saved metrics summary to {metric_path}.")
        else:
            print(f"Saved metrics summary to {metric_path}.")
    