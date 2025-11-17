'''
RAGPulseLogger: Advanced logging class for RAG Pulse project

Usage:
    rag_logger = RAGPulseLogger(log_level="DEBUG")
    rag_logger.info("This is an info message.")
    rag_logger.error("This is an error message with details: {detail}", detail="File not found")
'''
import logging
import os
from typing import Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler
logging.basicConfig(
    level=logging.DEBUG,  # the minimum logging level
    format='【%(levelname)s】%(asctime)s  - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
class RAGPulseLogger:
    """
    Advanced logging class
    
    Features:
    - Outputs to both console and file simultaneously
    - Supports different log levels
    - Automatically creates log directory
    - Log format includes timestamp, level, module, and message
    """
    
    def __init__(
        self,
        logger_name: str = "RAG_Pulse_Logger",
        log_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        log_level: str = "DEBUG",
        log_format: Optional[str] = None,
        date_format: Optional[str] = None,
        max_bytes: int = 10485760,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize the logger
        
        Args:
            logger_name: Name of the logger instance
            log_dir: Directory to store log files, uses default if None
            log_file: Path to the log file, uses default path if None
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Custom log format string
            date_format: Custom date format string
            max_bytes: Maximum size for each log file
            backup_count: Number of backup log files to keep
        """
        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {log_level}. Must be one of {valid_levels}")
        
        # Create logger instance
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.propagate = False  # Prevent duplicate logging
        
        # Set default log format if not provided
        if log_format is None:
            log_format = (
                '【%(levelname)s】%(asctime)s   %(filename)s:%(lineno)d  %(message)s'
            )
        
        # Set default date format if not provided
        if date_format is None:
            date_format = '%Y-%m-%d %H:%M:%S'
        
        # Set default log file path if not provided
        if log_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(current_dir, "logs") if log_dir is None else log_dir
            os.makedirs(logs_dir, exist_ok=True)
            
            # Create log filename with current date
            today = datetime.now().strftime("%Y-%m-%d-%H-%M")
            log_file = os.path.join(logs_dir, f"rag_pulse_{today}.log")
        
        self.log_file = log_file
        
        # Create formatter and date formatter
        self.formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        
        # Configure handlers
        self._configure_handlers(max_bytes, backup_count)
    
    def _configure_handlers(self, max_bytes: int, backup_count: int):
        """Configure log handlers"""
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        console_handler.setLevel(logging.INFO)  # Console shows INFO and above
        self.logger.addHandler(console_handler)
        
        # Create file handler with rotation
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(self.formatter)
        file_handler.setLevel(logging.DEBUG)  # File records all levels
        self.logger.addHandler(file_handler)
    
    def log(self, message: str, level: str = "INFO", **kwargs):
        """
        Log a message
        
        Args:
            message: The log message
            level: Logging level
            **kwargs: Additional parameters for string formatting
        """
        level = level.upper()
        
        # Support string formatting
        if kwargs:
            try:
                message = message.format(**kwargs)
            except KeyError as e:
                self.logger.error(f"Log formatting error: {e}, message: {message}")
                return
        
        # Log based on level
        log_method = getattr(self.logger, level.lower(), None)
        if log_method is not None and callable(log_method):
            log_method(message)
        else:
            self.logger.warning(f"Unknown log level '{level}', using default level INFO")
            self.logger.info(message)
    
    def debug(self, message: str, **kwargs):
        """Log a DEBUG level message"""
        self.log(message, "DEBUG",** kwargs)
    
    def info(self, message: str, **kwargs):
        """Log an INFO level message"""
        self.log(message, "INFO", **kwargs)
    
    def warning(self, message: str,** kwargs):
        """Log a WARNING level message"""
        self.log(message, "WARNING", **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log an ERROR level message"""
        self.log(message, "ERROR",** kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log a CRITICAL level message"""
        self.log(message, "CRITICAL",** kwargs)
    
    def exception(self, message: str, exc_info: bool = True, **kwargs):
        """Log an exception with traceback"""
        if kwargs:
            message = message.format(**kwargs)
        self.logger.error(message, exc_info=exc_info)
    
    def set_level(self, level: str):
        """Set the logging level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {level}. Must be one of {valid_levels}")
        
        self.logger.setLevel(getattr(logging, level.upper()))
    
    def get_log_file_path(self) -> str:
        """Get the path to the log file"""
        return self.log_file
    
    def close(self):
        """Close all log handlers"""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
    
    def __del__(self):
        """Destructor to ensure proper resource cleanup"""
        self.close()


def test_logger():
    rag_logger = RAGPulseLogger(log_level="DEBUG")
    rag_logger.debug("This is a debug message.")
    rag_logger.info("This is an info message.")
    rag_logger.warning("This is a warning message.")
    rag_logger.error("This is an error message.")
    rag_logger.critical("This is a critical message.")
    try:
        1 / 0
    except ZeroDivisionError:
        rag_logger.exception("An exception occurred: Division by zero.")


if __name__ == "__main__":
    test_logger()