"""
Logging utilities for AHSD pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import time

def setup_logging(level: str = "INFO", 
                 log_file: Optional[str] = None,
                 console: bool = True) -> logging.Logger:
    """
    Setup logging configuration for AHSD pipeline.
    
    Parameters:
    -----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str, optional
        Path to log file. If None, only console logging
    console : bool
        Whether to log to console
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Create logger
    logger = logging.getLogger('ahsd')
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class TimingLogger:
    """Context manager for timing operations with logging."""
    
    def __init__(self, logger: logging.Logger, operation: str, level: str = "INFO"):
        self.logger = logger
        self.operation = operation
        self.level = getattr(logging, level.upper())
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.level, f"Starting {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is None:
            self.logger.log(self.level, f"Completed {self.operation} in {duration:.2f}s")
        else:
            self.logger.error(f"Failed {self.operation} after {duration:.2f}s: {exc_val}")

def log_memory_usage(logger: logging.Logger, message: str = "Memory usage"):
    """Log current memory usage."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"{message}: {memory_mb:.1f} MB")
        
    except ImportError:
        logger.debug("psutil not available for memory monitoring")
    except Exception as e:
        logger.debug(f"Failed to get memory usage: {e}")

def log_system_info(logger: logging.Logger):
    """Log system information."""
    try:
        import psutil
        import torch
        
        # CPU info
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1024**3
        
        logger.info(f"System: {cpu_count} CPUs, {memory_gb:.1f} GB RAM")
        
        # GPU info
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_count}x {gpu_name}")
        else:
            logger.info("No CUDA GPU available")
            
    except Exception as e:
        logger.debug(f"Failed to get system info: {e}")
