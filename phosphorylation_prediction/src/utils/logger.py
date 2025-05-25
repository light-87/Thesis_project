"""Logging utilities for phosphorylation prediction."""

import os
import logging
import sys
from typing import Optional
from datetime import datetime


def setup_logger(name: str, log_file: Optional[str] = None, 
                level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set level
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_colored_logger(name: str, log_file: Optional[str] = None, 
                        level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with colored console output.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Configured logger with colored output
    """
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set level
    logger.setLevel(level)
    
    # Create formatters
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False
    
    return logger


class ExperimentLogger:
    """Enhanced logger for experiment tracking."""
    
    def __init__(self, experiment_name: str, output_dir: str):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Output directory for logs
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        
        # Create log directory
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup main logger
        log_file = os.path.join(self.log_dir, "experiment.log")
        self.logger = setup_colored_logger(experiment_name, log_file)
        
        # Setup separate loggers for different components
        self.data_logger = setup_logger(
            f"{experiment_name}.data",
            os.path.join(self.log_dir, "data.log")
        )
        
        self.model_logger = setup_logger(
            f"{experiment_name}.model",
            os.path.join(self.log_dir, "model.log")
        )
        
        self.eval_logger = setup_logger(
            f"{experiment_name}.evaluation",
            os.path.join(self.log_dir, "evaluation.log")
        )
        
        # Log experiment start
        self.logger.info(f"Experiment '{experiment_name}' started")
        self.logger.info(f"Logs will be saved to: {self.log_dir}")
    
    def log_config(self, config: dict) -> None:
        """Log experiment configuration."""
        self.logger.info("Experiment Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_data_info(self, info: dict) -> None:
        """Log data information."""
        self.data_logger.info("Data Information:")
        for key, value in info.items():
            self.data_logger.info(f"  {key}: {value}")
    
    def log_model_info(self, model_name: str, info: dict) -> None:
        """Log model information."""
        self.model_logger.info(f"Model '{model_name}' Information:")
        for key, value in info.items():
            self.model_logger.info(f"  {key}: {value}")
    
    def log_metrics(self, metrics: dict, prefix: str = "") -> None:
        """Log evaluation metrics."""
        prefix_str = f"{prefix} " if prefix else ""
        self.eval_logger.info(f"{prefix_str}Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                self.eval_logger.info(f"  {metric}: {value:.4f}")
            else:
                self.eval_logger.info(f"  {metric}: {value}")
    
    def log_experiment_end(self, duration: float) -> None:
        """Log experiment completion."""
        self.logger.info(f"Experiment '{self.experiment_name}' completed")
        self.logger.info(f"Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    def get_logger(self, component: str = "main") -> logging.Logger:
        """
        Get logger for specific component.
        
        Args:
            component: Component name ('main', 'data', 'model', 'evaluation')
            
        Returns:
            Logger instance
        """
        if component == "data":
            return self.data_logger
        elif component == "model":
            return self.model_logger
        elif component == "evaluation":
            return self.eval_logger
        else:
            return self.logger


def log_memory_usage(logger: logging.Logger, message: str = "") -> None:
    """
    Log current memory usage.
    
    Args:
        logger: Logger instance
        message: Optional message prefix
    """
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        message_str = f"{message} " if message else ""
        logger.info(f"{message_str}Memory usage: {memory_mb:.2f} MB")
        
    except ImportError:
        logger.warning("psutil not available, cannot log memory usage")
    except Exception as e:
        logger.warning(f"Error logging memory usage: {e}")


def log_gpu_usage(logger: logging.Logger, message: str = "") -> None:
    """
    Log current GPU usage.
    
    Args:
        logger: Logger instance
        message: Optional message prefix
    """
    try:
        import torch
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024    # MB
            
            message_str = f"{message} " if message else ""
            logger.info(
                f"{message_str}GPU memory - Allocated: {memory_allocated:.2f} MB, "
                f"Reserved: {memory_reserved:.2f} MB"
            )
        else:
            logger.info("CUDA not available")
            
    except ImportError:
        logger.warning("PyTorch not available, cannot log GPU usage")
    except Exception as e:
        logger.warning(f"Error logging GPU usage: {e}")


class ProgressLogger:
    """Logger for tracking progress with timing information."""
    
    def __init__(self, logger: logging.Logger, total_steps: int, 
                 log_frequency: int = 10):
        """
        Initialize progress logger.
        
        Args:
            logger: Base logger instance
            total_steps: Total number of steps
            log_frequency: Log progress every N steps
        """
        self.logger = logger
        self.total_steps = total_steps
        self.log_frequency = log_frequency
        self.current_step = 0
        self.start_time = None
    
    def start(self) -> None:
        """Start progress tracking."""
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting progress tracking for {self.total_steps} steps")
    
    def update(self, step: Optional[int] = None, message: str = "") -> None:
        """
        Update progress.
        
        Args:
            step: Current step (if None, increment by 1)
            message: Optional message to include
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        if (self.current_step % self.log_frequency == 0 or 
            self.current_step == self.total_steps):
            
            self._log_progress(message)
    
    def _log_progress(self, message: str = "") -> None:
        """Log current progress."""
        if self.start_time is None:
            return
        
        import time
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        progress_pct = (self.current_step / self.total_steps) * 100
        
        # Estimate remaining time
        if self.current_step > 0:
            time_per_step = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            estimated_remaining = time_per_step * remaining_steps
        else:
            estimated_remaining = 0
        
        message_str = f" - {message}" if message else ""
        
        self.logger.info(
            f"Progress: {self.current_step}/{self.total_steps} ({progress_pct:.1f}%) "
            f"- Elapsed: {elapsed:.1f}s - ETA: {estimated_remaining:.1f}s{message_str}"
        )
    
    def finish(self, message: str = "") -> None:
        """Finish progress tracking."""
        if self.start_time is None:
            return
        
        import time
        total_time = time.time() - self.start_time
        
        message_str = f" - {message}" if message else ""
        self.logger.info(
            f"Progress completed: {self.total_steps}/{self.total_steps} "
            f"- Total time: {total_time:.1f}s{message_str}"
        )