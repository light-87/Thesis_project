"""Memory management utilities for phosphorylation prediction."""

import gc
import psutil
import torch
import numpy as np
from typing import Dict, Optional, Tuple, Any
import warnings

class MemoryManager:
    """Utilities for monitoring and managing memory usage."""
    
    def __init__(self, gpu_id: Optional[int] = None):
        """
        Initialize memory manager.
        
        Args:
            gpu_id: GPU device ID to monitor. If None, uses current device.
        """
        self.gpu_id = gpu_id
        self.initial_memory = self.get_memory_info()
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory information in GB.
        """
        info = {}
        
        # CPU memory
        process = psutil.Process()
        memory_info = process.memory_info()
        info['cpu_rss'] = memory_info.rss / (1024**3)  # GB
        info['cpu_vms'] = memory_info.vms / (1024**3)  # GB
        info['cpu_percent'] = process.memory_percent()
        
        # System memory
        system_memory = psutil.virtual_memory()
        info['system_total'] = system_memory.total / (1024**3)  # GB
        info['system_available'] = system_memory.available / (1024**3)  # GB
        info['system_percent'] = system_memory.percent
        
        # GPU memory
        if torch.cuda.is_available():
            device = self.gpu_id if self.gpu_id is not None else torch.cuda.current_device()
            try:
                info['gpu_allocated'] = torch.cuda.memory_allocated(device) / (1024**3)  # GB
                info['gpu_cached'] = torch.cuda.memory_reserved(device) / (1024**3)  # GB
                info['gpu_total'] = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
                info['gpu_free'] = info['gpu_total'] - info['gpu_allocated']
            except RuntimeError:
                info['gpu_allocated'] = 0.0
                info['gpu_cached'] = 0.0
                info['gpu_total'] = 0.0
                info['gpu_free'] = 0.0
        else:
            info['gpu_allocated'] = 0.0
            info['gpu_cached'] = 0.0
            info['gpu_total'] = 0.0
            info['gpu_free'] = 0.0
        
        return info
    
    def print_memory_info(self, prefix: str = ""):
        """Print current memory usage."""
        info = self.get_memory_info()
        if prefix:
            print(f"\n=== {prefix} ===")
        else:
            print("\n=== Memory Usage ===")
        
        print(f"CPU RSS: {info['cpu_rss']:.2f} GB ({info['cpu_percent']:.1f}%)")
        print(f"System: {info['system_total'] - info['system_available']:.2f} / {info['system_total']:.2f} GB ({info['system_percent']:.1f}%)")
        
        if info['gpu_total'] > 0:
            print(f"GPU: {info['gpu_allocated']:.2f} / {info['gpu_total']:.2f} GB ({info['gpu_allocated']/info['gpu_total']*100:.1f}%)")
            print(f"GPU Cached: {info['gpu_cached']:.2f} GB")
    
    def clear_gpu_cache(self):
        """Clear GPU cache if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def clear_memory(self):
        """Clear both CPU and GPU memory."""
        gc.collect()
        self.clear_gpu_cache()
    
    def get_memory_delta(self) -> Dict[str, float]:
        """Get memory usage delta since initialization."""
        current = self.get_memory_info()
        delta = {}
        for key, value in current.items():
            delta[key] = value - self.initial_memory.get(key, 0)
        return delta
    
    def estimate_batch_size(self, 
                          model_memory_gb: float,
                          sample_size_mb: float,
                          safety_factor: float = 0.8) -> int:
        """
        Estimate optimal batch size based on available GPU memory.
        
        Args:
            model_memory_gb: Model memory usage in GB
            sample_size_mb: Memory per sample in MB
            safety_factor: Safety factor to avoid OOM (0.0-1.0)
        
        Returns:
            Estimated batch size
        """
        if not torch.cuda.is_available():
            return 32  # Default for CPU
        
        info = self.get_memory_info()
        available_gb = info['gpu_free'] * safety_factor
        available_for_batch = available_gb - model_memory_gb
        
        if available_for_batch <= 0:
            return 1
        
        batch_size = int((available_for_batch * 1024) / sample_size_mb)
        return max(1, batch_size)


class MemoryProfiler:
    """Context manager for profiling memory usage."""
    
    def __init__(self, name: str = "Operation", gpu_id: Optional[int] = None):
        """
        Initialize memory profiler.
        
        Args:
            name: Name of the operation being profiled
            gpu_id: GPU device ID to monitor
        """
        self.name = name
        self.manager = MemoryManager(gpu_id)
        self.start_memory = None
        self.peak_memory = None
    
    def __enter__(self):
        """Start memory profiling."""
        self.start_memory = self.manager.get_memory_info()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End memory profiling and print results."""
        end_memory = self.manager.get_memory_info()
        
        print(f"\n=== Memory Profile: {self.name} ===")
        
        # CPU memory delta
        cpu_delta = end_memory['cpu_rss'] - self.start_memory['cpu_rss']
        print(f"CPU Memory Delta: {cpu_delta:+.2f} GB")
        
        # GPU memory delta
        if torch.cuda.is_available():
            gpu_delta = end_memory['gpu_allocated'] - self.start_memory['gpu_allocated']
            print(f"GPU Memory Delta: {gpu_delta:+.2f} GB")
        
        # Final memory usage
        print(f"Final CPU: {end_memory['cpu_rss']:.2f} GB")
        if torch.cuda.is_available():
            print(f"Final GPU: {end_memory['gpu_allocated']:.2f} GB")


def optimize_memory_usage():
    """Apply general memory optimization settings."""
    # Garbage collection
    gc.collect()
    
    # PyTorch optimizations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except AttributeError:
            pass
    
    # NumPy optimizations
    # Disable automatic memory mapping for large arrays
    np.seterr(over='ignore')


def check_memory_requirements(required_gb: float, 
                            device: str = 'auto') -> bool:
    """
    Check if sufficient memory is available.
    
    Args:
        required_gb: Required memory in GB
        device: Device type ('cpu', 'gpu', or 'auto')
    
    Returns:
        True if sufficient memory is available
    """
    manager = MemoryManager()
    info = manager.get_memory_info()
    
    if device == 'cpu' or (device == 'auto' and not torch.cuda.is_available()):
        available = info['system_available']
    elif device == 'gpu' or (device == 'auto' and torch.cuda.is_available()):
        available = info['gpu_free']
    else:
        available = min(info['system_available'], info['gpu_free'])
    
    if available < required_gb:
        warnings.warn(
            f"Insufficient memory: {available:.2f} GB available, "
            f"{required_gb:.2f} GB required"
        )
        return False
    
    return True


def get_optimal_num_workers() -> int:
    """Get optimal number of workers for data loading."""
    try:
        # Use 80% of available CPU cores, but at least 1
        num_cores = psutil.cpu_count(logical=False)
        return max(1, int(num_cores * 0.8))
    except:
        return 4  # Safe default