import time
import logging
from typing import Optional, Callable, Any
from functools import wraps
import os
from datetime import datetime

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "performance.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("performance")

class PerformanceMonitor:
    """Monitor and log performance metrics for key operations."""
    
    def __init__(self):
        self.metrics = {}
    
    def start_operation(self, operation_name: str) -> float:
        """Start timing an operation."""
        start_time = time.time()
        self.metrics[operation_name] = {
            'start_time': start_time,
            'end_time': None,
            'duration': None
        }
        return start_time
    
    def end_operation(self, operation_name: str) -> Optional[float]:
        """End timing an operation and log the duration."""
        if operation_name not in self.metrics:
            logger.warning(f"Operation {operation_name} was not started")
            return None
            
        end_time = time.time()
        self.metrics[operation_name]['end_time'] = end_time
        duration = end_time - self.metrics[operation_name]['start_time']
        self.metrics[operation_name]['duration'] = duration
        
        logger.info(f"Operation {operation_name} took {duration:.3f} seconds")
        return duration
    
    def get_operation_duration(self, operation_name: str) -> Optional[float]:
        """Get the duration of a completed operation."""
        if operation_name not in self.metrics or self.metrics[operation_name]['duration'] is None:
            return None
        return self.metrics[operation_name]['duration']
    
    def get_all_metrics(self) -> dict:
        """Get all recorded metrics."""
        return self.metrics

# Global performance monitor instance
monitor = PerformanceMonitor()

def measure_performance(operation_name: Optional[str] = None) -> Callable:
    """
    Decorator to measure the performance of a function.
    
    Args:
        operation_name: Optional name for the operation. If not provided,
                       the function name will be used.
    
    Returns:
        Decorated function that measures and logs its execution time.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            monitor.start_operation(op_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                monitor.end_operation(op_name)
        return wrapper
    return decorator

def log_performance_metrics() -> None:
    """Log all recorded performance metrics."""
    metrics = monitor.get_all_metrics()
    if not metrics:
        logger.info("No performance metrics recorded")
        return
        
    logger.info("\n=== Performance Metrics ===")
    for op_name, data in metrics.items():
        if data['duration'] is not None:
            logger.info(f"{op_name}: {data['duration']:.3f} seconds")
    logger.info("=========================\n") 