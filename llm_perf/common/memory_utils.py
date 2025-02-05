import os
import psutil
import gc
from typing import Dict, Optional
from loguru import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Memory thresholds in MB
MEMORY_THRESHOLDS = {
    'cpu_rss': 8192,  # 8GB
    'cpu_percent': 90,  # 90%
    'gpu_allocated': 8192  # 8GB
}

class MemoryTracker:
    def __init__(self):
        self.initial_memory: Dict = {}
        self.peak_memory: Dict = {
            'cpu_rss': 0,
            'cpu_percent': 0,
            'gpu_allocated': 0
        }
        self.consecutive_increases = 0
        self.last_memory: Optional[Dict] = None
        self.before_memory: Optional[Dict] = None  # Store memory state before benchmark

    def get_gpu_memory_info(self):
        """Get GPU memory usage if CUDA is available"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None
        
        try:
            gpu_memory = []
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)  # MB
                reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)    # MB
                gpu_memory.append({
                    'device': i,
                    'allocated': allocated,
                    'reserved': reserved
                })
            return gpu_memory
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return None

    def get_cpu_memory_info(self):
        """Get CPU memory usage"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                'rss': memory_info.rss / (1024 * 1024),    # MB
                'vms': memory_info.vms / (1024 * 1024),    # MB
                'percent': process.memory_percent()
            }
        except Exception as e:
            logger.warning(f"Failed to get CPU memory info: {e}")
            return None

    def check_thresholds(self, cpu_info: Optional[Dict], gpu_info: Optional[list]):
        """Check if memory usage exceeds thresholds"""
        if cpu_info:
            if cpu_info['rss'] > MEMORY_THRESHOLDS['cpu_rss']:
                logger.warning(f"CPU RSS memory ({cpu_info['rss']:.0f}MB) exceeds threshold ({MEMORY_THRESHOLDS['cpu_rss']}MB)")
            if cpu_info['percent'] > MEMORY_THRESHOLDS['cpu_percent']:
                logger.warning(f"CPU usage ({cpu_info['percent']:.1f}%) exceeds threshold ({MEMORY_THRESHOLDS['cpu_percent']}%)")

        if gpu_info:
            for device in gpu_info:
                if device['allocated'] > MEMORY_THRESHOLDS['gpu_allocated']:
                    logger.warning(
                        f"GPU {device['device']} allocated memory ({device['allocated']:.0f}MB) "
                        f"exceeds threshold ({MEMORY_THRESHOLDS['gpu_allocated']}MB)"
                    )

    def check_persistent_growth(self, cpu_info: Optional[Dict], gpu_info: Optional[list]):
        """Monitor for persistent memory growth"""
        if not cpu_info:
            return

        current_memory = {
            'cpu_rss': cpu_info['rss'],
            'cpu_percent': cpu_info['percent'],
            'gpu_allocated': gpu_info[0]['allocated'] if gpu_info else 0
        }

        # Update peak memory
        for key in self.peak_memory:
            self.peak_memory[key] = max(self.peak_memory[key], current_memory[key])

        # Check for persistent growth
        if self.last_memory:
            is_increasing = all(
                current_memory[key] > self.last_memory[key] * 1.05  # 5% increase threshold
                for key in current_memory
            )
            
            if is_increasing:
                self.consecutive_increases += 1
                if self.consecutive_increases >= 3:  # Alert after 3 consecutive increases
                    logger.warning(
                        "Detected persistent memory growth over last 3 benchmarks:\n"
                        f"Initial: CPU RSS={self.initial_memory.get('cpu_rss', 0):.0f}MB\n"
                        f"Current: CPU RSS={current_memory['cpu_rss']:.0f}MB\n"
                        f"Peak: CPU RSS={self.peak_memory['cpu_rss']:.0f}MB"
                    )
            else:
                self.consecutive_increases = 0

        # Store current memory for next comparison
        self.last_memory = current_memory

        # Store initial memory on first run
        if not self.initial_memory:
            self.initial_memory = current_memory

    def log_memory_usage(self, phase: str = "current"):
        """Log current memory usage for both CPU and GPU"""
        # Force garbage collection
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get memory info
        cpu_info = self.get_cpu_memory_info()
        gpu_info = self.get_gpu_memory_info()

        # Check thresholds and persistent growth
        self.check_thresholds(cpu_info, gpu_info)
        self.check_persistent_growth(cpu_info, gpu_info)
        
        # Store before memory state
        if phase == "before":
            self.before_memory = {
                'cpu': cpu_info,
                'gpu': gpu_info
            }
            prefix = "Before benchmark -"
        elif phase == "after" and self.before_memory:
            prefix = "After benchmark -"
        else:
            prefix = "Current -"
        
        # Log CPU memory
        if cpu_info:
            cpu_msg = f"{prefix} CPU Memory - RSS: {cpu_info['rss']:.2f}MB, VMS: {cpu_info['vms']:.2f}MB, Percent: {cpu_info['percent']:.1f}%"
            
            # Add delta if we're in after phase
            if phase == "after" and self.before_memory and self.before_memory['cpu']:
                before_cpu = self.before_memory['cpu']
                cpu_msg += f" (Δ RSS: {cpu_info['rss'] - before_cpu['rss']:+.2f}MB, Δ VMS: {cpu_info['vms'] - before_cpu['vms']:+.2f}MB, Δ %: {cpu_info['percent'] - before_cpu['percent']:+.1f})"
            
            logger.info(cpu_msg)
        
        # Log GPU memory if available
        if gpu_info:
            for device in gpu_info:
                gpu_msg = f"{prefix} GPU {device['device']} Memory - Allocated: {device['allocated']:.2f}MB, Reserved: {device['reserved']:.2f}MB"
                
                # Add delta if we're in after phase
                if phase == "after" and self.before_memory and self.before_memory['gpu']:
                    before_gpu = next((g for g in self.before_memory['gpu'] if g['device'] == device['device']), None)
                    if before_gpu:
                        gpu_msg += f" (Δ Allocated: {device['allocated'] - before_gpu['allocated']:+.2f}MB, Δ Reserved: {device['reserved'] - before_gpu['reserved']:+.2f}MB)"
                
                logger.info(gpu_msg)

# Create a global memory tracker instance
memory_tracker = MemoryTracker()

# Function to use in other modules
def log_memory_usage(phase: str = "current"):
    """Global function to log memory usage"""
    memory_tracker.log_memory_usage(phase) 