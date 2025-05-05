"""
GPU Utilities for Wafer Map Processing
This module handles GPU setup, memory management, and device selection for deep learning models.
"""

import torch
import numpy as np
import psutil
import os
from typing import Union, Tuple, Optional

class GPUManager:
    """Manages GPU resources and provides utility functions for deep learning models"""
    
    def __init__(self):
        self.device = self._get_device()
        self.gpu_info = self._get_gpu_info()
        
    def _get_device(self) -> torch.device:
        """Determine the best available device (CUDA GPU or CPU)"""
        if torch.cuda.is_available():
            # Get the GPU with the most free memory
            device_id = self._get_best_gpu()
            device = torch.device(f'cuda:{device_id}')
            print(f"🎮 Using CUDA GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
        else:
            device = torch.device('cpu')
            print("⚠️ No CUDA GPU available, using CPU")
        
        return device
    
    def _get_gpu_info(self) -> dict:
        """Get information about available GPU resources"""
        gpu_info = {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        }
        
        if gpu_info['available']:
            gpu_info.update({
                'device_name': torch.cuda.get_device_name(),
                'memory_allocated': torch.cuda.memory_allocated(),
                'memory_cached': torch.cuda.memory_reserved(),
            })
        
        return gpu_info
    
    def _get_best_gpu(self) -> int:
        """Select the GPU with the most available memory"""
        if not torch.cuda.is_available():
            return -1
        
        # Get memory info for all GPUs
        memory_available = []
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            memory_available.append(torch.cuda.get_device_properties(i).total_memory - 
                                 torch.cuda.memory_allocated())
        
        # Return the GPU with the most available memory
        return int(np.argmax(memory_available))
    
    def optimize_cuda_performance(self):
        """Apply various CUDA optimizations"""
        if not torch.cuda.is_available():
            return
        
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True
        
        # Enable TF32 on Ampere GPUs
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def get_memory_status(self) -> dict:
        """Get current memory usage statistics"""
        memory_status = {
            'system': {
                'total': psutil.virtual_memory().total / (1024**3),  # GB
                'available': psutil.virtual_memory().available / (1024**3),  # GB
                'used': psutil.virtual_memory().used / (1024**3),  # GB
                'percent': psutil.virtual_memory().percent
            }
        }
        
        if torch.cuda.is_available():
            memory_status['gpu'] = {
                'allocated': torch.cuda.memory_allocated() / (1024**3),  # GB
                'cached': torch.cuda.memory_reserved() / (1024**3),  # GB
                'max_allocated': torch.cuda.max_memory_allocated() / (1024**3)  # GB
            }
        
        return memory_status
    
    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU memory cache cleared")
    
    def to_device(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Move data to the current device"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data.to(self.device)
    
    def get_batch_size(self, model_size_mb: float) -> int:
        """Calculate optimal batch size based on available memory"""
        if not torch.cuda.is_available():
            return 8  # Default CPU batch size
        
        # Get available GPU memory in MB
        available_memory = (torch.cuda.get_device_properties(self.device).total_memory -
                          torch.cuda.memory_allocated()) / (1024 * 1024)
        
        # Reserve 20% memory for overhead
        available_memory *= 0.8
        
        # Calculate maximum possible batch size
        max_batch_size = int(available_memory / model_size_mb)
        
        # Ensure batch size is at least 1 and a power of 2
        batch_size = max(1, min(32, 2 ** int(np.log2(max_batch_size))))
        
        return batch_size

def setup_gpu_environment() -> GPUManager:
    """Initialize GPU environment and return a GPUManager instance"""
    # Set environment variables for better GPU performance
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_CUDA_ARCH_LIST'] = 'All'
    
    # Initialize GPU manager
    gpu_manager = GPUManager()
    
    # Apply optimizations
    gpu_manager.optimize_cuda_performance()
    
    # Print system info
    print("\n🖥️ System Information:")
    print(f"Python: {os.sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    return gpu_manager 