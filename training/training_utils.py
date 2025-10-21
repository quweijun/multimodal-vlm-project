"""
训练工具函数
"""

import torch
import os
from typing import Dict, Any, List
import json
from transformers import TrainerCallback

class TrainingUtils:
    """训练工具类"""
    
    @staticmethod
    def setup_training_environment():
        """设置训练环境"""
        # 设置CUDA设备
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 启用TF32以获得更好的性能（Ampere架构及以上）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    @staticmethod
    def calculate_model_size(model):
        """计算模型大小"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    @staticmethod
    def save_training_metrics(metrics: Dict[str, Any], filepath: str):
        """保存训练指标"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def load_training_metrics(filepath: str) -> Dict[str, Any]:
        """加载训练指标"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

class MemoryUsageCallback(TrainerCallback):
    """内存使用回调"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and torch.cuda.is_available():
            logs["memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
            logs["memory_reserved"] = torch.cuda.memory_reserved() / 1024**3   # GB

class ProgressCallback(TrainerCallback):
    """训练进度回调"""
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"开始第 {state.epoch} 轮训练")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"第 {state.epoch} 轮训练完成")
    
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            print(f"训练步骤: {state.global_step}")