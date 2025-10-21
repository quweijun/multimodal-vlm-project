"""
全参数微调训练器
"""

import torch
from transformers import Trainer, TrainingArguments
from typing import Optional

from ..data import MultimodalDataset, DataCollator
from ..models import QwenVLWrapper
from ..configs import MultimodalTrainingConfig

class FullFinetuneTrainer:
    """全参数微调训练器"""
    
    def __init__(self, 
                 model_wrapper: QwenVLWrapper,
                 train_dataset: MultimodalDataset,
                 eval_dataset: Optional[MultimodalDataset] = None,
                 training_config: Optional[MultimodalTrainingConfig] = None):
        """
        初始化全参数微调训练器
        
        Args:
            model_wrapper: 模型包装器
            train_dataset: 训练数据集
            eval_dataset: 验证数据集
            training_config: 训练配置
        """
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_config = training_config or MultimodalTrainingConfig()
        
        # 启用梯度检查点以节省显存
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
    
    def train(self, output_dir: str = "./full_finetune_output"):
        """训练模型"""
        print("开始全参数微调...")
        
        # 转换为TrainingArguments
        training_args = self.training_config.to_training_args()
        training_args.output_dir = output_dir
        
        # 数据整理器
        data_collator = DataCollator(self.model_wrapper.processor)
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )
        
        # 开始训练
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        self.model_wrapper.processor.save_pretrained(output_dir)
        
        print(f"全参数微调完成，模型保存在: {output_dir}")
        
        return trainer