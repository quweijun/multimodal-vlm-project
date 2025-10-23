"""
LoRA微调训练器
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from typing import Optional, Dict, Any

from data.dataset_loader import MultimodalDataset, DataCollator
from models.qwen_vl_wrapper import QwenVLWrapper
from configs.model_config import get_lora_config
from configs.training_config import MultimodalTrainingConfig

class LoRATrainer:
    """LoRA微调训练器"""
    
    def __init__(self, 
                 model_wrapper: QwenVLWrapper,
                 train_dataset: MultimodalDataset,
                 eval_dataset: Optional[MultimodalDataset] = None,
                 training_config: Optional[Dict] = None):
        """
        初始化LoRA训练器
        
        Args:
            model_wrapper: 模型包装器
            train_dataset: 训练数据集
            eval_dataset: 验证数据集
            training_config: 训练配置
        """
        self.model_wrapper = model_wrapper
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # 配置
        self.training_config = training_config or {}
        self.lora_config = get_lora_config()
        
        # 准备模型
        self._setup_model()
    
    def _setup_model(self):
        """设置LoRA模型"""
        print("设置LoRA微调...")
        
        # 创建LoRA配置
        lora_config = LoraConfig(
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["lora_alpha"],
            lora_dropout=self.lora_config["lora_dropout"],
            bias=self.lora_config["bias"],
            target_modules=self.lora_config["target_modules"],
            task_type="CAUSAL_LM",
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model_wrapper.model, lora_config)
        self.model.print_trainable_parameters()
    
    def train(self, output_dir: str = "./lora_output"):
        """训练模型"""
        print("开始LoRA训练...")
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.training_config.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=self.training_config.get("per_device_eval_batch_size", 2),
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 4),
            learning_rate=self.training_config.get("learning_rate", 5e-5),
            num_train_epochs=self.training_config.get("num_train_epochs", 3),
            logging_dir=f"{output_dir}/logs",
            logging_steps=self.training_config.get("logging_steps", 50),
            save_steps=self.training_config.get("save_steps", 500),
            evaluation_strategy="steps" if self.eval_dataset else "no",
            eval_steps=self.training_config.get("eval_steps", 500),
            save_total_limit=self.training_config.get("save_total_limit", 3),
            remove_unused_columns=False,
            report_to=["tensorboard"],
            ddp_find_unused_parameters=False,
        )
        
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
        
        print(f"训练完成，模型保存在: {output_dir}")
        
        return trainer