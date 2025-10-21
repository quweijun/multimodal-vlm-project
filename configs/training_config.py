"""
训练配置
"""

from dataclasses import dataclass
from transformers import TrainingArguments

@dataclass
class MultimodalTrainingConfig:
    """多模态训练配置"""
    
    # 基础配置
    output_dir: str = "./output"
    overwrite_output_dir: bool = True
    
    # 训练参数
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    max_steps: int = -1
    
    # 优化器
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 学习率调度
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    
    # 日志和保存
    logging_dir: str = "./logs"
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    evaluation_strategy: str = "steps"
    save_total_limit: int = 3
    
    # 其他配置
    remove_unused_columns: bool = False
    ddp_find_unused_parameters: bool = False
    dataloader_pin_memory: bool = False
    
    def to_training_args(self):
        """转换为TrainingArguments"""
        return TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=self.overwrite_output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            weight_decay=self.weight_decay,
            adam_epsilon=self.adam_epsilon,
            max_grad_norm=self.max_grad_norm,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_steps=self.warmup_steps,
            logging_dir=self.logging_dir,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            evaluation_strategy=self.evaluation_strategy,
            save_total_limit=self.save_total_limit,
            remove_unused_columns=self.remove_unused_columns,
            ddp_find_unused_parameters=self.ddp_find_unused_parameters,
            dataloader_pin_memory=self.dataloader_pin_memory,
        )