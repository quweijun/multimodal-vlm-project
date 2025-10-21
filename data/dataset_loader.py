"""
多模态数据集加载器
"""

import json
from PIL import Image
from torch.utils.data import Dataset
import torch
from typing import Dict, List, Optional, Union

class MultimodalDataset(Dataset):
    """多模态数据集类"""
    
    def __init__(self, 
                 data_path: str, 
                 processor=None,
                 max_length: int = 2048,
                 is_training: bool = True):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            processor: 处理器
            max_length: 最大序列长度
            is_training: 是否为训练模式
        """
        self.processor = processor
        self.max_length = max_length
        self.is_training = is_training
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载图像
        image_path = item["image_path"]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            # 返回空图像作为fallback
            image = Image.new('RGB', (224, 224), color='white')
        
        # 获取文本
        if "conversations" in item:
            # 多轮对话格式
            text = self._process_conversations(item["conversations"])
        else:
            # 单轮问答格式
            text = item["text"]
        
        return {
            "image": image,
            "text": text,
            "image_path": image_path
        }
    
    def _process_conversations(self, conversations: List[Dict]) -> str:
        """处理对话数据"""
        formatted_text = ""
        for conv in conversations:
            role = conv["from"]
            content = conv["value"]
            formatted_text += f"<|{role}|>{content}</s>"
        return formatted_text

class DataCollator:
    """多模态数据整理器"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]
        
        # 处理多模态输入
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        # 设置labels
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs