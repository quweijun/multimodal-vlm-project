"""
数据预处理工具
"""

import os
import json
from typing import List, Dict, Any
from PIL import Image
import base64
from io import BytesIO

class DataPreprocessor:
    """数据预处理器"""
    
    @staticmethod
    def convert_to_multimodal_format(data: List[Dict], output_path: str):
        """
        转换为多模态训练格式
        
        Args:
            data: 原始数据
            output_path: 输出路径
        """
        formatted_data = []
        
        for item in data:
            formatted_item = {
                "id": item.get("id", len(formatted_data)),
                "image_path": item["image_path"],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{item['question']}"
                    },
                    {
                        "from": "gpt", 
                        "value": item["answer"]
                    }
                ]
            }
            formatted_data.append(formatted_item)
        
        # 保存格式化数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def image_to_base64(image_path: str) -> str:
        """图像转base64"""
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode()
    
    @staticmethod
    def create_conversation_data(questions: List[str], 
                               answers: List[str], 
                               image_paths: List[str]) -> List[Dict]:
        """创建对话数据"""
        assert len(questions) == len(answers) == len(image_paths)
        
        data = []
        for i, (q, a, img_path) in enumerate(zip(questions, answers, image_paths)):
            data.append({
                "id": i,
                "image_path": img_path,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{q}"},
                    {"from": "gpt", "value": a}
                ]
            })
        
        return data