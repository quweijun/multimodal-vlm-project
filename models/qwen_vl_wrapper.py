"""
Qwen-VL模型封装
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import List, Dict, Optional, Union
import warnings

class QwenVLWrapper:
    """Qwen-VL模型封装类"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
                 device: str = "auto",
                 torch_dtype: torch.dtype = torch.bfloat16):
        """
        初始化Qwen-VL模型
        
        Args:
            model_name: 模型名称
            device: 设备
            torch_dtype: 数据类型
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        
        # 加载模型和处理器
        self.processor = AutoProcessor.from_pretrained(model_name)
        # self.model = Qwen2VLForConditionalGeneration.from_pretrained(
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )
        
        print(f"成功加载模型: {model_name}")
    
    def chat(self, 
             image: Union[Image.Image, str],
             text: str,
             max_new_tokens: int = 512,
             temperature: float = 0.7,
             do_sample: bool = False) -> str:
        """
        与模型对话
        
        Args:
            image: 图像或图像路径
            text: 输入文本
            max_new_tokens: 最大生成长度
            temperature: 温度参数
            do_sample: 是否采样
            
        Returns:
            模型回复
        """
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text}
                ]
            }
        ]
        
        # 应用聊天模板
        formatted_text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 处理输入
        inputs = self.processor(
            text=[formatted_text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # 生成回复
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # 提取生成的文本
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return response[0]
    
    def batch_chat(self, 
                   image_text_pairs: List[tuple],
                   max_new_tokens: int = 512) -> List[str]:
        """
        批量对话
        
        Args:
            image_text_pairs: (image, text)对列表
            max_new_tokens: 最大生成长度
            
        Returns:
            回复列表
        """
        responses = []
        
        for image, text in image_text_pairs:
            response = self.chat(image, text, max_new_tokens)
            responses.append(response)
        
        return responses
    
    def visual_question_answering(self, 
                                image_path: str, 
                                question: str) -> str:
        """
        视觉问答
        
        Args:
            image_path: 图像路径
            question: 问题
            
        Returns:
            答案
        """
        return self.chat(image_path, question)
    
    def image_captioning(self, image_path: str) -> str:
        """
        图像描述
        
        Args:
            image_path: 图像路径
            
        Returns:
            图像描述
        """
        prompt = "请详细描述这张图片的内容"
        return self.chat(image_path, prompt)