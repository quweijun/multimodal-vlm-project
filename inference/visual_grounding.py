"""
视觉定位推理
"""

import torch
from PIL import Image
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from ..models import QwenVLWrapper

class VisualGrounding:
    """视觉定位类"""
    
    def __init__(self, model_wrapper: QwenVLWrapper):
        self.model_wrapper = model_wrapper
        self.processor = model_wrapper.processor
        self.model = model_wrapper.model
    
    def object_localization(self, 
                          image_path: str, 
                          object_description: str) -> str:
        """
        对象定位
        
        Args:
            image_path: 图像路径
            object_description: 对象描述
            
        Returns:
            定位结果描述
        """
        prompt = f"请定位并描述图像中的 {object_description} 的位置"
        return self.model_wrapper.chat(image_path, prompt)
    
    def spatial_relationship(self,
                           image_path: str,
                           object1: str,
                           object2: str) -> str:
        """
        空间关系分析
        
        Args:
            image_path: 图像路径
            object1: 对象1
            object2: 对象2
            
        Returns:
            空间关系描述
        """
        prompt = f"请分析图像中 {object1} 和 {object2} 之间的空间关系"
        return self.model_wrapper.chat(image_path, prompt)
    
    def draw_bounding_boxes(self, 
                          image: np.ndarray, 
                          boxes: List[Tuple[int, int, int, int]],
                          labels: List[str],
                          colors: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
        """
        在图像上绘制边界框
        
        Args:
            image: 输入图像
            boxes: 边界框列表 (x1, y1, x2, y2)
            labels: 标签列表
            colors: 颜色列表
            
        Returns:
            绘制后的图像
        """
        if colors is None:
            colors = [(0, 255, 0)] * len(boxes)  # 默认绿色
        
        img_with_boxes = image.copy()
        
        for box, label, color in zip(boxes, labels, colors):
            x1, y1, x2, y2 = box
            
            # 绘制边界框
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img_with_boxes, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # 绘制标签文本
            cv2.putText(img_with_boxes, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return img_with_boxes
    
    def region_description(self,
                         image_path: str,
                         region_coords: Tuple[int, int, int, int]) -> str:
        """
        区域描述
        
        Args:
            image_path: 图像路径
            region_coords: 区域坐标 (x1, y1, x2, y2)
            
        Returns:
            区域描述
        """
        x1, y1, x2, y2 = region_coords
        prompt = f"请描述图像中坐标区域 ({x1}, {y1}) 到 ({x2}, {y2}) 内的内容"
        return self.model_wrapper.chat(image_path, prompt)