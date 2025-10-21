"""
图像处理工具
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
import base64
from io import BytesIO

class ImageUtils:
    """图像工具类"""
    
    @staticmethod
    def resize_image(image: Image.Image, 
                    max_size: Tuple[int, int] = (1024, 1024),
                    keep_ratio: bool = True) -> Image.Image:
        """
        调整图像大小
        
        Args:
            image: 输入图像
            max_size: 最大尺寸 (宽, 高)
            keep_ratio: 是否保持宽高比
            
        Returns:
            调整后的图像
        """
        if keep_ratio:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        else:
            image = image.resize(max_size, Image.Resampling.LANCZOS)
        
        return image
    
    @staticmethod
    def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
        """PIL图像转OpenCV格式"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
        """OpenCV图像转PIL格式"""
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    
    @staticmethod
    def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
        """图像转base64字符串"""
        buffered = BytesIO()
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode()
    
    @staticmethod
    def base64_to_image(base64_string: str) -> Image.Image:
        """base64字符串转图像"""
        image_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(image_data))
    
    @staticmethod
    def add_text_to_image(image: Image.Image, 
                         text: str,
                         position: Tuple[int, int] = (10, 10),
                         font_size: int = 20,
                         color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """
        在图像上添加文本
        
        Args:
            image: 输入图像
            text: 要添加的文本
            position: 文本位置
            font_size: 字体大小
            color: 文本颜色
            
        Returns:
            添加文本后的图像
        """
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        draw.text(position, text, fill=color, font=font)
        return image
    
    @staticmethod
    def create_image_grid(images: List[Image.Image], 
                         grid_size: Tuple[int, int] = None) -> Image.Image:
        """
        创建图像网格
        
        Args:
            images: 图像列表
            grid_size: 网格尺寸 (列数, 行数)
            
        Returns:
            网格图像
        """
        if not grid_size:
            grid_size = (int(np.ceil(np.sqrt(len(images)))), 
                        int(np.ceil(np.sqrt(len(images)))))
        
        cols, rows = grid_size
        width, height = images[0].size
        
        grid_image = Image.new('RGB', (cols * width, rows * height))
        
        for i, img in enumerate(images):
            if i >= cols * rows:
                break
            row = i // cols
            col = i % cols
            grid_image.paste(img, (col * width, row * height))
        
        return grid_image