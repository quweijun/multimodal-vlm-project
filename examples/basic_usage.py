"""
基础使用示例
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import QwenVLWrapper
from inference import VisualGrounding, SpatialReasoning
from utils import ImageUtils

def basic_chat_demo():
    """基础对话演示"""
    print("=== Qwen-VL 基础对话演示 ===")
    
    # 初始化模型
    model = QwenVLWrapper(model_name="Qwen/Qwen3-VL-4B-Instruct")
    
    # 示例图像和问题
    image_path = "data/sample_data/demo_images/demo.jpg"  # 请准备测试图像
    question = "请描述这张图片的内容"
    
    # 进行对话
    response = model.chat(image_path, question)
    print(f"问题: {question}")
    print(f"回答: {response}")
    print()

def visual_qa_demo():
    """视觉问答演示"""
    print("=== 视觉问答演示 ===")
    
    model = QwenVLWrapper()
    grounding = VisualGrounding(model)
    reasoning = SpatialReasoning(model)
    
    image_path = "data/sample_data/demo_images/scene.jpg"
    
    # 对象定位
    location_result = grounding.object_localization(
        image_path, "红色的物体"
    )
    print(f"对象定位结果: {location_result}")
    
    # 空间关系分析
    spatial_result = grounding.spatial_relationship(
        image_path, "人", "车"
    )
    print(f"空间关系分析: {spatial_result}")
    
    # 场景布局分析
    layout_result = reasoning.analyze_scene_layout(image_path)
    print(f"场景布局分析: {layout_result}")

def image_captioning_demo():
    """图像描述演示"""
    print("=== 图像描述演示 ===")
    
    model = QwenVLWrapper()
    
    image_path = "data/sample_data/demo_images/landscape.jpg"
    
    # 图像描述
    caption = model.image_captioning(image_path)
    print(f"图像描述: {caption}")

def batch_processing_demo():
    """批量处理演示"""
    print("=== 批量处理演示 ===")
    
    from inference import BatchInference
    
    model = QwenVLWrapper()
    batch_processor = BatchInference(model)
    
    # 示例数据
    sample_data = [
        {
            "id": 1,
            "image_path": "data/sample_data/demo_images/img1.jpg",
            "question": "图片中有什么？"
        },
        {
            "id": 2,
            "image_path": "data/sample_data/demo_images/img2.jpg",
            "question": "主要颜色是什么？"
        },
        {
            "id": 3,
            "image_path": "data/sample_data/demo_images/img3.jpg",
            "question": "这是什么场景？"
        }
    ]
    
    # 批量处理
    results = batch_processor.run_batch_inference(
        sample_data,
        output_file="output/batch_results.json"
    )
    
    print("批量处理完成！")
    for result in results:
        print(f"ID: {result['id']}, 状态: {result['status']}")
        print(f"回答: {result['answer'][:100]}...")
        print()

if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("output", exist_ok=True)
    
    # 运行演示
    basic_chat_demo()
    visual_qa_demo()
    image_captioning_demo()
    batch_processing_demo()