"""
视觉定位演示示例
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
from PIL import Image

from models.qwen_vl_wrapper import QwenVLWrapper
from inference.visual_grounding import VisualGrounding
from inference.spatial_reasoning import SpatialReasoning
from utils.image_utils import ImageUtils

# from models import QwenVLWrapper
# from inference import VisualGrounding, SpatialReasoning
# from utils import ImageUtils

def visual_grounding_comprehensive_demo():
    """综合视觉定位演示"""
    print("=== 综合视觉定位演示 ===")
    
    # 初始化模型和工具
    model = QwenVLWrapper()
    grounding = VisualGrounding(model)
    reasoning = SpatialReasoning(model)
    
    # 示例图像路径（请准备测试图像）
    image_path = "data/sample_data/demo_images/complex_scene.jpg"
    
    if not os.path.exists(image_path):
        print(f"测试图像不存在: {image_path}")
        print("请准备测试图像或修改路径")
        return
    
    # 1. 对象定位演示
    print("\n1. 对象定位:")
    objects_to_locate = ["人", "车", "建筑", "树木"]
    for obj in objects_to_locate:
        result = grounding.object_localization(image_path, obj)
        print(f"定位 '{obj}': {result}")
    
    # 2. 空间关系分析
    print("\n2. 空间关系分析:")
    relationships = [
        ("人", "车"),
        ("建筑", "树木"),
        ("天空", "地面")
    ]
    for obj1, obj2 in relationships:
        result = grounding.spatial_relationship(image_path, obj1, obj2)
        print(f"'{obj1}' 和 '{obj2}' 的关系: {result}")
    
    # 3. 场景布局分析
    print("\n3. 场景布局分析:")
    layout_analysis = reasoning.analyze_scene_layout(image_path)
    print(f"场景布局: {layout_analysis}")
    
    # 4. 深度估计
    print("\n4. 深度估计:")
    depth_analysis = reasoning.depth_estimation(image_path)
    print(f"深度分析: {depth_analysis}")
    
    # 5. 视觉常识推理
    print("\n5. 视觉常识推理:")
    commonsense_reasoning = reasoning.visual_commonsense_reasoning(image_path)
    print(f"常识推理: {commonsense_reasoning}")

def interactive_grounding_demo():
    """交互式视觉定位演示"""
    print("=== 交互式视觉定位演示 ===")
    
    model = QwenVLWrapper()
    grounding = VisualGrounding(model)
    
    # 示例图像
    image_path = "data/sample_data/demo_images/interactive.jpg"
    
    if not os.path.exists(image_path):
        print(f"测试图像不存在: {image_path}")
        return
    
    # 交互式问答循环
    print("输入 'quit' 退出交互")
    while True:
        user_input = input("\n请输入关于图像的问题: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            break
        
        if user_input:
            try:
                response = model.chat(image_path, user_input)
                print(f"模型回答: {response}")
            except Exception as e:
                print(f"处理错误: {e}")

def visualization_demo():
    """可视化演示"""
    print("=== 可视化演示 ===")
    #函数单独调用时加上以下两行
    model = QwenVLWrapper()
    grounding = VisualGrounding(model)
    # 创建示例图像
    width, height = 800, 600
    image = Image.new('RGB', (width, height), color='lightblue')
    
    # 转换为OpenCV格式进行绘制
    cv_image = ImageUtils.pil_to_cv2(image)
    
    # 示例边界框和标签
    boxes = [
        (100, 100, 300, 200),  # (x1, y1, x2, y2)
        (400, 150, 600, 300),
        (200, 400, 500, 500)
    ]
    
    labels = ["物体A", "物体B", "物体C"]
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    
    # 绘制边界框
    print("Boxes:", boxes)
    result_image = grounding.draw_bounding_boxes(cv_image, boxes, labels, colors)
    #result_image = grounding.draw_bounding_boxes(cv_image, boxes, labels, colors)
    
    # 保存结果
    output_path = "output/visualization_demo.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"可视化结果已保存: {output_path}")

def performance_benchmark():
    """性能基准测试"""
    print("=== 性能基准测试 ===")
    
    import time
    from inference.batch_inference import BatchInference
    
    model = QwenVLWrapper()
    batch_processor = BatchInference(model, max_workers=2)
    
    # 创建测试数据
    test_data = []
    for i in range(5):  # 测试5个样本
        test_data.append({
            "id": i,
            "image_path": "data/sample_data/demo_images/test.jpg",  # 使用同一张图像测试
            "question": f"测试问题 {i+1}: 描述这张图片"
        })
    
    # 性能测试
    start_time = time.time()
    
    results = batch_processor.run_batch_inference(test_data)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 统计结果
    success_count = sum(1 for r in results if r["status"] == "success")
    avg_time_per_sample = total_time / len(test_data)
    
    print(f"总样本数: {len(test_data)}")
    print(f"成功处理: {success_count}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每个样本: {avg_time_per_sample:.2f} 秒")
    print(f"吞吐量: {len(test_data)/total_time:.2f} 样本/秒")

if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("output", exist_ok=True)
    
    # 运行演示
    # visual_grounding_comprehensive_demo()
    # interactive_grounding_demo()
    visualization_demo()
    performance_benchmark()