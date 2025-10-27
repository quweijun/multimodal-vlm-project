"""
微调演示示例
"""

import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.qwen_vl_wrapper import QwenVLWrapper
from training.lora_trainer import LoRATrainer
from data.dataset_loader import MultimodalDataset
from data.data_preprocessor import DataPreprocessor

def prepare_sample_data():
    """准备示例训练数据"""
    sample_data = [
        {
            "id": 1,
            "image_path": "data/sample_data/demo_images/train1.jpg",
            "question": "这张图片的主要内容是什么？",
            "answer": "这是一张风景照片，展示了美丽的山脉和湖泊。"
        },
        {
            "id": 2,
            "image_path": "data/sample_data/demo_images/train2.jpg",
            "question": "图片中有多少人？",
            "answer": "图片中有三个人，他们正在交谈。"
        },
        {
            "id": 3,
            "image_path": "data/sample_data/demo_images/train3.jpg",
            "question": "这是什么动物？",
            "answer": "这是一只猫，它正在睡觉。"
        }
    ]
    
    # 保存示例数据
    os.makedirs("data/training_data", exist_ok=True)
    with open("data/training_data/sample_train.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print("示例训练数据已准备完成")

def lora_finetuning_demo():
    """LoRA微调演示"""
    print("=== LoRA微调演示 ===")
    
    # 准备数据
    prepare_sample_data()
    
    # 初始化模型
    model = QwenVLWrapper()
    
    # 加载数据集
    train_dataset = MultimodalDataset(
        "data/training_data/sample_train.json",
        processor=model.processor,
        is_training=True
    )
    
    # 配置训练参数
    training_config = {
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 1,  # 小批量以适应显存
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 1,  # 演示用1个epoch
        "logging_steps": 10,
        "save_steps": 50,
    }
    
    # 创建训练器
    trainer = LoRATrainer(
        model_wrapper=model,
        train_dataset=train_dataset,
        training_config=training_config
    )
    
    print("开始LoRA微调...")
    # 注意：在实际演示中，由于数据量小，训练可能很快完成
    # trainer.train(output_dir="./output/lora_demo")
    print("LoRA微调演示完成（实际训练已注释）")

def create_custom_dataset():
    """创建自定义数据集示例"""
    print("=== 创建自定义数据集 ===")
    
    # 示例数据
    questions = [
        "描述这张图片",
        "图片中的主要物体是什么？",
        "这是什么场景？"
    ]
    
    answers = [
        "这是一张美丽的风景照片",
        "主要物体是山脉和树木",
        "这是一个自然风光场景"
    ]
    
    image_paths = [
        "data/sample_data/demo_images/img1.jpg",
        "data/sample_data/demo_images/img2.jpg", 
        "data/sample_data/demo_images/img3.jpg"
    ]
    
    # 创建数据集
    custom_data = DataPreprocessor.create_conversation_data(
        questions, answers, image_paths
    )
    
    # 保存数据集
    os.makedirs("data/custom_data", exist_ok=True)
    with open("data/custom_data/custom_dataset.json", "w", encoding="utf-8") as f:
        json.dump(custom_data, f, ensure_ascii=False, indent=2)
    
    print("自定义数据集已创建: data/custom_data/custom_dataset.json")

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs("output", exist_ok=True)
    os.makedirs("data/training_data", exist_ok=True)
    os.makedirs("data/custom_data", exist_ok=True)
    
    # 运行演示
    create_custom_dataset()
    lora_finetuning_demo()