"""
批量推理模块
"""

import torch
from typing import List, Dict, Any
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor
import os

from ..models import QwenVLWrapper

class BatchInference:
    """批量推理类"""
    
    def __init__(self, model_wrapper: QwenVLWrapper, max_workers: int = 4):
        self.model_wrapper = model_wrapper
        self.max_workers = max_workers
    
    def process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个数据项
        
        Args:
            item: 数据项
            
        Returns:
            处理结果
        """
        try:
            image_path = item["image_path"]
            question = item["question"]
            
            answer = self.model_wrapper.visual_question_answering(image_path, question)
            
            return {
                "id": item.get("id", "unknown"),
                "image_path": image_path,
                "question": question,
                "answer": answer,
                "status": "success"
            }
        except Exception as e:
            return {
                "id": item.get("id", "unknown"),
                "image_path": item.get("image_path", ""),
                "question": item.get("question", ""),
                "answer": "",
                "status": f"error: {str(e)}"
            }
    
    def run_batch_inference(self, 
                          data: List[Dict[str, Any]],
                          output_file: str = None) -> List[Dict[str, Any]]:
        """
        运行批量推理
        
        Args:
            data: 数据列表
            output_file: 输出文件路径
            
        Returns:
            推理结果
        """
        results = []
        
        print(f"开始批量推理，共 {len(data)} 个样本")
        
        # 使用多线程处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {
                executor.submit(self.process_single_item, item): item 
                for item in data
            }
            
            for future in tqdm(future_to_item, desc="处理进度"):
                result = future.result()
                results.append(result)
        
        # 保存结果
        if output_file:
            self.save_results(results, output_file)
            print(f"结果已保存到: {output_file}")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """保存结果到文件"""
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def evaluate_accuracy(self, 
                        results: List[Dict[str, Any]],
                        ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估准确率（简化版）
        
        Args:
            results: 推理结果
            ground_truth: 真实标签
            
        Returns:
            评估指标
        """
        # 这里实现简单的关键词匹配评估
        # 实际应用中应该使用更复杂的评估方法
        
        correct_count = 0
        total_count = len(results)
        
        for result, truth in zip(results, ground_truth):
            if result["status"] == "success":
                # 简单的关键词匹配（实际应该使用更复杂的方法）
                predicted_answer = result["answer"].lower()
                true_answer = truth["answer"].lower()
                
                # 检查是否有共同的关键词
                common_words = set(predicted_answer.split()) & set(true_answer.split())
                if len(common_words) > 0:
                    correct_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "success_rate": len([r for r in results if r["status"] == "success"]) / total_count
        }