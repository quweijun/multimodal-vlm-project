"""
可视化工具
"""

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
from typing import List, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualization:
    """可视化类"""
    
    @staticmethod
    def plot_training_metrics(metrics: Dict[str, List[float]], 
                            save_path: str = None):
        """
        绘制训练指标
        
        Args:
            metrics: 训练指标
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics_to_plot = {
            'loss': '训练损失',
            'learning_rate': '学习率',
            'epoch': '训练轮次'
        }
        
        for i, (metric, title) in enumerate(metrics_to_plot.items()):
            if metric in metrics:
                axes[i].plot(metrics[metric])
                axes[i].set_title(title)
                axes[i].set_xlabel('步骤')
                axes[i].set_ylabel(metric)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def create_interactive_qa_visualization(questions: List[str],
                                          answers: List[str],
                                          images: List[Image.Image] = None):
        """
        创建交互式问答可视化
        
        Args:
            questions: 问题列表
            answers: 答案列表
            images: 图像列表（可选）
        """
        fig = make_subplots(
            rows=len(questions), 
            cols=2 if images else 1,
            subplot_titles=[f"问题 {i+1}" for i in range(len(questions))],
            specs=[[{"secondary_y": False}] * (2 if images else 1)] * len(questions)
        )
        
        for i, (question, answer) in enumerate(zip(questions, answers)):
            # 添加文本内容
            fig.add_annotation(
                x=0.5, y=1 - (i * 0.2),
                text=f"<b>问题:</b> {question}<br><b>回答:</b> {answer}",
                showarrow=False,
                xref="paper", yref="paper",
                align="left"
            )
        
        fig.update_layout(
            title="多模态问答结果可视化",
            showlegend=False,
            height=200 * len(questions)
        )
        
        fig.show()
    
    @staticmethod
    def plot_keyword_frequency(keywords: Dict[str, int], 
                             top_n: int = 20,
                             save_path: str = None):
        """
        绘制关键词频率
        
        Args:
            keywords: 关键词频率字典
            top_n: 显示前n个关键词
            save_path: 保存路径
        """
        top_keywords = dict(sorted(keywords.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:top_n])
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(top_keywords.values()), 
                   y=list(top_keywords.keys()))
        plt.title(f'前{top_n}个关键词频率')
        plt.xlabel('频率')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()