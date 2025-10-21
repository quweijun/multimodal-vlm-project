"""
文本处理工具
"""

import re
import jieba
from typing import List, Dict, Any
from collections import Counter

class TextUtils:
    """文本工具类"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        清理文本
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        return text.strip()
    
    @staticmethod
    def extract_keywords(text: str, top_k: int = 10) -> List[str]:
        """
        提取关键词（中文）
        
        Args:
            text: 输入文本
            top_k: 返回前k个关键词
            
        Returns:
            关键词列表
        """
        # 使用jieba进行分词
        words = jieba.cut(text)
        
        # 过滤停用词和短词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        filtered_words = [word for word in words if len(word) > 1 and word not in stop_words]
        
        # 统计词频
        word_freq = Counter(filtered_words)
        
        return [word for word, freq in word_freq.most_common(top_k)]
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        计算文本相似度（基于Jaccard相似度）
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度得分 (0-1)
        """
        words1 = set(jieba.cut(text1))
        words2 = set(jieba.cut(text2))
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """
        将文本分割成句子
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        # 简单的中文句子分割
        sentences = re.split(r'[。！？!?]', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def analyze_response_quality(response: str, 
                               min_length: int = 10) -> Dict[str, Any]:
        """
        分析回复质量
        
        Args:
            response: 模型回复
            min_length: 最小有效长度
            
        Returns:
            质量分析结果
        """
        sentences = TextUtils.split_into_sentences(response)
        keywords = TextUtils.extract_keywords(response)
        
        return {
            "length": len(response),
            "sentence_count": len(sentences),
            "keyword_count": len(keywords),
            "is_meaningful": len(response) >= min_length,
            "has_multiple_sentences": len(sentences) > 1,
            "keywords": keywords
        }