# embedder
import os
import logging
from typing import List, Optional, Dict, Any

from langchain_community.embeddings import OllamaEmbeddings
import numpy as np

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Embedder:
    """
    一个用于文本嵌入的封装类。
    使用 LangChain 的 OllamaEmbeddings 作为后端。
    """
    
    def __init__(self, model_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        初始化嵌入器。
        
        Args:
            model_name: 要使用的 Ollama 嵌入模型名称。
            config: 额外的配置参数，将传递给 OllamaEmbeddings。
        """
        # 从环境变量获取配置，或使用默认值
        self.model_name = model_name or os.getenv("EMBED_MODEL", "qwen3-embedding:8b")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # 合并配置
        self.config = config or {}
        self.config.setdefault("model", self.model_name)
        self.config.setdefault("base_url", self.base_url)
        
        self.embeddings: Optional[OllamaEmbeddings] = None
        
        try:
            logger.info(f"正在初始化嵌入模型: '{self.model_name}' (base_url: {self.base_url})")
            self.embeddings = OllamaEmbeddings(**self.config)
            logger.info(f"嵌入模型 '{self.model_name}' 初始化成功。")
        except Exception as e:
            logger.error(f"初始化嵌入模型 '{self.model_name}' 失败: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表编码为向量列表。
        
        Args:
            texts: 要嵌入的文本字符串列表。
            
        Returns:
            一个向量列表，其中每个向量对应输入列表中的一个文本。
        """
        if not self.embeddings:
            raise RuntimeError("嵌入模型未初始化。")
        
        if not texts:
            logger.warning("没有提供文本进行嵌入。")
            return []
            
        try:
            logger.info(f"正在为 {len(texts)} 个文本生成嵌入...")
            # 使用 LangChain 的 embed_documents 方法
            embeddings_list = self.embeddings.embed_documents(texts)
            logger.info(f"成功生成 {len(embeddings_list)} 个嵌入向量。")
            return embeddings_list
        except Exception as e:
            logger.error(f"生成文本嵌入时发生错误: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        为单个查询文本生成嵌入。
        
        Args:
            text: 要嵌入的查询文本。
            
        Returns:
            对应的嵌入向量。
        """
        if not self.embeddings:
            raise RuntimeError("嵌入模型未初始化。")
        
        if not text:
            logger.warning("没有提供查询文本进行嵌入。")
            return []
            
        try:
            # 使用 LangChain 的 embed_query 方法
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"生成查询嵌入时发生错误: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """
        获取当前模型生成的嵌入向量的维度。
        
        Returns:
            嵌入向量的维度。
        """
        if not self.embeddings:
            raise RuntimeError("嵌入模型未初始化。")
        
        # 通过嵌入一个简单的文本获取维度
        try:
            sample_embedding = self.embed_query("test")
            return len(sample_embedding)
        except Exception as e:
            logger.error(f"获取嵌入维度失败: {e}")
            raise

    @staticmethod
    def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度。
        
        Args:
            vec1: 第一个向量。
            vec2: 第二个向量。
            
        Returns:
            余弦相似度分数，范围在 [-1, 1] 之间。
        """
        try:
            # 转换为 numpy 数组
            arr1 = np.array(vec1)
            arr2 = np.array(vec2)
            
            # 计算点积
            dot_product = np.dot(arr1, arr2)
            
            # 计算向量的 L2 范数
            norm_vec1 = np.linalg.norm(arr1)
            norm_vec2 = np.linalg.norm(arr2)
            
            # 计算余弦相似度
            if norm_vec1 == 0 or norm_vec2 == 0:
                return 0.0
            return float(dot_product / (norm_vec1 * norm_vec2))
        except Exception as e:
            logger.error(f"计算余弦相似度失败: {e}")
            raise

# 创建一个全局实例，方便其他模块直接导入使用
# 这样可以确保模型只被加载一次
embedder = Embedder()

# 导出供外部使用的主要函数和实例
__all__ = [
    "Embedder",
    "embedder",
    "embed_texts",
    "embed_query",
    "get_embedding_dimension",
    "calculate_similarity"
]

# 导出常用函数
embed_texts = embedder.embed_texts
embed_query = embedder.embed_query
get_embedding_dimension = embedder.get_embedding_dimension
calculate_similarity = Embedder.calculate_similarity