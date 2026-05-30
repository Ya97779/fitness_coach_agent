"""文档加载器 - 支持多种格式，带重试机制"""

import os
import time
from typing import List, Optional
from functools import wraps
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredImageLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    TextLoader
)
from langchain_core.embeddings import Embeddings

logger = {"errors": [], "loaded": [], "failed": []}

MAX_RETRIES = 3
RETRY_DELAY = 1.0


def retry_on_failure(max_retries: int = MAX_RETRIES, delay: float = RETRY_DELAY):
    """重试装饰器

    Args:
        max_retries: 最大重试次数
        delay: 重试间隔（秒）
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        time.sleep(delay)
            raise last_error
        return wrapper
    return decorator


class DocumentLoader:
    """统一文档加载器

    支持格式：PDF, DOCX, TXT, HTML, 图片
    特性：
    - 自动检测文件类型
    - 重试机制
    - 详细日志记录
    """

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".txt": TextLoader,
        ".html": UnstructuredHTMLLoader,
        ".htm": UnstructuredHTMLLoader,
        ".jpg": UnstructuredImageLoader,
        ".jpeg": UnstructuredImageLoader,
        ".png": UnstructuredImageLoader,
    }

    def __init__(self, knowledge_base_dir: str = "./knowledge_base"):
        """初始化加载器

        Args:
            knowledge_base_dir: 知识库根目录
        """
        self.knowledge_base_dir = knowledge_base_dir
        self._reset_logger()

    def _reset_logger(self):
        """重置日志"""
        logger["errors"] = []
        logger["loaded"] = []
        logger["failed"] = []

    def get_loader(self, file_path: str):
        """根据文件扩展名获取对应加载器

        Args:
            file_path: 文件路径

        Returns:
            对应的 Loader 类
        """
        ext = os.path.splitext(file_path)[1].lower()
        loader_class = self.LOADER_MAP.get(ext)

        if loader_class is None:
            raise ValueError(f"不支持的文件格式: {ext}")

        return loader_class

    @retry_on_failure(max_retries=MAX_RETRIES, delay=RETRY_DELAY)
    def load_single_file(self, file_path: str) -> List[Document]:
        """加载单个文件（带重试）

        Args:
            file_path: 文件路径

        Returns:
            Document 列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.LOADER_MAP:
            raise ValueError(f"不支持的文件格式: {ext}")

        loader_class = self.LOADER_MAP[ext]
        loader = loader_class(file_path)
        docs = loader.load()

        logger["loaded"].append(file_path)
        return docs

    def load_directory(self, directory: str = None) -> List[Document]:
        """加载目录下的所有支持的文件

        Args:
            directory: 目录路径，默认使用 knowledge_base_dir

        Returns:
            所有加载的 Document
        """
        self._reset_logger()
        directory = directory or self.knowledge_base_dir

        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger["errors"].append(f"目录不存在，已创建: {directory}")
            return []

        all_docs = []
        supported_exts = set(self.LOADER_MAP.keys())

        for root, _, files in os.walk(directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1].lower()

                if ext not in supported_exts:
                    continue

                try:
                    docs = self.load_single_file(file_path)
                    all_docs.extend(docs)
                except Exception as e:
                    logger["failed"].append({
                        "file": file_path,
                        "error": str(e)
                    })
                    logger["errors"].append(
                        f"加载失败 [{filename}]: {str(e)}"
                    )

        return all_docs

    def get_load_report(self) -> dict:
        """获取加载报告

        Returns:
            包含加载统计信息的字典
        """
        return {
            "total_loaded": len(logger["loaded"]),
            "total_failed": len(logger["failed"]),
            "loaded_files": logger["loaded"],
            "failed_files": logger["failed"],
            "errors": logger["errors"]
        }
