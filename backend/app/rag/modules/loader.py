"""文档加载器 - 支持多种格式，带重试机制，扫描件 PDF 自动 OCR"""

import os
import time
import logging
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

log = logging.getLogger(__name__)

# 条件导入 OCR 依赖（服务器上可能未安装）
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# 加载状态跟踪（用于 get_load_report）
_load_state = {"errors": [], "loaded": [], "failed": []}

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

    支持格式：PDF（含扫描件 OCR）, DOCX, TXT, HTML, 图片
    特性：
    - 自动检测文件类型
    - 扫描件 PDF 自动切换 OCR 识别
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

    # OCR 检测阈值：平均每页少于此字符数则判定为扫描件
    OCR_TEXT_THRESHOLD = 50

    def __init__(self, knowledge_base_dir: str = "./knowledge_base"):
        """初始化加载器

        Args:
            knowledge_base_dir: 知识库根目录
        """
        self.knowledge_base_dir = knowledge_base_dir
        self._reset_logger()

    def _reset_logger(self):
        """重置日志"""
        _load_state["errors"] = []
        _load_state["loaded"] = []
        _load_state["failed"] = []

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

    @staticmethod
    def _ocr_pdf(file_path: str) -> List[Document]:
        """对扫描件 PDF 进行 OCR 识别

        使用 pdf2image 将每页转为图片，再用 pytesseract 识别文字。

        Args:
            file_path: PDF 文件路径

        Returns:
            Document 列表，每页一个 Document
        """
        if not OCR_AVAILABLE:
            raise ImportError(
                "OCR 依赖未安装，请执行: pip install pdf2image pytesseract "
                "并安装系统依赖: apt install poppler-utils tesseract-ocr tesseract-ocr-chi-sim"
            )

        log.info(f"[OCR] 开始识别扫描件: {os.path.basename(file_path)}")

        # PDF 转图片（每页一张）
        images = convert_from_path(file_path, dpi=200)
        docs = []

        for page_idx, image in enumerate(images):
            try:
                text = pytesseract.image_to_string(image, lang="chi_sim+eng")
                text = text.strip()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "page": page_idx,
                            "total_pages": len(images),
                            "ocr": True,
                        }
                    ))
            except Exception as e:
                log.warning(f"[OCR] 第 {page_idx + 1} 页识别失败: {e}")

        log.info(f"[OCR] 完成: {os.path.basename(file_path)}，"
                     f"共 {len(images)} 页，识别 {len(docs)} 页有内容")
        return docs

    @retry_on_failure(max_retries=MAX_RETRIES, delay=RETRY_DELAY)
    def load_single_file(self, file_path: str) -> List[Document]:
        """加载单个文件（带重试）

        PDF 文件自动检测：先尝试文字提取，若内容过少则切换 OCR。
        其他格式直接使用对应 Loader。

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

        # PDF 特殊处理：先尝试文字提取，扫描件自动切换 OCR
        if ext == ".pdf":
            docs = self._load_pdf_with_ocr_fallback(file_path)
        else:
            loader_class = self.LOADER_MAP[ext]
            loader = loader_class(file_path)
            docs = loader.load()

        _load_state["loaded"].append(file_path)
        return docs

    def _load_pdf_with_ocr_fallback(self, file_path: str) -> List[Document]:
        """加载 PDF，提取不足时自动切换 OCR

        Args:
            file_path: PDF 文件路径

        Returns:
            Document 列表
        """
        # 第一步：尝试 PyPDFLoader 文字提取
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            total_text = sum(len(d.page_content.strip()) for d in docs)
            num_pages = max(len(docs), 1)

            if total_text >= self.OCR_TEXT_THRESHOLD * num_pages:
                # 文字提取成功，内容充足
                log.info(f"[PDF] 文字提取成功: {os.path.basename(file_path)}，"
                            f"{num_pages} 页，{total_text} 字符")
                return docs

            log.info(f"[PDF] 文字提取内容不足 ({total_text} 字符/{num_pages} 页)，"
                        f"疑似扫描件，尝试 OCR: {os.path.basename(file_path)}")
        except Exception as e:
            log.warning(f"[PDF] 文字提取失败: {e}，尝试 OCR: {os.path.basename(file_path)}")

        # 第二步：OCR 识别
        try:
            docs = self._ocr_pdf(file_path)
            if docs:
                return docs
            log.warning(f"[OCR] 未识别到任何文字: {os.path.basename(file_path)}")
        except ImportError as e:
            log.error(f"[OCR] 依赖缺失: {e}")
        except Exception as e:
            log.error(f"[OCR] 识别失败: {e}")

        # 两步都失败，返回空列表
        return []

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
            _load_state["errors"].append(f"目录不存在，已创建: {directory}")
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
                    _load_state["failed"].append({
                        "file": file_path,
                        "error": str(e)
                    })
                    _load_state["errors"].append(
                        f"加载失败 [{filename}]: {str(e)}"
                    )

        return all_docs

    def get_load_report(self) -> dict:
        """获取加载报告

        Returns:
            包含加载统计信息的字典
        """
        return {
            "total_loaded": len(_load_state["loaded"]),
            "total_failed": len(_load_state["failed"]),
            "loaded_files": _load_state["loaded"],
            "failed_files": _load_state["failed"],
            "errors": _load_state["errors"]
        }
