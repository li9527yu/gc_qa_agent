from typing import List
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException
import uuid
import shutil
from pathlib import Path
from markdownify import markdownify as md
from app.read_corpus import Reader
from app.config import DATA_PATH,ROOT_PATH

router = APIRouter()

# ---------------- 路径常量 ----------------
UPLOAD_DIR = Path(ROOT_PATH+"/app/dataset/uploads")
MARKDOWN_DIR = Path(DATA_PATH)

# ---------------- 工具函数 ----------------
def save_upload_files(files: List[UploadFile]) -> List[Path]:
    """保存多个上传文件到本地上传目录，返回保存后的路径列表"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files:
        # 安全检查：防止路径遍历和空文件名
        safe_name = Path(f.filename).name
        if ".." in safe_name or not safe_name:
            raise ValueError(f"非法文件名: {f.filename}")
        
        # 使用唯一前缀，防止并发请求覆盖同名文件
        unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
        file_path = UPLOAD_DIR / unique_name
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        paths.append(file_path)
    return paths


def convert_to_markdown(text: str, original_filename: str) -> Path:
    """与原函数保持一致"""
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(original_filename).stem
    md_path = MARKDOWN_DIR / f"{stem}_{uuid.uuid4().hex[:8]}.md"
    markdown_text = md(text, heading_style="ATX")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    return md_path


def process_uploaded_files(files: List[UploadFile]) -> List[dict]:
    """批量处理上传文件"""
    results = []
    saved_paths = save_upload_files(files)

    for file, saved_path in zip(files, saved_paths):
        try:
            reader = Reader(corpus_path=str(saved_path))
            # 处理 corpus 可能是字符串列表或字典列表的情况
            corpus_items = []
            for item in reader.corpus:
                if isinstance(item, dict):
                    corpus_items.append(item.get('page_content', ''))
                else:
                    corpus_items.append(str(item))
            text = "\n".join(corpus_items)
            md_path = convert_to_markdown(text, file.filename)
            results.append({
                "original_file": str(saved_path),
                "markdown_file": str(md_path),
                "status": "success"
            })
        except Exception as e:
            results.append({
                "original_file": str(saved_path),
                "markdown_file": "",
                "status": "error",
                "error": str(e)
            })
    return results