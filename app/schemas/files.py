from typing import List, Optional
from pydantic import BaseModel

class UploadedFile(BaseModel):
    original_file: str
    markdown_file: str
    status: str
    error: Optional[str] = None

class UploadResponse(BaseModel):
    results: List[UploadedFile]
    task_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {"original_file": "/path/to/file1.txt", "markdown_file": "/path/to/file1.md", "status": "success"},
                    {"original_file": "/path/to/file2.pdf", "status": "error", "error": "文件处理失败"}
                ],
                "task_id": "c1f8e9a1-acde-4c3b-bb27-xxxx"
            }
        }
        
class DeleteRequest(BaseModel):
    filenames: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "filenames": ["text1.md", "test.txt"]
            }
        }

class DeletedFile(BaseModel):
    filename: str
    size_mb: float
    modified_time: str

class FailedFile(BaseModel):
    filename: str
    error: str

class DeleteResponse(BaseModel):
    status: str
    message: str
    deleted_files: List[DeletedFile]
    failed_files: List[FailedFile]
    total_deleted_size_mb: float
    timestamp: str
    task_id: Optional[str]
