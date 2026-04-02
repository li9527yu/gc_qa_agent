# from typing import Optional, List, Dict, Any
# from pydantic import BaseModel, Field
# from typing import Union

# class DeleteRequest(BaseModel):
#     filenames: List[str]

#     class Config:
#         schema_extra = {
#             "example": {
#                 "filenames": [
#                     "text1.md",
#                     "test.txt"
#                 ]
#             }
#         }

# class DeletedFile(BaseModel):
#     filename: str
#     size_mb: float
#     modified_time: str


# class FailedFile(BaseModel):
#     filename: str
#     error: str

# class DeleteResponse(BaseModel):
#     status: str
#     message: str
#     deleted_files: List[DeletedFile]
#     failed_files: List[FailedFile]
#     total_deleted_size_mb: float
#     timestamp: str
#     task_id: str | None

#     class Config:
#         schema_extra = {
#             "example": {
#                 "status": "success",
#                 "message": "批量删除完成，成功删除 2 个文件，失败 1 个文件，知识库正在后台更新",
#                 "deleted_files": [
#                     {
#                         "filename": "file1.txt",
#                         "size_mb": 0.12,
#                         "modified_time": "2025-01-01T12:00:00"
#                     }
#                 ],
#                 "failed_files": [
#                     {
#                         "filename": "file2.pdf",
#                         "error": "文件不存在"
#                     }
#                 ],
#                 "total_deleted_size_mb": 0.12,
#                 "timestamp": "2025-01-01T12:00:01",
#                 "task_id": "c1f8e9a1-acde-4c3b-bb27-xxxx"
#             }
#         }

# class QueryRequest(BaseModel):
#     question: str = Field(..., description="用户问题", min_length=1, max_length=1000)
#     num_docs: int = Field(10, description="检索文档数量", ge=1, le=50)

# class QueryResponse(BaseModel):
#     answer: str
#     contexts: Optional[List[str]] = None
#     metadata: Optional[Dict[str, Any]] = None
#     success: bool = True
#     error_message: Optional[str] = None

# class MaterialItem(BaseModel):
#     id: Optional[int] = None
#     materialId: Optional[str] = None
#     accountId: Optional[str] = None
#     materialName: Optional[str] = None
#     materialCode: Optional[str] = None
#     checkState: Optional[int] = None
#     belongDataPool: Optional[int] = None
#     domain: Optional[str] = None
#     categoryOneLevel: Optional[str] = None
#     categoryTwoLevel: Optional[str] = None
#     categoryThreeLevel: Optional[str] = None
#     standardCategoryName: Optional[Union[str, None]] = None
#     standardCategoryCode: Optional[Union[str, None]] = None
#     standardCategoryUnit: Optional[Union[str, None]] = None
#     standardCategoryConfidence: Optional[Union[float, None]] = None
#     standardFeatures: Optional[Union[str, None]] = None
#     label: Optional[str] = None
#     price: Optional[float] = None
#     taxRate: Optional[float] = None
#     materialModelSpec: Optional[str] = None
#     releaseDepartment: Optional[str] = None
#     releaseTime: Optional[str] = None
#     releaseDate: Optional[Union[str, None]] = None
#     unit: Optional[str] = None
#     province: Optional[str] = None
#     city: Optional[str] = None
#     materialDescribe: Optional[str] = None
#     savePath: Optional[str] = None
#     createTime: Optional[str] = None
#     createTimeDate: Optional[str] = None
#     updateTime: Optional[str] = None
#     updateTimeDate: Optional[str] = None
#     categoryOneLevelName: Optional[Union[str, None]] = None
#     categoryTwoLevelName: Optional[Union[str, None]] = None
#     provinceId: Optional[str] = None
#     cityId: Optional[str] = None

# class DirectPriceQueryRequest(BaseModel):
#     datatype: str = Field(..., description="数据类型，如 informaterial 表示信息价")
#     list: List[MaterialItem] = Field(..., description="材料信息列表")
#     question: Optional[str] = Field(None, description="用户问题，用于提取额外的查询条件")

# class DirectPriceQueryResponse(BaseModel):
#     text: Optional[str] = None
#     intent: str = "price_recommendation"
#     channel: Optional[str] = None
#     metadata: Optional[Dict[str, Any]] = None
#     success: bool = True
#     error_message: Optional[str] = None
#     price_data: Optional[List[Dict[str, Any]]] = None