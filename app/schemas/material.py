from typing import Optional, Union
from pydantic import BaseModel

class MaterialItem(BaseModel):
    id: Optional[int] = None
    materialId: Optional[str] = None
    accountId: Optional[str] = None
    materialName: Optional[str] = None
    materialCode: Optional[str] = None
    checkState: Optional[int] = None
    belongDataPool: Optional[int] = None
    domain: Optional[str] = None
    categoryOneLevel: Optional[str] = None
    categoryTwoLevel: Optional[str] = None
    categoryThreeLevel: Optional[str] = None
    standardCategoryName: Optional[str] = None
    standardCategoryCode: Optional[str] = None
    standardCategoryUnit: Optional[str] = None
    standardCategoryConfidence: Optional[float] = None
    standardFeatures: Optional[str] = None
    label: Optional[str] = None
    price: Optional[float] = None
    taxRate: Optional[float] = None
    materialModelSpec: Optional[str] = None
    releaseDepartment: Optional[str] = None
    releaseTime: Optional[str] = None
    releaseDate: Optional[str] = None
    unit: Optional[str] = None
    province: Optional[str] = None
    city: Optional[str] = None
    materialDescribe: Optional[str] = None
    savePath: Optional[str] = None
    createTime: Optional[str] = None
    updateTime: Optional[str] = None
