# schemas.py
from typing import Optional, List

from pydantic import BaseModel


class ProductBase(BaseModel):
    name: str
    description: Optional[str] = ""
    category: Optional[str] = ""
    price: Optional[float] = 0.0
    quantity: Optional[int] = 0
    image: Optional[str] = "product-placeholder.svg"
    ocr: Optional[str] = ""
    inventoryStatus: Optional[str] = ""


class ProductCreate(ProductBase):
    pass


class ProductUpdate(ProductBase):
    pass


class ProductOut(ProductBase):
    id: int
    code: str
    ocr_normalized: str

    class Config:
        from_attributes = True  # <--- NEW (Pydantic v2+)


class ProductLookupItem(BaseModel):
    track_id: int
    category: str
    ocr: str


class BatchProductLookupRequest(BaseModel):
    items: Optional[List[ProductLookupItem]] = []

class ProductLookupResponse(BaseModel):
    track_id: int
    code: Optional[str] = None
    confidence: float = 0.0
    category: Optional[str] = None
    # Add other product fields as needed


class BatchProductLookupResponse(BaseModel):
    results: List[ProductLookupResponse]
