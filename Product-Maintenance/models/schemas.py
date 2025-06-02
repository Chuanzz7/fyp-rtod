# schemas.py
from typing import Optional

from pydantic import BaseModel


class ProductBase(BaseModel):
    name: str
    description: Optional[str] = ""
    inventoryStatus: Optional[str] = "INSTOCK"
    category: Optional[str] = ""
    price: Optional[float] = 0.0
    quantity: Optional[int] = 0
    image: Optional[str] = "product-placeholder.svg"


class ProductCreate(ProductBase):
    pass


class ProductUpdate(ProductBase):
    pass


class ProductOut(ProductBase):
    id: int

    class Config:
        from_attributes = True  # <--- NEW (Pydantic v2+)
