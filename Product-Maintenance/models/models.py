
# models.py
from sqlalchemy import String, Integer, Float
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(AsyncAttrs, DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = 'products'

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    code: Mapped[str] = mapped_column(String(10), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(String(1024), default="")
    category: Mapped[str] = mapped_column(String(64), default="")
    price: Mapped[float] = mapped_column(Float, default=0.0)
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    image: Mapped[str] = mapped_column(String(256), default="product-placeholder.svg")
    ocr: Mapped[str] = mapped_column(String(256), default="")
    ocr_normalized: Mapped[str] = mapped_column(String(1000), default="")
    inventoryStatus: Mapped[str] = mapped_column(String(256), default="INSTOCK")

