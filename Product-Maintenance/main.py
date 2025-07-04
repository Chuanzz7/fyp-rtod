import asyncio
import re
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Body, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text, func, Integer, update, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from starlette.responses import JSONResponse

from Detection.processor.processorSingleImage import SingleImageProcessor
from config.database import get_db, engine
from models.models import Product, Base
from models.schemas import ProductOut, BatchProductLookupResponse, BatchProductLookupRequest, ProductLookupResponse

# Allow your frontend origin, or use ["*"] for any (dev only!)
origins = [
    "http://localhost:5173",
    # add more origins if needed, e.g. "http://127.0.0.1:5173"
]

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
# --- In-memory product update cache ---
product_cache = defaultdict(lambda: {"quantity": 0, "inventoryStatus": "out_of_stock"})
trackid_cache = {}
cache_lock = asyncio.Lock()
trackid_cache_lock = asyncio.Lock()

# Store last detected product_ids for "out_of_stock" logic
detected_product_ids = set()
LOW_STOCK_THRESHOLD = 1  # Define your low stock threshold
STARTED = False
started_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run at startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        asyncio.create_task(batch_commit_products())
        app.state.single_image_processor = SingleImageProcessor()

    yield
    # Run at shutdown (if needed)


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"],  # allow all headers
)

# Mount static files to serve uploaded images
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


async def batch_commit_products():
    global STARTED
    while True:
        async with started_lock:
            if not STARTED:
                await asyncio.sleep(1)
                continue  # Wait until started
        # If started, run your normal logic below
        await asyncio.sleep(1)
        async with cache_lock:
            # Copy for thread safety
            updates = product_cache.copy()
            product_cache.clear()
            detected_product_ids.clear()

        async with trackid_cache_lock:
            trackid_cache.clear()

        # New DB session for this flush
        async for db in get_db():
            print("run")
            # 1. Update detected products
            for product_id, info in updates.items():
                await db.execute(
                    update(Product)
                    .where(Product.id == product_id)
                    .values(quantity=info["quantity"], inventoryStatus=info["inventoryStatus"])
                )
            # 2. Set NOT detected products to 0/out_of_stock
            all_ids = set()
            result = await db.execute(select(Product.id))
            for pid, in result:
                all_ids.add(pid)
            undetected_ids = all_ids - set(updates.keys())
            if undetected_ids:
                await db.execute(
                    update(Product)
                    .where(Product.id.in_(undetected_ids))
                    .values(quantity=0, inventoryStatus="out_of_stock")
                )
            await db.commit()
            break  # Only want one DB instance per cycle


async def generate_product_code(db: AsyncSession) -> str:
    """Generate next product code with P prefix (P00001, P00002, etc.)"""
    # Get the highest existing code number
    result = await db.execute(
        select(func.max(func.substr(Product.code, 2).cast(Integer)))
        .where(Product.code.like('P%'))
    )
    max_num = result.scalar()

    if max_num is None:
        next_num = 1
    else:
        next_num = max_num + 1

    return f"P{next_num:05d}"


def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return the file path"""
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename

    # Save file
    with open(file_path, "wb") as buffer:
        content = file.file.read()
        buffer.write(content)

    return str(unique_filename)


@app.get("/api/products", response_model=list[ProductOut])
async def list_products(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Product).order_by(desc(Product.quantity)))
    return result.scalars().all()


@app.post("/api/products", response_model=ProductOut)
async def create_product(
        name: str = Form(...),
        description: str = Form(""),
        category: str = Form(""),
        price: float = Form(0.0),
        quantity: int = Form(0),
        ocr: str = Form(""),
        image: UploadFile = File(None),
        db: AsyncSession = Depends(get_db)
):
    # Handle image upload
    image_filename = "product-placeholder.svg"  # default
    if image:
        image_filename = save_uploaded_file(image)

    # Generate product code
    product_code = await generate_product_code(db)

    ocr_normalized = re.sub(r'\s+', '', ocr).lower()

    # Determine inventory status at backend
    if quantity > LOW_STOCK_THRESHOLD:
        inventory_status = "in_stock"
    elif LOW_STOCK_THRESHOLD == quantity:
        inventory_status = "low_stock"
    else:
        inventory_status = "out_of_stock"

    # Create product data
    product_data = {
        "code": product_code,
        "name": name,
        "description": description,
        "category": category,
        "price": price,
        "quantity": quantity,
        "image": image_filename,
        "ocr": ocr,
        "ocr_normalized": ocr_normalized,
        "inventoryStatus": inventory_status
    }

    db_product = Product(**product_data)
    db.add(db_product)
    await db.commit()
    await db.refresh(db_product)
    return db_product


@app.put("/api/products/{product_id}", response_model=ProductOut)
async def update_product(
        product_id: int,
        name: str = Form(None),
        description: str = Form(None),
        category: str = Form(None),
        price: float = Form(None),
        quantity: int = Form(None),
        ocr: str = Form(None),
        image: UploadFile = File(None),
        db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Product).where(Product.id == product_id))
    db_product = result.scalar_one_or_none()
    if not db_product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Update fields that are provided
    update_data = {}
    if name is not None:
        update_data["name"] = name
    if description is not None:
        update_data["description"] = description
    if category is not None:
        update_data["category"] = category
    if price is not None:
        update_data["price"] = price
    if quantity is not None:
        update_data["quantity"] = quantity
        # --- Backend logic for inventory status ---
        if quantity > LOW_STOCK_THRESHOLD:
            update_data["inventoryStatus"] = "in_stock"
        elif LOW_STOCK_THRESHOLD <= quantity <= LOW_STOCK_THRESHOLD:
            update_data["inventoryStatus"] = "low_stock"
        else:
            update_data["inventoryStatus"] = "out_of_stock"
    if ocr is not None:
        update_data["ocr"] = ocr
        update_data["ocr_normalized"] = re.sub(r'\s+', '', ocr).lower()

    # Handle image upload
    if image:
        # Delete old image if it's not the placeholder
        if db_product.image != "product-placeholder.svg":
            old_image_path = UPLOAD_DIR / db_product.image
            if old_image_path.exists():
                old_image_path.unlink()

        # Save new image
        update_data["image"] = save_uploaded_file(image)

    # Apply updates
    for key, value in update_data.items():
        setattr(db_product, key, value)

    await db.commit()
    await db.refresh(db_product)
    return db_product


@app.delete("/api/products/{product_id}")
async def delete_product(product_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Product).where(Product.id == product_id))
    db_product = result.scalar_one_or_none()
    if not db_product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Delete associated image file if it's not the placeholder
    if db_product.image != "product-placeholder.svg":
        image_path = UPLOAD_DIR / db_product.image
        if image_path.exists():
            image_path.unlink()

    await db.delete(db_product)
    await db.commit()
    return {"detail": "Product deleted"}


@app.post("/api/product_lookup")
async def fuzzy_product_lookup(
        category: str = Body(...),
        ocr: str = Body(...),
        db: AsyncSession = Depends(get_db)
):
    clean_ocr = re.sub(r'\s+', '', ocr).lower()

    # The query is now much simpler and directly uses the indexed column
    query = text("""
                 SELECT *, similarity(ocr_normalized, :clean_ocr) AS sim
                 FROM products
                 WHERE ocr_normalized % :clean_ocr
                   AND category = :category
                 ORDER BY sim DESC
                     LIMIT 1
                 """)
    result = await db.execute(query, {"clean_ocr": clean_ocr, "category": category})
    row = result.first()

    if not row:
        raise HTTPException(status_code=404, detail="No similar product found")

    product = dict(row._mapping)
    product["confidence"] = float(product.pop("sim"))
    return product


@app.post("/api/product_lookup_batch", response_model=BatchProductLookupResponse)
async def batch_fuzzy_product_lookup(
        request: BatchProductLookupRequest,
        db: AsyncSession = Depends(get_db)
):
    results = []

    # Process each item in the batch
    for item in request.items:
        try:
            clean_ocr = re.sub(r'\s+', '', item.ocr).lower()

            # Same query as the original endpoint
            query = text("""
                         SELECT *, similarity(ocr_normalized, :clean_ocr) AS sim
                         FROM products
                         WHERE ocr_normalized % :clean_ocr
                           AND category = :category
                         ORDER BY sim DESC
                             LIMIT 1
                         """)
            result = await db.execute(query, {"clean_ocr": clean_ocr, "category": item.category})
            row = result.first()

            if row:
                product = dict(row._mapping)
                confidence = float(product.pop("sim"))

                # Create response with track_id mapping
                response_item = ProductLookupResponse(
                    track_id=item.track_id,
                    code=product.get("code"),
                    confidence=confidence,
                    category=product.get("category")
                    # Add other fields as needed: name=product.get("name"), etc.
                )
                results.append(response_item)

        except Exception as e:
            # Handle individual item errors without failing the entire batch
            print(f"Error processing item {item.track_id}: {e}")
            response_item = ProductLookupResponse(
                track_id=item.track_id,
                code=None,
                confidence=0.0,
                category=item.category
            )
            results.append(response_item)

    return BatchProductLookupResponse(results=results)


@app.post("/api/start_monitor")
async def start_monitor():
    global STARTED
    async with started_lock:
        if not STARTED:
            STARTED = True
            # Optionally start your background task here, if needed
            # e.g., asyncio.create_task(batch_commit_products())
        return {"status": "started"}


@app.post("/api/stop_monitor")
async def stop_monitor():
    global STARTED
    async with started_lock:
        if STARTED:
            STARTED = False
        return {"status": "stopped"}


@app.post("/api/product_lookup_batch_monitor")
async def batch_fuzzy_product_lookup(
        request: BatchProductLookupRequest = Body(...),
        db: AsyncSession = Depends(get_db),
):
    results = []
    items = request.items if request and request.items else []
    found_ids = set()

    for item in items:

        async with trackid_cache_lock:
            cached_result = trackid_cache.get(item.track_id)
        if cached_result:
            results.append(cached_result)
            continue  # Skip the DB query, go to next item

        try:
            clean_ocr = re.sub(r'\s+', '', item.ocr).lower()
            query = text("""
                         SELECT *, similarity(ocr_normalized, :clean_ocr) AS sim
                         FROM products
                         WHERE ocr_normalized % :clean_ocr
                  AND category = :category
                         ORDER BY sim DESC
                             LIMIT 1
                         """)
            result = await db.execute(query, {"clean_ocr": clean_ocr, "category": item.category})
            row = result.first()
            if row:
                product = dict(row._mapping)
                confidence = float(product.pop("sim"))
                product_id = product.get("id")
                found_ids.add(product_id)
                new_qty = getattr(item, "count", 1)
                if new_qty > LOW_STOCK_THRESHOLD:
                    status = "in_stock"
                elif new_qty == LOW_STOCK_THRESHOLD:
                    status = "low_stock"
                else:
                    status = "out_of_stock"
                async with cache_lock:
                    product_cache[product_id] = {"quantity": new_qty, "inventoryStatus": status}
                response_item = ProductLookupResponse(
                    track_id=item.track_id,
                    code=product.get("code"),
                    confidence=confidence,
                    category=product.get("category")
                )
            else:
                response_item = ProductLookupResponse(
                    track_id=item.track_id,
                    code=None,
                    confidence=0.0,
                    category=item.category
                )

            async with trackid_cache_lock:
                trackid_cache[item.track_id] = response_item
            results.append(response_item)
        except Exception as e:
            print(f"Error processing item {getattr(item, 'track_id', None)}: {e}")
            response_item = ProductLookupResponse(
                track_id=getattr(item, 'track_id', None),
                code=None,
                confidence=0.0,
                category=getattr(item, 'category', None)
            )
            results.append(response_item)
    # Update detected IDs for background task
    async with cache_lock:
        detected_product_ids.update(found_ids)
    return BatchProductLookupResponse(results=results)


@app.post("/api/detect_item")
async def upload_frame(image: UploadFile = File(...)):
    frame_bytes = await image.read()
    result = app.state.single_image_processor.process_image(
        image_input=frame_bytes,
        include_ocr=True,
        detection_threshold=0.8,
        ocr_threshold=0.5
    )
    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
