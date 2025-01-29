# python
import asyncio
import sys

# 3rdparty
from fastapi import FastAPI

# project
from routers.api_info import router as InfoRouter
from routers.api_resize_image import router as ResizeImageRouter

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI(
    title="Detection and Classification Ships/Planes API",
    version="0.1.0",
    description="",
    docs_url=None,
    redoc_url=None,
)
api_v1_prefix = ""

app.include_router(InfoRouter, prefix=api_v1_prefix)
app.include_router(ResizeImageRouter, prefix=api_v1_prefix)

app.docs_url = "/docs"
app.redoc_url = "/redocs"
app.setup()
