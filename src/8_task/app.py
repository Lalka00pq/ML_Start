# python
import asyncio
import sys

# 3rdparty
from fastapi import FastAPI

# project
from routers.api_info import router as InfoRouter
from routers.api_get_detector_classifier import router_detect as DetectorRouter
from routers.api_get_detector_classifier import router_classifier as ClassifierRouter
from routers.api_find_object import router as FindObject


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
app.include_router(DetectorRouter, prefix=api_v1_prefix)
app.include_router(ClassifierRouter, prefix=api_v1_prefix)
app.include_router(FindObject, prefix=api_v1_prefix)

app.docs_url = "/docs"
app.redoc_url = "/redocs"
app.setup()
