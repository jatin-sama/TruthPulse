# main.py
from dotenv import load_dotenv
load_dotenv()

import asyncio
from fastapi import FastAPI
from bot_worker import run_bot

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    # run bot in background as a task
    loop = asyncio.get_event_loop()
    loop.create_task(run_bot())
