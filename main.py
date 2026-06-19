import logging

from fastapi import FastAPI
from fastapi import  status, Body
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from service.model import GroqProvider, openai_client
from service.preprocessing import preprocess_text


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):    
    logger.info("✅ LLMs Ready")

    yield
    logger.info("Server disconnected ❌")


app = FastAPI(lifespan=lifespan,
                title="EOL - Ease Of Learning",
                version="0.1.0")


@app.get("/")
def welcom():
    return{"message": "Welcom To Text Sentiment Analysis"}

@app.post("/sentiment")
async def analyze_text(text: str = Body(..., embed=True)):
    try :
        # 1. تنظيف النص
        clean_text = preprocess_text(text)
        sentiment_score = await openai_client.generate_text(clean_text)
        if sentiment_score is None:
            raise ValueError("Model failed to return a score")
   
        label = "Violence" if sentiment_score > 0.5 else "Non-Violence"

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
               "TEXT": text,
               "Cleand_Text": clean_text,
               "Sentiment_Label": sentiment_score,
               "label": label
            }
        )

    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": "TEXT_FAILD"
            }
        )
    

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)