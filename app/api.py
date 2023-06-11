from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from service import get_cm, predictor, report, test_image
import uvicorn

# uvicorn api:app --host 0.0.0.0 --port 8000 --reload

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Image(BaseModel):
    uri: str


@app.post("/predict")
async def predict(image: Image):
    res = predictor(image.uri)
    return {"result": res}


@app.post("/test")
async def test(image: Image):
    i = test_image(image.uri)
    return {"image": i}


@app.get("/gestures")
async def get_file():
    return FileResponse("../store/full_gesture.jpg")


@app.get("/report")
async def get_report():
    res = report()
    cm = get_cm()
    return {"data": res, "cm": cm}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
