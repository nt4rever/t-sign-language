import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from recognize_api import predictor, test_image, report, get_cm


# uvicorn api:app --host 0.0.0.0 --port 8000 --reload

class Image(BaseModel):
    uri: str
    l_h: int
    l_s: int
    l_v: int
    u_h: int
    u_s: int
    u_v: int


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(image: Image):
    res = predictor(image.uri, image.l_h, image.l_s,
                    image.l_v, image.u_h, image.u_s, image.u_v)
    return {"result": res}


@app.post("/test")
async def test(image: Image):
    i = test_image(image.uri, image.l_h, image.l_s,
                   image.l_v, image.u_h, image.u_s, image.u_v)
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