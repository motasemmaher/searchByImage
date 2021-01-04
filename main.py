from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import PIL.Image as Image
from Train import startTrain
from LoadTest import getNameIamge
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:8100",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/start-training")
def read_root():
    startTrain()
    return {"finshed": True}


@app.get('/get-image-name/{name}')
def getImageName(name: str, img: bytes = File(...)):
    return {"name": getNameIamge(img, name)}


@app.post('/add-image/{category_name}/{name}')
def addImage(category_name: str, name: str, img: bytes = File(...)):
    directoryName = f'images/{category_name}'
    if not os.path.exists(directoryName):
        os.mkdir(directoryName)
    path = f'{directoryName}/{name}'
    with open(path, 'wb') as f:
        f.write(img)
    return {"finshed": True}
