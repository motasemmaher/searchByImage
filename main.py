from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import PIL.Image as Image
import os
from Train import startTrain
from LoadTest import getNameIamge

app = FastAPI()


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
