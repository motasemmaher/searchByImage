from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import PIL.Image as Image
from Train import startTrain
from LoadTest import getNameIamge
from fastapi.middleware.cors import CORSMiddleware
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:8100",
    "https://b2b-stg.herokuapp.com/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/start-training")
def read_root():
  startTrain()
  return {"finshed": True}

class ImageReader(BaseModel):
  img: str

@app.post('/get-image-name/{name}')
def getImageName(name: str, img: ImageReader):
  print('asddsadsadds')
  return {"name": getNameIamge(img, name)}


@app.post('/add-image/{category_name}/{name}')
def addImage(category_name: str, name: str, img: ImageReader):
  directoryName = f'images/{category_name}'
  if not os.path.exists(directoryName):
      os.mkdir(directoryName)
  path = f'{directoryName}/{name}'
  with open(path, 'wb') as f:
      f.write(img)
  return {"finshed": True}
