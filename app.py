from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import base64
from fastapi.middleware.cors import CORSMiddleware
import torch
from vit import VisionTransformer

app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://zakariabouchelaghm.github.io"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# define/import VisionTransformer here

BATCH_SIZE=128
EPOCHS=10
LEARNING_RATE= 3e-4
PATCH_SIZE=7
NUM_CLASSES=28
IMAGE_SIZE=32
CHANNELS=1
EMBED_DIM=256
NUM_HEADS=8
DEPTH=6
MLP_DIM=512
DROP_RATE=0.1

model=VisionTransformer(IMAGE_SIZE,PATCH_SIZE,CHANNELS,NUM_CLASSES,EMBED_DIM,NUM_HEADS,DEPTH,MLP_DIM,DROP_RATE)

model.load_state_dict(torch.load("vit_weights.pth", map_location="cpu"))
model.eval()


with torch.no_grad():
    input_tensor = torch.randn(1, 1, 32, 32)  # Example input tensor
    outputs = model(input_tensor)   # input_tensor shape: [B, C, H, W]
    predicted = outputs.argmax(dim=1)
print("Predicted labels:", predicted)



@app.post("/predict")

async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("L")

    # Preprocessing (same as before)
    img = np.array(image)
    img = cv2.resize(img, (32, 32))
    img = cv2.bitwise_not(img)
    img = img / 255.0
    img_input = np.array(img, dtype="float32")
    img_input = np.expand_dims(img, axis=-1)   # (32, 32, 1)
    img_input = np.expand_dims(img_input, axis=0)  # (1, 32, 32, 1)
    img_input = torch.from_numpy(img_input).permute(0, 3, 1, 2).float()
    img_input = (img_input - 0.5) / 0.5
    with torch.no_grad():
        outputs = model(img_input)   # input_tensor shape: [B, C, H, W]
        predicted = outputs.argmax(dim=1)
    return {"predicted_class": predicted.item()}
