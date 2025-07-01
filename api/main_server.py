from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import keras
import requests

app = FastAPI()

endpoint = "http://localhost:8501/v1/models/potato_model:predict"  # TensorFlow Serving endpoint

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


def read_file_as_image(data) -> np.ndarray: # converts file into numpy array
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read()) # Allows a second request to be served while the first is being processed
    image_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": image_batch.tolist()
    }
    
    response = requests.post(endpoint, json=json_data)

    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

    prediction = response.json()["predictions"][0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)