from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests

app = FastAPI()

BASE_URL = "http://localhost:8501/v1/models"

CROP_MODELS = {
    "potato": ["Early Blight", "Late Blight", "Healthy", "Unknown"],
    "tomato": ["Bacterial Spot", "Early Blight", "Healthy", "Late Blight", "Septoria Leaf Spot", "Yellow Leaf Curl Virus", "Leaf Mold", "Spider mites two spotted spider mite", "Target spot", "Mosaic virus", "Unknown"],
    "pepperbell": ["Bacterial spot", "Healthy", "Unknown"],
    "strawberry": ["Healthy", "Leaf Scorch","angular leafspot", "leaf spot","powdery mildew leaf", "Unknown"],
    "peach": ["Bacterial Spot", "Healthy", "peach leaf curl", "Unknown"],
    "apple": ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy", "Unknown"],
    "grape": ["Black Rot", "Esca (Black Measles)", "Healthy", "Leaf Blight", "Unknown"],
    "cherry": ["Healthy", "Powdery Mildew", "Unknown"],
    "corn": ["Cercospora Leaf Spot", "Common Rust", "Healthy", "Northern Leaf Blight", "Unknown"]
}

def read_file_as_image(data) -> np.ndarray: # converts file into numpy array
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict/{crop_name}")
async def predict(
    crop_name: str,
    file: UploadFile = File(...)
):
    if crop_name not in CROP_MODELS:
        return {"error": f"Model for crop '{crop_name}' not available."}
    
    image = read_file_as_image(await file.read()) # Allows a second request to be served while the first is being processed
    image_batch = np.expand_dims(image, 0) 

    json_data = {
        "instances": image_batch.tolist()
    }
    
    endpoint = f"{BASE_URL}/{crop_name}_model:predict"
    response = requests.post(endpoint, json=json_data)

    try:
        prediction = response.json()["predictions"][0]
    except Exception as e:
        return {"error": str(e), "details": response.json()}

    class_names = CROP_MODELS[crop_name]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "crop": crop_name,
        "class": predicted_class,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)