from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = tf.keras.models.load_model('best_model.h5')

def preprocess_image(image):
    image = image.resize((150, 150))  # Адаптируйте под ваш датасет
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = int(np.argmax(predictions[0]))
    probabilities = predictions[0].tolist()
    return JSONResponse(content={"predicted_class": predicted_class, "probabilities": probabilities})

