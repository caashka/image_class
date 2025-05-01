from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = tf.keras.models.load_model('cnn_model.h5')

def preprocess_image(image):
    image = image.convert("RGB")
    image = tf.image.resize(image, [32, 32])
    image = tf.reshape(image, (1, 32, 32, 3))
    image = np.array(image, dtype="float32") / 255.0
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    classes = ['Кот', 'Собака', 'Панда']
    predicted_class_num = int(np.argmax(predictions[0]))
    predicted_class = classes[predicted_class_num]
    probabilities = predictions[0].tolist()
    return JSONResponse(content={"predicted_class": predicted_class, "probabilities": probabilities})

