from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import  cv2

app = FastAPI()
model = tf.keras.models.load_model('best_model.h5')

def preprocess_image(image):
    # Преобразование в оттенки серого, если цветное изображение
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = np.mean(image, axis=2)

    # Изменение размера до 28x28
    image = tf.image.resize(tf.expand_dims(image, -1), [150, 150])

    # Нормализация значений пикселей [0, 1]
    image = image / 255.0

    # Добавление размерности батча
    image = tf.expand_dims(image, 0)

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

