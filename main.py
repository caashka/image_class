from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2

app = FastAPI()
model = tf.keras.models.load_model('best_model_for_user.h5')
input_shape = model.input_shape

if input_shape and len(input_shape) == 4 and input_shape[1] is not None:
    target_h, target_w = input_shape[1], input_shape[2]
else:
    target_h, target_w = 64, 64  # Резервный размер (стандарт для EuroSAT)

def preprocess_image(img):
    image = np.array(img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (target_w, target_h))
    image = image.reshape((1, target_h, target_w, 3))
    image = image.astype("float32") / 255.0
    return image

@app.get("/")
def read_root():
    return {"message": "API is running successfully! Add /docs to the URL to test the /predict endpoint."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = int(np.argmax(predictions[0]))
    probabilities = predictions[0].tolist()
    return {"predicted_class": predicted_class, "probabilities": probabilities}
